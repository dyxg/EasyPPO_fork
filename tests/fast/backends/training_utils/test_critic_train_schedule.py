"""EasyPPO critic mini-batches: `get_critic_train_schedule` re-splits a rank's rollout shard."""

from argparse import Namespace

import pytest
import torch
import torch.distributed as dist

from miles.backends.training_utils.data.rollout import get_critic_train_schedule
from miles.backends.training_utils.parallel import GroupInfo, ParallelState, set_parallel_state


@pytest.fixture(scope="module", autouse=True)
def process_group(tmp_path_factory):
    if dist.is_initialized():
        yield
        return

    rendezvous = tmp_path_factory.mktemp("critic-train-schedule") / "process-group"
    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=0, world_size=1)
    try:
        yield
    finally:
        dist.destroy_process_group()


@pytest.fixture(autouse=True)
def _trivial_parallel_state() -> None:
    trivial_group = GroupInfo(rank=0, size=1, group=None)
    set_parallel_state(
        ParallelState(
            intra_dp=trivial_group,
            intra_dp_cp=trivial_group,
            cp=trivial_group,
            tp=trivial_group,
            pp=trivial_group,
            ep=trivial_group,
            etp=trivial_group,
            indep_dp=trivial_group,
        )
    )


def _rollout_data(total_lengths: list[int]) -> dict:
    return {"total_lengths": total_lengths, "sample_ids": list(range(len(total_lengths)))}


def _drain(data_iterator, num_microbatches: list[int]) -> list[list[list[int]]]:
    """Sample ids of every micro-batch, grouped by critic step."""
    [iterator] = data_iterator
    iterator.reset()
    return [[iterator.get_next(["sample_ids"])["sample_ids"] for _ in range(count)] for count in num_microbatches]


def test_static_schedule_visits_every_sample_once_in_equal_steps():
    args = Namespace(critic_num_steps_per_rollout=4, seed=1234, use_dynamic_batch_size=False, micro_batch_size=2)
    rollout_data = _rollout_data([10] * 16)

    data_iterator, num_microbatches, num_rollouts = get_critic_train_schedule(
        args, rollout_data, rollout_id=3, num_rollouts=[16]
    )

    assert num_microbatches == [2, 2, 2, 2]
    assert num_rollouts == [4, 4, 4, 4]
    steps = _drain(data_iterator, num_microbatches)
    assert all(len(micro_batch) == 2 for step in steps for micro_batch in step)
    visited = [i for step in steps for micro_batch in step for i in micro_batch]
    assert sorted(visited) == list(range(16))
    assert visited != list(range(16)), "critic mini-batches are shuffled"


def test_schedule_depends_on_the_rollout_but_is_reproducible():
    args = Namespace(critic_num_steps_per_rollout=4, seed=1234, use_dynamic_batch_size=False, micro_batch_size=1)

    def visited(rollout_id: int) -> list[int]:
        data_iterator, num_microbatches, _ = get_critic_train_schedule(
            args, _rollout_data([10] * 32), rollout_id=rollout_id, num_rollouts=[32]
        )
        return [i for step in _drain(data_iterator, num_microbatches) for micro_batch in step for i in micro_batch]

    assert visited(0) == visited(0)
    assert visited(0) != visited(1)


def test_dynamic_schedule_packs_each_step_under_the_token_budget(monkeypatch):
    monkeypatch.setattr(torch.cuda, "current_device", lambda: "cpu")
    args = Namespace(critic_num_steps_per_rollout=2, seed=7, use_dynamic_batch_size=True, max_tokens_per_gpu=100)
    total_lengths = [60, 60, 30, 30, 30, 30, 90, 10]

    data_iterator, num_microbatches, num_rollouts = get_critic_train_schedule(
        args, _rollout_data(total_lengths), rollout_id=0, num_rollouts=[8]
    )

    assert num_rollouts == [4, 4]
    steps = _drain(data_iterator, num_microbatches)
    assert [sum(len(micro_batch) for micro_batch in step) for step in steps] == [4, 4]
    assert sorted(i for step in steps for micro_batch in step for i in micro_batch) == list(range(8))


def test_step_count_must_divide_the_rollout_count():
    args = Namespace(critic_num_steps_per_rollout=4, seed=0, use_dynamic_batch_size=False, micro_batch_size=1)
    with pytest.raises(AssertionError, match="does not divide"):
        get_critic_train_schedule(args, _rollout_data([10] * 6), rollout_id=0, num_rollouts=[6])


@pytest.mark.parametrize("dp_size", [2, 4])
def test_full_history_shards_feed_existing_critic_minibatches(monkeypatch, dp_size):
    from types import SimpleNamespace

    from miles.backends.training_utils.data import rollout as scheduling
    from miles.ray.rollout.train_data_conversion import process_rollout_data_shard, split_full_history_shards
    from tests.fast.ray.rollout.test_full_history_shards import _args, _data

    monkeypatch.setattr(torch.cuda, "current_device", lambda: "cpu")
    # Exercise each advertised rank on CPU; the tiny episode lengths require
    # one micro-batch per critic step on every rank, so the MAX is identity.
    monkeypatch.setattr(
        scheduling.GeneralPGUtil, "create", lambda group: SimpleNamespace(all_reduce=lambda *a, **kw: None)
    )
    args = _args(seed=1234)
    actor, critic, spans = _data()
    _, shards = split_full_history_shards(args, actor, critic, spans, {"dp_size": dp_size})
    global_steps = [[], []]
    for rank, shard in enumerate(shards):
        state = SimpleNamespace(
            effective_dp=SimpleNamespace(rank=rank, size=dp_size, group=None), cp=SimpleNamespace(size=1), vpp_size=1
        )
        monkeypatch.setattr(scheduling, "get_parallel_state", lambda state=state: state)
        data = process_rollout_data_shard(args, shard)
        data["sample_ids"] = data["rollout_ids"]
        iterators, counts, rollouts = get_critic_train_schedule(args, data, rollout_id=3, num_rollouts=[8])
        assert rollouts == [4, 4]
        for step, batches in enumerate(_drain(iterators, counts)):
            ids = [i for batch in batches for i in batch]
            assert len(ids) == 4 // dp_size
            global_steps[step].extend(ids)
    assert all(len(step) == 4 for step in global_steps)
    assert sorted(global_steps[0] + global_steps[1]) == list(range(100, 108))


def test_shared_critic_keeps_sample_based_minibatches_even_with_rollout_ids(monkeypatch):
    monkeypatch.setattr(torch.cuda, "current_device", lambda: "cpu")
    args = Namespace(
        critic_context_mode="shared",
        critic_num_steps_per_rollout=2,
        seed=7,
        use_dynamic_batch_size=True,
        max_tokens_per_gpu=100,
    )
    data = _rollout_data([10] * 8)
    # Unequal sibling counts: adopting the upstream shared-mode episode scheduler
    # would give different membership here. Shared must still shuffle sample rows.
    data["rollout_ids"] = [0, 0, 0, 1, 2, 3, 3, 3]
    actual, counts, _ = get_critic_train_schedule(args, data, rollout_id=0, num_rollouts=[4])
    expected, expected_counts, _ = get_critic_train_schedule(
        args, _rollout_data([10] * 8), rollout_id=0, num_rollouts=[4]
    )
    assert _drain(actual, counts) == _drain(expected, expected_counts)

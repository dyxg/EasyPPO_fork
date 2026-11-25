"""Kill a producer after checkpointing, then recover candidates without trial logs."""

import asyncio
import multiprocessing
import os
import shutil
import signal
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest

from agentic_critic.terminal_env import Completion, TerminalTrace
from agentic_critic.trial_archive import TrialArchiveWriter
from miles.ray.rollout.output_snapshotter import _RolloutExecutorOutputSnapshotter
from miles.rollout.base_types import RolloutFnTrainInput
from miles.rollout.fully_async_data_buffer import DataBufferInput
from miles.rollout.fully_async_rollout import _RunningTask
from miles.utils.types import Sample
from tests.fast.rollout.test_fully_async_rollout import FakeDataSource, make_args, make_fn, make_group


def _args(root):
    return make_args(save=str(root), load=str(root), rollout_batch_size=1,
                     async_data_buffer_capacity_factor=4, terminal_archive_trials=True)


def _snapshotter():
    return _RolloutExecutorOutputSnapshotter(
        args=Namespace(ci_test=False, ci_inject_missing_prefetched_batch_bug=False)
    )


def _make_sample(index, group, root):
    trace = TerminalTrace()
    ids = np.arange(128, dtype=np.int32)[None, :].repeat(4, axis=0)
    logps = np.full((4, 128), -6.0, dtype=np.float32)
    trace.record([256], Completion([0, 0, 0, 0], [-6.0] * 4, "sample",
                                  topk_token_ids=ids, topk_log_probs=logps), kind="action")
    root.mkdir(parents=True)
    trace.write_trajectory([], root)
    assert not (root / "easyppo-topk.npz").exists()
    sample = Sample(index=index, group_index=group, prompt="task", response="sample",
                    reward=0.75, status=Sample.Status.COMPLETED,
                    metadata={"trial_dir": str(root)})
    for key, value in trace.sample_fields()[0].items():
        setattr(sample, key, value)
    return sample


async def _produce(root):
    with pytest.MonkeyPatch.context() as mp:
        fn = make_fn(mp, _args(root), FakeDataSource())
        buffered = [_make_sample(50 + i, 5, root / "local" / f"buffer-{i}") for i in range(2)]
        prefetched = [_make_sample(60 + i, 6, root / "local" / f"prefetch-{i}") for i in range(2)]
        await fn._output.put(DataBufferInput(prompt_group=make_group(5), group=buffered))
        running = make_group(7)
        running[0].rollout_topk_token_ids = buffered[0].rollout_topk_token_ids
        running[0].rollout_topk_log_probs = buffered[0].rollout_topk_log_probs
        fn._running_tasks.append(_RunningTask(prompt_group=running, task=asyncio.Future()))
        snapshot = _snapshotter()
        snapshot.capture(trainer_model_id=None, rollout_id=3, data=prefetched, metadata={})
        writer = TrialArchiveWriter(root / "archives", shard_trials=128, max_delay=60)
        for sample in buffered + prefetched:
            await writer.enqueue(sample.metadata["trial_dir"])
        # Save while archive tails are still pending. No graceful exit/flush.
        fn.save(root / "buffer")
        snapshot.save(root / "prefetch")
        os.kill(os.getpid(), signal.SIGKILL)


def _child(root):
    asyncio.run(_produce(root))


def _check(sample):
    np.testing.assert_array_equal(sample.rollout_topk_token_ids,
                                  np.arange(128, dtype=np.int32)[None, :].repeat(4, axis=0))
    np.testing.assert_array_equal(sample.rollout_topk_log_probs,
                                  np.full((4, 128), -6.0, dtype=np.float32))
    assert sample.rollout_topk_token_ids.dtype == np.int32
    assert sample.rollout_topk_log_probs.dtype == np.float32
    assert sample.loss_mask == [1] * 4
    assert sample.rollout_log_probs == [-6.0] * 4
    assert sample.reward == 0.75


async def _resume(root):
    with pytest.MonkeyPatch.context() as mp:
        fn = make_fn(mp, _args(root), FakeDataSource())
        fn.load(root / "buffer")
        [pending] = fn._retry_buffer
        assert [s.index for s in pending] == [70, 71]
        assert all(s.rollout_topk_token_ids is None and not s.response for s in pending)
        fn._worker = asyncio.create_task(asyncio.Event().wait())
        try:
            batch = await asyncio.wait_for(fn._drain(RolloutFnTrainInput(rollout_id=4, weight_version=1)), 5)
            assert [s.index for s in batch.samples[0]] == [50, 51]
            for s in batch.samples[0]:
                _check(s)
            assert fn._output.state_dict() == []
        finally:
            fn._worker.cancel()
            await asyncio.gather(fn._worker, return_exceptions=True)
        snapshot = _snapshotter()
        snapshot.load(root / "prefetch")
        replay = snapshot.get(trainer_model_id=None, rollout_id=3)
        assert [s.index for s in replay.data] == [60, 61]
        for s in replay.data:
            _check(s)
        with pytest.raises(AssertionError, match="already replayed"):
            snapshot.get(trainer_model_id=None, rollout_id=3)


def test_sigkill_resume_without_diagnostic_files(tmp_path: Path):
    process = multiprocessing.get_context("spawn").Process(target=_child, args=(tmp_path,))
    process.start()
    process.join(60)
    if process.is_alive():
        process.kill()
        process.join()
        pytest.fail("checkpoint producer timed out")
    assert process.exitcode == -signal.SIGKILL
    assert (tmp_path / "buffer/state.pt").is_file()
    assert (tmp_path / "prefetch/state.pt").is_file()
    shutil.rmtree(tmp_path / "local")
    shutil.rmtree(tmp_path / "archives", ignore_errors=True)
    asyncio.run(_resume(tmp_path))

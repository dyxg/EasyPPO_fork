from pathlib import Path

import pytest

from scripts.make_miles_terminal_data import task_records


def make_task(root: Path, name="task"):
    task = root / name
    (task / "environment").mkdir(parents=True)
    (task / "tests").mkdir()
    (task / "task.toml").write_text('version = "1.0"\n')
    (task / "instruction.md").write_text("Create the requested artifact.")
    (task / "environment" / "Dockerfile").write_text("FROM ubuntu:24.04\n")
    (task / "tests" / "test.sh").write_text("#!/bin/bash\necho 1 > /logs/verifier/reward.txt\n")
    return task


def test_harbor_records_are_sorted_portable_and_do_not_expose_tests(tmp_path):
    make_task(tmp_path, "b")
    make_task(tmp_path, "a")
    rows = task_records(tmp_path, dataset="tmax", split="train", limit=1)
    assert len(rows) == 1 and rows[0]["metadata"]["task_path"] == "a"
    assert rows[0]["prompt"] == "Create the requested artifact."
    assert rows[0]["metadata"]["agent_name"] == "mini-swe-agent"


def test_missing_task_and_instruction_fail(tmp_path):
    with pytest.raises(ValueError, match="No Harbor"):
        task_records(tmp_path, dataset="tmax", split="train")
    task = make_task(tmp_path)
    (task / "instruction.md").unlink()
    with pytest.raises(ValueError, match="instruction"):
        task_records(tmp_path, dataset="terminal-bench-2.1", split="test")


def test_common_root_keeps_train_and_eval_paths_distinct(tmp_path):
    make_task(tmp_path / "train", "same-name")
    make_task(tmp_path / "eval", "same-name")
    train = task_records(tmp_path / "train", dataset="tmax", split="train", path_root=tmp_path)
    test = task_records(tmp_path / "eval", dataset="terminal-bench-2.1", split="test", path_root=tmp_path)
    assert train[0]["metadata"]["task_path"] == "train/same-name"
    assert test[0]["metadata"]["task_path"] == "eval/same-name"

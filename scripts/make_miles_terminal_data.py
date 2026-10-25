"""Convert downloaded Harbor TMAX / Terminal-Bench 2.1 tasks to Miles JSONL.

Example:
    python scripts/make_miles_terminal_data.py --tasks-dir /data/TMax-15K-Harbor \
        --dataset tmax --out /data/tmax.jsonl

Download datasets with Harbor first (see examples/easyppo/TERMINAL.md). Task
paths are relative to --tasks-dir, which must also be mounted on rollout workers.
"""

import argparse
import json
from pathlib import Path


def task_records(
    tasks_dir: Path, *, dataset: str, split: str, limit: int | None = None, path_root: Path | None = None
) -> list[dict]:
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive")
    root = tasks_dir.resolve(strict=True)
    path_root = path_root.resolve(strict=True) if path_root is not None else root
    configs = sorted(root.rglob("task.toml"))
    if not configs:
        raise ValueError(f"No Harbor task.toml files under {root}")
    records = []
    for config in configs[:limit]:
        task = config.parent
        instruction = task / "instruction.md"
        if not instruction.is_file() or not instruction.read_text().strip():
            raise ValueError(f"Missing or empty instruction.md in {task}")
        if not (task / "environment").is_dir() or not (task / "tests").is_dir():
            raise ValueError(f"Expected Harbor environment/ and tests/ in {task}")
        path = task.relative_to(path_root).as_posix()
        records.append(
            {
                "prompt": instruction.read_text(),
                "metadata": {
                    "task_path": path,
                    "task_id": f"{dataset}/{path}",
                    "dataset": dataset,
                    "split": split,
                    "agent_name": "mini-swe-agent",
                },
            }
        )
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks-dir", type=Path, required=True)
    parser.add_argument("--dataset", choices=("tmax", "terminal-bench-2.1"), required=True)
    parser.add_argument("--split", help="defaults to train for TMAX, test for Terminal-Bench")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--path-root", type=Path, help="Common task root for mixed training/evaluation datasets")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    split = args.split or ("train" if args.dataset == "tmax" else "test")
    records = task_records(
        args.tasks_dir, dataset=args.dataset, split=split, limit=args.limit, path_root=args.path_root
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records))
    print(f"Wrote {len(records)} {args.dataset}/{split} tasks to {args.out}")


if __name__ == "__main__":
    main()

"""EasyPPO on Harbor terminal tasks using mini-swe-agent and compaction.

Reuses the Frontier-CS recipe's optimizer and parallelism settings; these are
starting settings, not tuned TMAX reproduction hyperparameters. Requires a
Miles GPU environment, terminal extras, Docker and downloaded Harbor tasks.

Args:
    --prompt-data: JSONL produced by scripts/make_miles_terminal_data.py.
    --tasks-dir: Root of Harbor tasks on every rollout worker.
    --trials-dir: Persistent Harbor logs/results directory.
    --compaction: summary (default), keep_last_turns, or none.
    --compact-threshold: Context tokens at which compaction is requested.
    --critic-context-mode: shared or full_history (joint episode placement across DP ranks).
    Other checkpoint, batch and resource flags are inherited from ScriptArgs.

Example:
    python examples/easyppo/run_qwen3_5_9b_terminal.py \
        --prompt-data /data/tmax.jsonl --tasks-dir /data/TMax-15K-Harbor
"""

import shlex
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Literal

import typer

from miles.utils.external_utils import command_utils as U
from miles.utils.external_utils.model_args_utils import _import_module_from_path

# examples/ is not an installed package; resolve the shared recipe from this
# checkout, just as model variants load sibling definitions by file path.
_base = _import_module_from_path(Path(__file__).with_name("run_qwen3_5_9b_frontier_cs.py"), "easyppo_terminal_base")


@dataclass
class ScriptArgs(_base.ScriptArgs):
    run_id: str = field(default_factory=U.create_run_id)
    model_name: str = "Qwen3.5-9B"
    hf_repo: str = "Qwen/Qwen3.5-9B"
    agentic_generate_function_path: str = "agentic_critic.miles_terminal.generate"
    tasks_dir: str = ""
    trials_dir: str = "terminal-trials"
    compaction: Literal["summary", "keep_last_turns", "none"] = "summary"
    compact_threshold: int = 24576
    max_context_tokens: int = 32768
    max_new_tokens: int = 4096
    max_total_tokens: int = 65536
    max_steps: int = 100


def _validate(args: ScriptArgs):
    if not args.prompt_data or not args.tasks_dir:
        raise ValueError("Terminal training requires --prompt-data and --tasks-dir")
    if args.agentic_generate_function_path != "agentic_critic.miles_terminal.generate":
        raise ValueError("The terminal launcher requires agentic_critic.miles_terminal.generate")
    if args.custom_rm_path:
        raise ValueError("Harbor supplies the terminal reward; do not pass --custom-rm-path")


def prepare(args: ScriptArgs):
    _base.prepare(args)


def execute(args: ScriptArgs):
    _validate(args)
    terminal_args = shlex.join(
        [
            "--terminal-tasks-dir",
            args.tasks_dir,
            "--terminal-trials-dir",
            args.trials_dir,
            "--terminal-compaction",
            args.compaction,
            "--terminal-compact-threshold",
            str(args.compact_threshold),
            "--terminal-max-context-tokens",
            str(args.max_context_tokens),
            "--terminal-max-new-tokens",
            str(args.max_new_tokens),
            "--terminal-max-total-tokens",
            str(args.max_total_tokens),
            "--terminal-max-steps",
            str(args.max_steps),
            "--rollout-max-context-len",
            str(args.max_context_tokens),
        ]
    )
    # Explicit extra_args are last, following the base launcher's override order.
    _base.execute(replace(args, extra_args=f"{terminal_args} {args.extra_args}"))


@U.dataclass_cli
def main(args: ScriptArgs):
    _validate(args)
    prepare(args)
    execute(args)


if __name__ == "__main__":
    typer.run(main)

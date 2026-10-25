# TMAX and Terminal-Bench 2.1 on Miles

This recipe trains EasyPPO on Harbor tasks with a mini-swe-agent harness and
actor context compaction. TMAX supplies training environments; Terminal-Bench
2.1 can be evaluated through the same generator. No GPU training result is
claimed by this integration.

## Design and prior implementations

- The existing [Miles Harbor example](../swe-agent-harbor-docker/README.md)
  already delegates sandbox lifecycle and verification to Harbor, including
  mini-swe-agent selection. This recipe keeps that separation, but invokes
  Harbor `Trial.create()` / `Trial.run()` directly on each rollout worker;
  it does not require the external `miles_agent_server.py` fork.
- [TMAX](https://github.com/hamishivi/tmax) publishes self-contained Harbor
  tasks as `tmax/TMax-15K-Harbor`. Its original Vanillux2 harness is replaced
  here by mini-swe-agent; this is not an exact reproduction of TMAX's DPPO
  training recipe or published scores.
- [Terminal-Bench 2.1](https://github.com/harbor-framework/terminal-bench-2-1)
  is used in its original Harbor format. Its containers, task instructions,
  resource limits, timeouts and verifiers remain owned by Harbor.
- The harness subclasses [mini-swe-agent 1.17.1 DefaultAgent](https://github.com/SWE-agent/mini-swe-agent/blob/v1.17.1/src/minisweagent/agents/default.py).
  It reuses bash parsing, format recovery and submission detection. A small
  async loop and terminal-specific prompt replace the synchronous issue-solving
  loop, allowing native cancellation during model calls and sandbox commands.
  This is not Harbor's container-installed mini CLI, nor mini's default prompt.
- Segment and full-history critic mapping reuse this branch's
  [Frontier-CS implementation](../../agentic_critic/miles_fcs.py) and
  [critic_episode](../../agentic_critic/miles_samples.py).

The full task tree is available to Harbor, but the policy only sees the task
instruction and sandbox command outputs. Tests and verifier results are not
injected into the policy conversation. Verification runs once after the agent
finishes or reaches a budget; infrastructure/verifier errors raise errors
instead of becoming successful zero-reward rollouts.

## Install and prepare data

Use Python **3.12+**, the branch's Miles-compatible SGLang/Megatron environment,
and a working Docker daemon on rollout workers. The optional runtime versions
are pinned to Harbor 0.24.0 and mini-swe-agent 1.17.1:

```bash
python -m pip install -e '.[agentic,terminal]'

harbor datasets download 'tmax/TMax-15K-Harbor@latest' --output-dir /data/harbor
harbor datasets download 'terminal-bench/terminal-bench-2-1@latest' --output-dir /data/harbor

python scripts/make_miles_terminal_data.py \
  --tasks-dir /data/harbor/TMax-15K-Harbor --path-root /data/harbor \
  --dataset tmax --out /data/tmax.jsonl
python scripts/make_miles_terminal_data.py \
  --tasks-dir /data/harbor/terminal-bench-2-1 --path-root /data/harbor \
  --dataset terminal-bench-2.1 --out /data/tb21.jsonl
```

Harbor exports `<output-dir>/<dataset-name>/<task-name>/`. If a downloaded
release uses a different enclosing directory, use that directory for
`--tasks-dir`. Replace `@latest` with a fixed registry revision for experiments;
archive the exported tasks and Harbor's per-trial `lock.json` with results.
`--limit 8` on the converter is useful for a first run. The default split is
`train` for TMAX and `test` for TB 2.1; it does not randomly split tasks.

Both JSONLs use paths relative to the common `--path-root`, so training and
evaluation can share one `--terminal-tasks-dir`. All rollout workers must see
that task tree. Individual records may also contain an absolute `task_path`.
Do not feed TMAX's raw Apptainer `container.def` corpus to this converter: use
the published Harbor export. Grading still runs `tests/test.sh` through Harbor.

## Train and evaluate

```bash
python examples/easyppo/run_qwen3_5_9b_terminal.py \
  --prompt-data /data/tmax.jsonl --tasks-dir /data/harbor \
  --trials-dir /data/output/terminal-trials \
  --model-dir /data/models --data-dir /data --output-dir /data/output \
  --compaction summary --compact-threshold 24576 \
  --max-context-tokens 32768 --max-new-tokens 4096 --max-total-tokens 65536 \
  --critic-context-mode shared \
  --extra-args '--eval-prompt-data tb21 /data/tb21.jsonl --eval-interval 10 --n-samples-per-eval-prompt 1'
```

The launcher starts from `Qwen/Qwen3.5-9B` and reuses the existing EasyPPO
recipe's optimizer, warmup and parallelism defaults. These are starting
settings, not terminal-task-tuned hyperparameters. Use smaller rollout batches
and sample counts initially: each concurrent episode owns a sandbox.
The inherited `--rollout-max-response-len` (and eval request settings) is an
additional per-call upper bound; `--max-total-tokens` caps the whole episode,
including summaries. `--max-steps` bounds action attempts, including format
errors. Harbor's task timeout is an independent wall-clock ceiling.

`--critic-context-mode full_history` supports DP>1 by jointly placing complete
critic episodes and their actor segments. For example, 8 GPUs with TP=2 gives
DP=4. Its episode-aligned forward schedule uses one complete episode per
micro-batch on each rank, so uneven actor segment counts do not change the
number of collective calls. `shared` keeps its existing placement and schedule.

Full-history currently requires dynamic micro-batching, CP=1, VPP=1 and complete
logical batches. Global batch size must divide evenly across DP ranks; the total
episode count must divide evenly across DP ranks and critic mini-batch steps.
Each full critic episode and the sum of its actor segment lengths must fit
`--max-tokens-per-gpu`. Full-history sequences can be much larger than the actor
context; plan training token/memory budgets accordingly. No critic context is
silently truncated. Unsupported layouts fail before publishing training shards.

The joint placement follows
[actor-summary-compact at 72fa1ce](https://github.com/EasyPPO/EasyPPO/blob/72fa1ce/miles/ray/rollout/train_data_conversion.py).
Its global changes to shared-context sharding and critic mini-batch scheduling
were deliberately not imported. The local implementation also checks budgets
and handles multiple complete logical batches per rollout.

For another existing Miles launcher, use:

```text
--custom-generate-function-path agentic_critic.miles_terminal.generate
--input-key prompt --metadata-key metadata
--terminal-tasks-dir /data/harbor
--terminal-trials-dir /data/output/terminal-trials
```

Do not add `--apply-chat-template` or a custom reward function. Keep
`--use-dynamic-global-batch-size` off: a rollout may have multiple segments,
and the batch is counted in episodes. This initial adapter explicitly refuses
partial rollout, routing/indexer replay, sampling-support replay and top-k
logprob side channels. Use unbounded training sampling (`top_p=1`, `top_k=-1`).

`--terminal-environment docker` is the default. Harbor's `daytona` and `modal`
backends can also be selected after installing the corresponding Harbor extra
and configuring credentials; neither cloud backend has been exercised here.

## Compaction and training semantics

- `summary`: request an actor-generated summary, then restart from the original
  system/task messages and the summary. Summary generation is trainable and
  counts toward the episode budget. Its shell-looking content is never executed.
- `keep_last_turns`: retain the original task and the most recent complete
  assistant/observation turns (`--terminal-keep-last-turns`, default 2), without
  a summary call.
- `none`: run until submission or a context/token/step/wall-clock limit.

The threshold is lowered when necessary to reserve room for a full action or
summary. Oversized initial instructions or tool outputs can still exhaust the
context; the harness reports a limit instead of silently cropping tokens.
A summary that hits its generation limit is retained as training data and ends
the episode; it is not used to resume. A compaction that cannot shorten the
context also stops. Command output is bounded by `--terminal-max-output-chars`.

Every generation records the IDs and log probabilities returned by SGLang.
Only exact token-prefix continuations are merged into a segment. If a chat
template rewrites history (for example, removing old thinking), a new segment
is created automatically. Thus segment count can exceed compaction count.
Environment text and reused summaries have loss mask 0; newly sampled actions
and summaries have loss mask 1. All episode segments share the original
`rollout_id` and final verifier reward. Incomplete episodes mark all segments
`TRUNCATED` for EasyPPO's actor-only truncation filter.

Full-history critic inputs concatenate the exact actor segments in order,
including repeated task/summary prefixes, as in the existing Frontier-CS path.
Each `critic_span` maps a segment's response positions back to that input.

Episode options can be overridden with `metadata.terminal`, for example:

```json
{"max_steps": 50, "compaction": "keep_last_turns", "keep_last_turns": 3}
```

These replace launch defaults without mutating shared args. Unknown fields are
errors; per-episode context length cannot exceed Miles' launch context limit.
Sandbox backend and task/log roots are launch-only. The scalar reward key
(default `reward`) can be set with `--terminal-reward-key` or
`metadata.reward_key`; missing/nonfinite rewards fail explicitly.

Harbor writes results, verifier logs and its lock file under `--trials-dir`.
`agent/easyppo-trajectory.json` additionally stores token segments, call budgets,
compaction count and the final mini conversation, including partial traces on
cancellation. Sample metadata points to the trial directory.

## Validation scope

CPU tests use the real pinned mini parser and Harbor task/agent construction,
scripted model responses, fake command results, and Miles' real request payload
and Sample validation. They cover summary/retained-turn compaction, token/logprob
and critic-span alignment, format recovery, truncation, cancellation, reward
errors, data conversion, and launcher snapshots.

```bash
python -m pytest tests/agentic_critic/test_terminal_env.py \
  tests/agentic_critic/test_terminal_data.py \
  tests/agentic_critic/test_harbor_terminal.py \
  tests/agentic_critic/test_miles_terminal.py
python -m pytest tests/fast/launch_scripts tests/manual/launch_scripts
```

GPU rollout/training, actual task-container execution, and benchmark scores
remain to be validated. The implementation session had no Docker socket access.

Implementation-session results: 93 focused runtime/critic regression tests and
14 EasyPPO launcher/snapshot checks passed, as did Ruff and `git diff --check`.
The CPU environment contained an older upstream SGLang, so the focused tests
used `--confcutdir=tests/agentic_critic` and a temporary import of the real
`sglang-miles` `encoding_dsv4.py` (blob
`52c8bebd7716dd7f00771aed04c848e8cb307aa3`), without changing repository code.
Launcher checks used `--noconftest`. The full launcher suites had the same
65 failures and 52 errors on the unmodified base and this worktree, including
pre-existing `/root/models` permissions, unavailable fixture checkpoints and
legacy command utility API mismatches; those unrelated issues were not changed.

The full-history DP follow-up passed 145 CPU tests across joint placement,
existing scheduling/critic regressions and terminal integration. Coverage
includes DP=1/2/4, unequal segment counts, balanced and round-robin placement,
multiple complete batches, value-slice order, existing critic mini-batches and
unchanged shared-mode routing. These simulate DP layouts on CPU; they are not
multi-GPU training validation.

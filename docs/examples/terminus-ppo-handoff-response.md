---
title: "Terminal PPO on Modal — response to the handoff"
description: "Experiment decisions, code changes, B200 measurements and recovery checks for the Terminus PPO handoff."
---

**November 28, 2026** · Qwen3.5-9B · AgentPick training · Terminal-Bench 2.1 evaluation

Thank you for tracing the failure in run 008 to the 243,638-token critic input. Your distinction between a packing error and a measured GPU memory limit helped us decide how to proceed. We will start a fresh run from the pinned base model and apply the same trajectory eligibility rule to both critic modes. The single-B200 boundary test has now passed. The full-detach critic processed a 245,760-token trajectory and completed an optimizer step. [Original handoff](https://github.com/nanjiangwill/EasyPPO/blob/ppo-handoff/ppo-handoff.md)

## What we changed

The original investigation reviewed `137d57f76`, now `e92adf0a6` on the mainline after the co-author metadata update. We pushed two subsequent commits to `EasyPPO/EasyPPO:miles`.

| Area | Change made for the Terminus recipe | Commit |
|---|---|---|
| Packing and trajectory admission | Raised the default critic budget from **184,320 to 245,760 tokens (240 Ki)**. Both critic modes check each draw's complete chronological history. A draw above the limit is excluded from both actor and critic updates. Eligible siblings remain available. The rule covers new draws, restored queues and replayed prefetched batches. | [415085c5](https://github.com/EasyPPO/EasyPPO/commit/415085c5d3236b201e0b3a1d12bcdc041a6007bf) |
| Filter accounting | Added a running W&B count, `critic_overlength/dropped`, and an interval count, `rollout/fully_async/critic_overlength_dropped`. Each discarded draw counts once across its segments. Documented the rule in `Claude.md`. | [415085c5](https://github.com/EasyPPO/EasyPPO/commit/415085c5d3236b201e0b3a1d12bcdc041a6007bf) |
| PPO weighting | Disabled actor prompt-mean subtraction. Retained the critic's existing inverse-standard-deviation weighting and changed its default floor from 0.075 to the binary-reward formula. For a configured group size of 12, the floor is approximately 0.14434. | [415085c5](https://github.com/EasyPPO/EasyPPO/commit/415085c5d3236b201e0b3a1d12bcdc041a6007bf) |
| Score centering | Reduced the number of recorded candidates from 128 to 32 in both modes. The candidate ID and log-probability payload falls by 75% per recorded position. At comparable queued token counts, we expect a substantially smaller rollout-state file. | [9c9e943a](https://github.com/EasyPPO/EasyPPO/commit/9c9e943a7c528e1ae4e7510c8a1279d96c791731) |

We also tested the existing checkpoint and evaluation code. Those results appear in answers 14–15. The fresh start, diagnostic scope and acceptance criteria are experiment decisions. The single-GPU memory results are recorded below. End-to-end recovery on the two-node Modal configuration remains pending.

## B200 measurement at the 240 Ki training limit

The team ran the Qwen3.5-9B `full_detach` critic on **one B200 with TP=1** using one trajectory of **245,760 tokens**, with a longest segment of **32,000 tokens**. The critic completed the value forward pass, training forward pass, backward pass and one optimizer step.

| Phase | Sampled peak device memory (GiB) | Peak PyTorch active memory (GiB) |
|---|---:|---:|
| Value forward | 89.16 | 66.98 |
| Training forward | 134.17 | 123.23 |
| Backward | **173.71** | **161.17** |
| Optimizer step | 53.93 | 48.69 |

Backward produced the highest sampled device usage. The device reported **178.36 GiB** of capacity, leaving **4.65 GiB** at that sampled peak for this trajectory. During backward, the activations retained across layers overlap in memory with the MLP gradient buffers. Compared with the previous 256 Ki test, the layer-boundary activations across 32 layers occupied about **4 GiB less** in total. The SwiGLU buffer allocation that exhausted memory at 256 Ki fell from **12 GiB to 11.25 GiB** and succeeded at 240 Ki.

## Responses to the 17 questions

The numbers and questions follow the original handoff.

### Intended objective and length semantics

**1.** Is the intended `full_detach` experiment required to retain every otherwise valid trajectory within the generated-token budget, irrespective of chronological critic length?

**Decision.** Keep a draw at exactly 245,760 chronological tokens. Above that limit, exclude the entire draw from both training updates. We accept that filtering changes the training distribution and will report how often it occurs.

**2.** Is `critic_max_tokens_per_gpu` purely a resource/packing limit, or an intended dataset-admission rule? Where should that contract be documented?

**Implemented.** We chose the packing budget based on GPU memory. For this experiment, the configured critic budget also sets the per-draw training admission limit. The separate episode budget remains 102,400 model-generated tokens, including summaries. [Terminus recipe documentation](https://github.com/EasyPPO/EasyPPO/blob/9c9e943a7c528e1ae4e7510c8a1279d96c791731/Claude.md#L137)

**3.** Was 184,320 chosen from measured Qwen3.5-9B forward/backward memory, a bound on possible histories, or an estimate? Is there an existing validated long-history benchmark we should reproduce?

**Measured at the current boundary.** We initially chose 240 Ki from estimated B200 memory headroom. The single-GPU test above now provides a concrete benchmark at exactly 245,760 tokens with a 32,000-token longest segment. The original handoff gives no corresponding hardware measurement for the earlier 184,320-token setting.

**4.** Is a separate chronological-history or environment-output limit intended? If so, what should happen at the boundary: task termination, bootstrapped truncation, changed reward, or something else?

**Decision.** Use the existing episode termination and scoring rules. Full chronological history includes prompts and tool observations as well as generated tokens. The 240 Ki rule applies at training admission. It introduces no additional generation cutoff, reward adjustment or bootstrapped truncation.

**5.** Must `full_detach` and `actor-summary` see the same task/trajectory eligibility policy for the intended comparison? What rejection rate, if any, is scientifically acceptable?

**Implemented.** Use the complete chronological history for admission in both modes, including when every compressed actor-summary segment fits individually. Please report the count and fraction of candidate draws rejected in each arm. We expect rejections in the long tail. The frequency is unmeasured, and we have not set a numerical acceptance threshold.

**6.** When infrastructure failures leave fewer than 12 siblings, are current prompt-mean/variance statistics and fixed-size critic mini-batches the intended estimator? Does backfilling from other prompts need additional weighting or accounting?

**Implemented and checked.** Actor prompt-mean subtraction is off. For the critic, compute the empirical population standard deviation over the valid draws for each prompt. Use the EasyPPO floor ε = Δ/(2√n), where Δ is the gap between the binary reward values and n is the configured group size. We checked the scoring scripts for all 802 tasks in the pinned AgentPick training release. They produce 0 or 1, so Δ = 1. If 10 of 12 draws survive, use those 10 to calculate the empirical standard deviation and retain the floor of 1/(2√12). Keep the existing fixed-size batch backfill. We introduced no extra weighting for missing siblings. [EasyPPO, Appendix B.3](https://arxiv.org/pdf/2609.36802#page=23)

### Processing complete histories

**7.** Can a long episode run alone with a larger measured packing budget while preserving the current batch/loss normalization? Are there independent position, kernel or workspace limits beyond the packer?

**Decision and hardware result.** Keep automatic packing and allow a long eligible episode to occupy one microbatch. The single-trajectory B200 test completed backward and an optimizer step at the exact admission limit. Keep the logical batch, number of critic updates and loss normalization as configured. The measurement covers the segment layout reported above.

**8.** The existing [detached-memory implementation](https://github.com/EasyPPO/EasyPPO/blob/e92adf0a61c4eafdc94cff5654c13cdb39508714/miles_plugins/models/detached_memory.py) already processes attention segments with detached prefix K/V and carries detached recurrent state. What is the actual peak-memory bottleneck: full-sequence activations, prefix K/V, recurrent state, temporary copies, top-k data, or something else?

**Measured.** Backward was the limiting phase on this input. Saved layer activations were still live when the MLP gradient buffers were allocated. The phase-by-phase measurements and the SwiGLU allocation sizes from the 240 Ki and 256 Ki tests are reported above.

**9.** If further streaming or rematerialization is needed, what reference outputs and gradients should it match? Which segment boundaries must retain versus stop gradients? Resetting history independently per chunk would change `full_detach` semantics.

**Decision.** Keep the current `full_detach` implementation. If memory optimization becomes necessary, match its values, loss and gradients within numerical tolerance. Preserve positions, masks, cross-segment prefix K/V, recurrent state, normalization and the existing gradient-detachment boundaries.

**10.** Is context/tensor parallelism a supported alternative for this exact path? Current [Qwen detached attention](https://github.com/EasyPPO/EasyPPO/blob/e92adf0a61c4eafdc94cff5654c13cdb39508714/miles_plugins/models/qwen3_5_memory.py) explicitly requires context-parallel size 1, so we should not assume a topology change is immediately supported.

**Decision.** Use TP=1 and CP=1 for this run. The current detached-attention path requires CP=1. We have not validated an alternative tensor-parallel configuration for this experiment.

### Resume and checkpoint correctness

**11.** Can checkpoint 1 from run 008 be resumed across the terminator and trace-validation changes, or should it be used only after a queue compatibility audit? How should incompatible completed trajectories be handled without an undocumented sampling change?

**Decision.** Start from the pinned base model. Run 008 checkpoint 1 and its completed queue will not be used. A compatibility audit of that queue is unnecessary for this launch.

**12.** Which invariants must match between uninterrupted and resumed execution: actor/critic weights, optimizer states, RNG, policy versions/staleness, data-source position, in-flight work and completed queue contents?

**Recovery contract.** Resume actor and critic weights, both optimizers and learning-rate schedulers, saved RNG state, policy-version and staleness metadata, the data-source cursor and the completed queue from the same published joint checkpoint. Preserve the identities and stored order of completed draws. Prompts in flight at preemption may be sampled again, and their subsequent completion order may differ.

**13.** Is the approximately 118.1 GB rollout state expected at the current prefetch/top-k settings? Which fields can be encoded more compactly without changing probability values, sample identities or future consumption?

**Implemented.** At comparable queued token counts, we expect the saved rollout state to be substantially smaller with top-32. Reducing the candidate set from 128 to 32 cuts the stored candidate IDs and log-probabilities from 1,024 to 256 bytes per recorded position (75%). The reduction in total file size depends on the share of those fields in the old 118.1 GB file. We have not measured that share. Please record the new rollout-state file size and the number of queued tokens for comparison. Sampling and the sampled token's log-probability use the existing path. The smaller candidate set can change the centering estimate, and its effect on training remains to be measured.

**14.** Are the CPU-offloaded optimizer and asynchronous checkpoint writers isolated from subsequent parameter/optimizer mutations in every supported lifecycle? Which preemption windows should we inject faults into?

**Tested locally.** With CPU optimizer offload enabled, the async driver allowed the next rollout to overlap writing and waited for both writers before the next training update. An injected writer error prevented publication and the next update. In a separate small-model CPU test, we sent a real `SIGKILL` in three windows. Recovery selected the previous checkpoint when only the actor tracker had advanced, the previous checkpoint when both model trackers had advanced ahead of the joint marker, and the new checkpoint after joint publication. Following resume, model weights, Adam states, schedulers, RNG and inference outputs matched uninterrupted execution exactly.

The Modal recipe saves each batch and retains three complete sets. Local tests retained the latest K complete sets for K=1, 2 and 3. With K=3, we also killed the process during pruning, just after publication of checkpoint 4 and removal of checkpoint 1's actor directory. Recovery loaded checkpoint 4. After saving checkpoint 5, checkpoints 3–5 were readable. The old critic and rollout directories from checkpoint 1 still occupied disk. Two-node Modal preemption with the actual Qwen model remains to be validated. [Async driver](https://github.com/EasyPPO/EasyPPO/blob/9c9e943a7c528e1ae4e7510c8a1279d96c791731/train_async.py#L83-L86) · [Joint publication and pruning](https://github.com/EasyPPO/EasyPPO/blob/9c9e943a7c528e1ae4e7510c8a1279d96c791731/miles/backends/training_utils/checkpoint/joint_checkpoint.py#L88-L116)

### Failure accounting and observability

**15.** Should infrastructure-failed evaluation trials score zero, be retried, or be reported outside task-quality metrics? The present policy should be agreed independently of the training crash.

**Decision and test result.** Count a final infrastructure failure as zero in TB2.1 mean@5, retain it in the denominator, and report it separately by exception type. In run 007, two of 445 trials (89 tasks × 5) ended in verifier timeouts, or about 0.45%. We accept this scoring convention for the experiment. In a local test with three successes, one genuine task failure and one verifier timeout, the evaluation code returned 3/5 = 0.60 and counted one infrastructure failure. Splitting a successful trajectory into two segments preserved the denominator of five. The training buffer discarded the infrastructure-failed draw and kept the other four. [Run 007](https://forge.coreweave.com/wandb/nan-playground/miles-terminus_ppo/runs/tb21-9b-full-detach-2n-b200-007)

**16.** Which logs must survive hard preemption? Is retaining the driver traceback plus trial archives sufficient, or should we persist bounded per-worker Ray logs as well?

**Decision.** Use the existing driver traceback and trial archives for this run. We have dropped the proposal to archive individual Ray worker logs because of the additional collection and storage overhead. A hard preemption can therefore lose the unflushed tail of a worker's local log.

**17.** What evidence would you consider sufficient before calling the run healthy: critic-only updates, an actor update after warmup, post-update inference, a resumed optimizer step, and/or an evaluation cycle?

**Agreed acceptance criteria.** We would consider the run healthy once all five checks pass on the target configuration:

1. Critic parameters update during warmup.
2. Actor parameters update after warmup.
3. After weight publication, the inference engines use the updated policy and generate normally.
4. Training resumes from a joint checkpoint created by the new run and completes the next optimizer update.
5. A full TB2.1 evaluation finishes.

The CPU recovery tests cover part of this sequence. End-to-end B200 validation remains open.

## Follow-up on the proposed work

Thank you for profiling the 240 Ki boundary. The fresh run can now focus on observed rejection rates, the transition from critic warmup to actor training, and recovery on the target Modal configuration.

| Original proposal | Agreed action or evidence still needed |
|---|---|
| Add telemetry | Use the implemented counters to report rejection rates for each arm. If more instrumentation is added, check that enabling it preserves admitted sample IDs and training inputs. |
| Profile long histories on B200 | The single-trajectory capacity test at 245,760 tokens passed through the optimizer step. Explicit gradient-finiteness and fixed-input value-alignment checks were not reported in this measurement. |
| Reduce memory inside detached attention | Defer until profiling identifies a need. Answer 9 gives the reference for a future implementation. |
| Audit restored queues | Start fresh. Check completed sample IDs, queue state, cursor and policy versions when resuming a checkpoint from the new run. |
| Set a chronological cap or filter long draws | The shared training filter has been pushed. Measure its frequency in each arm and revisit the threshold if the observed rates affect the comparison. |
| Exercise preemption recovery and retain diagnostics | Use the driver traceback and trial archives. On Modal, exercise commit/reload, weight publication and resume, then verify the next optimizer update and the policy used for inference. |

For follow-up, please record the rejection count and denominator for each arm as it runs, together with run or log references for the five checks in answer 17.

## Appendix — original investigation

The following sections preserve the investigating team's November 27 record at `137d57f76`. References here to the current budget, the local workaround, its counter names and the status of training refer to that historical snapshot. The two commits and the responses above describe the subsequent decisions and changes.

### Experiment and run status at the time of the report


The intended experiment is Qwen3.5-9B asynchronous PPO on AgentPick training tasks, evaluated on Terminal-Bench 2.1, starting with `full_detach` and eventually comparing it with `actor-summary`. The authoritative configuration is [the Terminus recipe](https://github.com/EasyPPO/EasyPPO/blob/e92adf0a61c4eafdc94cff5654c13cdb39508714/examples/easyppo/terminus_ppo.py), [the Modal configuration](https://github.com/EasyPPO/EasyPPO/blob/e92adf0a61c4eafdc94cff5654c13cdb39508714/modal_launchers/terminal_config.py), and [Claude.md](https://github.com/EasyPPO/EasyPPO/blob/e92adf0a61c4eafdc94cff5654c13cdb39508714/Claude.md).

The investigated run used:

- Two nodes with eight B200 GPUs each, with normal preemption. Six trainer GPUs shared by actor and critic through offloading, nine rollout GPUs, one evaluation GPU.
- The pinned Qwen3.5-9B base model and converted Megatron checkpoint; 802 prepared training tasks and 89 evaluation tasks.
- 32 prompts × 12 trajectories per training batch; 150 planned batches; the first 30 batches critic-only; four critic updates per batch.
- Separate v2 Volumes for model/data/trial diagnostics and checkpoints/evaluation snapshots. The Modal launcher publishes a checkpoint after each completed batch.

The latest actual run, identified here as **008**, used commit [`ef43481aa`](https://github.com/EasyPPO/EasyPPO/commit/ef43481aa5e0c6be550d98e7c1191d48553137cf). Genuine critic optimizer updates and checkpoint publication were observed. The last complete joint checkpoint is **1**. Collection of rollout **2** then failed with an oversized critic row. The run stopped on 2026-11-26 at 20:51 UTC. It did not reach the actor-training phase or establish completion of the full experiment.

No training has run with the local filtering patch described below. Earlier model-size and recipe changes were separate experiment changes, not evidence that every restart had the same bug.

### Current failure and evidence

The production exception was:

```text
ValueError: Training row of 243638 tokens exceeds --critic-max-tokens-per-gpu=153600
```

It arose in `RolloutExecutor.get()` during training-data conversion, at `_pack_shard_steps` in [train_data_conversion.py](https://github.com/EasyPPO/EasyPPO/blob/e92adf0a61c4eafdc94cff5654c13cdb39508714/miles/ray/rollout/train_data_conversion.py). This is an explicit packing-limit check. **The exception does not establish that a 243,638-token critic forward/backward actually runs out of B200 memory.** That workload has not yet been profiled through training.

There are three different lengths in this experiment:

| Quantity | Meaning in the reviewed recipe |
|---|---|
| Generated-token budget | 102,400 model-generated tokens across the episode, including summaries. |
| Actor context | At most 32,768 tokens at a time; compaction replaces older actor context with a summary. |
| Full-detach critic history | The chronological input, including prompts, generated tokens and tool/environment observations. Actor compaction does not impose a corresponding bound on this total. |

Commit [`5eb825294`](https://github.com/EasyPPO/EasyPPO/commit/5eb825294) raises the critic packing limit to **184,320** tokens. This is still below the production row of 243,638 tokens. The reviewed `Claude.md` still lists the former 153,600 value; the code is authoritative.

A synthetic integration reproduction on `137d57f76` produced:

```text
exact_history: true
generated_tokens: 102400
largest_actor_segment: 22650
critic_tokens: 294177
critic_budget: 184320
accepted_by_new_trace_validator: true
failure: Training row of 294177 tokens exceeds --critic-max-tokens-per-gpu=184320
```

This uses the repository's mock tokenizer and terminal-agent fixture, not a new Qwen-generated trajectory. Its purpose is to demonstrate that a valid exact history can pass producer validation and still fail real batch conversion under the recipe's limits. The independently observed 243,638-token production row establishes that this is not only a synthetic concern.

The relevant source path is:

1. [Terminal trace conversion](https://github.com/EasyPPO/EasyPPO/blob/e92adf0a61c4eafdc94cff5654c13cdb39508714/agentic_critic/miles_terminal.py), `_samples_from_trace`: requires exact history for full-detach training, but does not bound chronological input length.
2. [Critic construction](https://github.com/EasyPPO/EasyPPO/blob/e92adf0a61c4eafdc94cff5654c13cdb39508714/miles/ray/rollout/train_data_conversion.py), `_memory_critic_sample`: validates matching actor response tokens/masks and concatenates the explicit critic segments into a full episode.
3. The same file, `_pack_shard_steps`: rejects any individual row larger than its packing budget.
4. [Queue restore](https://github.com/EasyPPO/EasyPPO/blob/e92adf0a61c4eafdc94cff5654c13cdb39508714/miles/rollout/fully_async_data_buffer.py), `load_state_dict`: the reviewed upstream implementation directly restores completed entries. [Fully async rollout restore](https://github.com/EasyPPO/EasyPPO/blob/e92adf0a61c4eafdc94cff5654c13cdb39508714/miles/rollout/fully_async_rollout.py) does not regenerate those entries through the new producer validator.

**Resume implication:** fixing only new generation does not address already completed samples in the checkpoint queue. We have not established that the particular offending production row is in checkpoint 1. We have established that restored entries can bypass producer-side checks.

### What the local workaround does and why it is not an agreed solution

The local patch checks exact chronological segment lengths at buffer admission and checkpoint-queue restore. If a draw is too long, it discards that entire draw from both actor and critic training, retains usable sibling draws, and lets other draws fill the fixed-size batch. It shares budget resolution with the packer and emits rejection warnings and W&B counters:

- `sampling/critic_budget_rejected_draws`
- `rollout/fully_async/oversized_critic_draws_filtered`

It neither truncates a retained history nor assigns a fabricated zero reward. Nevertheless, it **excludes otherwise valid long trajectories from the training distribution**. Duration, task difficulty, tool verbosity and outcome could correlate with rejection. The magnitude and direction of the resulting effect have not been measured. Filtering only `full_detach` also changes the data used by that arm of a comparison with `actor-summary`.

The patch passed **340 targeted tests**, including fresh and restored versions of the 294,177-token reproduction, boundary lengths, valid-sibling preservation, batch backfilling and unchanged handling of other critic modes. These tests show that the workaround behaves as implemented. They do not demonstrate unbiased learning, equivalent gradients for the original experiment, or a successful full training run.

The earlier description of this patch as simply “fixed” was too strong. It is a tested crash-avoidance policy with an algorithmic consequence. The patch remains in the development working tree separately from this documentation branch; building from that working tree would include it unless it is removed or explicitly selected. No agreement to use it for the experiment is assumed here.

### Previous failures and what the fixes establish

| Issue | What was observed | Fix or current status | Integrity implications |
|---|---|---|---|
| Sandbox and container setup | Earlier launches encountered task-image failures and a Docker session-keyring quota after repeated launches. | The older Docker path was stress-tested after its runtime fix; the current recipe uses Harbor Modal sandboxes. Task-image failures remain a distinct per-task failure category. | Execution failures must remain distinguishable from genuine task outcomes. Discarding failed draws or scoring failed evaluation trials is itself a policy to review. |
| Storage growth and supervisor lifetime | Excessive diagnostic file creation stressed the earlier storage arrangement. A separate CPU supervisor could tear down GPU work when interrupted. | v2 Volumes, bounded local trial staging, immutable diagnostic archives and GPU-job ownership of sandboxes were adopted. Relevant history includes [`0a13fb746`](https://github.com/EasyPPO/EasyPPO/commit/0a13fb746) and [`c20e12622`](https://github.com/EasyPPO/EasyPPO/commit/c20e12622). | Operational fixes should preserve training samples and checkpoint state. Diagnostic archiving is not permission to discard training-side top-k data. |
| Checkpoint Volume reload race | An audit-log append overlapped a checkpoint-volume reload and raised `FileNotFoundError`. | [`c20e12622`](https://github.com/EasyPPO/EasyPPO/commit/c20e12622) coordinates readers/writers and reloads, commits across nodes, and protects evaluation reads. Included upstream through [`1a0eb3109`](https://github.com/EasyPPO/EasyPPO/commit/d28b3fc3db8c406bac49800d763d695695169f38). | Intended to preserve computation while fixing synchronization. Real two-node storage tests and later production checkpoint continuation supplied evidence. |
| Actor values lost after offloading | During critic-only warmup, actor wake-up restored allocations without restoring actor parameter values. Affected run 007 actor checkpoints contained zero weights; subsequent rollout inference became nonsensical. | [`ef43481aa`](https://github.com/EasyPPO/EasyPPO/commit/ef43481aa) restores the canonical actor backup on wake-up. Included upstream. | This was a genuine model-correctness bug. Do not resume affected 007 actor checkpoints. For 008 checkpoint 0, all 403 actor tensors, covering 8,953,803,288 values, matched the base model bitwise, as expected during critic-only warmup. |
| Qwen turn terminators and exact history | Reconstructing conversation prefixes could lose sampled turn-ending tokens, compromising exact critic-history reconstruction. | [`5eb825294`](https://github.com/EasyPPO/EasyPPO/commit/5eb825294) preserves terminators; [`484d592b4`](https://github.com/EasyPPO/EasyPPO/commit/c6cb8a312336167e85a36ff0cdd42668ed3ab174) isolates invalid traces. | Restoring token fidelity is a correctness fix. Rejecting malformed traces does not solve the separate issue of valid oversized histories; completed old queues need a deliberate compatibility policy. |
| Exception propagation and teardown | Producer failures could be obscured by secondary errors; cleanup could continue long after training failed; W&B could show a misleading final state. | Changes in `c20e12622` preserve the original failure, cancel pending evaluation on failure and propagate nonzero finish status. [`c220ff056`](https://github.com/EasyPPO/EasyPPO/commit/ea05fdd93b0e2eb40aa0417aa88a0f88fb43b0e0) logs the driver traceback before teardown. | Improves failure visibility and lifecycle behavior. It does not repair already corrupted training data or weights. |
| Evaluation visibility and checkpoint latency | Snapshot visibility can lag between mounts; checkpoint writing is substantial for this model and rollout state. | [`44b296db4`](https://github.com/EasyPPO/EasyPPO/commit/e78b8f8a94a622dbe69cd20584dfac90c68ecef8) retries visibility and drains concurrent loads. [`33799be97`](https://github.com/EasyPPO/EasyPPO/commit/894932df45e64ab1f156161c532dd09733cf3345) and [`137d57f76`](https://github.com/EasyPPO/EasyPPO/commit/e92adf0a61c4eafdc94cff5654c13cdb39508714) enable/overlap asynchronous checkpoint writing before durable publication. | The reviewed changes have regression coverage. Their complete lifecycle under real preemption has not been validated in a new full training run. |

The run-008 validation demonstrated real critic updates, checkpoint publication, correct checkpoint-0 actor weights and sane inference on all ten rollout/evaluation engines before and after the first checkpoint. It did not establish all future training behavior. The later oversized-row exception was a different failure. Checkpoint 1 has consistent completion markers; the exhaustive actor comparison above was for checkpoint 0, not checkpoint 1.

### Logging and recovery observations

W&B console capture is enabled. Shared mode writes files named `output_<label>.log`; the failed run had a primary uploaded console file of about **119 MB**, and its console API returned captured logs. Four console files were also found on the persistent assets Volume. Thus the logs were not confined to a developer terminal or temporary GPU-container disk.

The fatal packing exception was absent from those W&B console files but retrievable from Modal application logs. In that run, final exception reporting happened after tracking shutdown. The current `c220ff056` change addresses that ordering. It still needs confirmation in the next live failure/smoke scenario.

Per-worker Ray files under `/tmp/ray` are ephemeral; the launcher does not archive that entire directory. Trial diagnostics are archived separately. Hard termination can lose the unflushed diagnostic tail, so W&B and Modal logs should be complemented by a bounded persistent diagnostic strategy if complete worker traces are required.

The saved `generate_rollout/state.pt` for checkpoint 1 is approximately **118.1 GB**. This is an observed artifact size, not proof that its contents are redundant. Top-k rollout records and prefetched samples may be needed for the estimator and faithful recovery. Its entire queue has not been loaded and audited for lengths/exactness in this investigation. Clearing it or removing probability records would require a separate correctness argument.

### Validation reported by the investigating team


- **1,319 targeted regression tests** passed on the reviewed upstream state. The separate oversized-trace reproduction still demonstrated the packing failure. Passing those regressions alone was insufficient for launch readiness.
- **340 targeted tests** passed with the local rejection workaround, including fresh/restore admission and real training-data conversion. This validates that policy's implementation, not the original experiment's statistical equivalence.
- The actor wake-up fix had **149 focused backend regression tests**, a GPU reproduction, and the production checkpoint-0 full actor comparison described above.
- Coordinated Volume reload was exercised on real two-node v2 mounts. A tiny CPU-model resume test matched the next optimizer step; it is not a substitute for a full Qwen resume test.
- Diagnostic test applications were stopped. Training remains stopped.

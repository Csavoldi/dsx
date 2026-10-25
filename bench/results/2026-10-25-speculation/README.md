# T16 desktop speculative decoding screen

The current desktop verifier offers limited throughput headroom before drafting, and prompt-lookup speculation regressed in this screen. No DCOCO drafter was ported or trained. The benchmark and its speculative sampling loop are implemented on `codex/t16-speculative`; the production server path is unchanged.

## Hardware and controls

Two RTX 3090s, main host link 24.6 GB/s and auxiliary link 3.3 GB/s, i7-12700K, 128 GB DDR4. GPU power limits remained 230 W each. CPU PL1 was capped at 125 W for the experiment and restored afterwards. Every model process ran under the existing 118 GB/no-swap scope, with temperature and available-memory guards.

T16 model and F16 Engram projections stayed fixed. Context allocation was 262144, chunk 8192, arena 110 GB, main/aux headroom 1.5/1 GB. All four October 25 prefill options were enabled. Adaptive expert placement was disabled, and GPU1 remained a fixed expert store. The initial controls held 1048 experts on GPU0 (8.9 GB) and 2705 on GPU1 (22.9 GB). These are fixed-placement diagnostics rather than an adaptive-cache serving claim.

## Forced-accept target verification

Each arm consumes the same 64 continuation tokens after a 3000-token code prompt. Two rounds alternate arm order. Reported TPS is total tokens divided by total elapsed time. Scalar `step` is width 0; verification widths count the consumed anchor plus draft positions. Timing includes output readback and `keep`, but excludes restoring the initial snapshot. Every proposed token is forced accepted, so this excludes drafting cost and rejection losses.

| Path | Deferral rank 3 | No deferral |
|---|---:|---:|
| Scalar decode | 40.70 TPS | 33.57 TPS |
| Verify 1 | 34.69 TPS | 29.24 TPS |
| Verify 2 | 38.18 TPS | 38.08 TPS |
| Verify 4 | 45.08 TPS | 45.23 TPS |
| Verify 8 | 51.53 TPS | 51.42 TPS |

The eight-token path reaches 1.27x the deployed-style scalar target before drafting. Its round costs approximately 155 ms; at a 40.70 TPS baseline it needs more than 6.32 committed tokens per round to win with zero draft cost, or 6.73 with an assumed 10 ms draft. The latter is an illustrative budget, not a measured DeepSeek drafter latency.

CPU expert busy time falls from 15.35 ms/token on scalar deferral-3 decode to 11.56 ms/token at width 8. Batching reuses some expert work, but it does not eliminate the union of experts selected by the block. CPU busy time overlaps GPU work; these counters do not establish an additive wall-time breakdown.

## Actual temperature-1 lookup speculation

A second code prompt extends the first fixture to 3060 tokens so rejected verification blocks cross the 128-position window boundary. It uses the final source revision `5e11a22`, temperature 1, top-p 1, two fresh sampling seeds, and 64 generated tokens per arm per seed. The proposer copies continuations from matching context suffixes; its cost and all verification/rejection work are included.

| Maximum consumed block | Output TPS | Accepted / proposed drafts |
|---|---:|---:|
| Scalar | 35.93 | — |
| 2 | 33.53 | 14 / 23 |
| 4 | 31.58 | 29 / 60 |
| 8 | 26.92 | 28 / 103 |

This is a short, two-seed diagnostic. Outputs differ between arms: speculative sampling consumes different random draws, and the verifier arithmetic also differs. It does not measure completion quality or establish a universal ceiling for neural drafting. On this second continuation, forced-accept width-8 verification measured 43.79 TPS versus 34.18 scalar TPS. Do not mix these fixture-specific rates with the first table or the 8K full-prompt prefill benchmark.

## State and target consistency

All 33 paired rejected-suffix comparisons were bit-identical across two unrelated suffixes, covering every kept-prefix length for widths 2, 4 and 8, and a following scalar step. There were no nonfinite logits. The final fixture additionally exercises the window wrap and compressor carry. This is useful evidence of rollback integrity for these tested histories; it is not exhaustive state verification.

Block verification does not reproduce the scalar target. In the first fixture, deferral-3 width-8 verification changed the greedy top token on 6/64 forced positions; even without deferral it changed 5/64. Deferral is one difference, and the independent batched arithmetic difference remains when deferral is disabled. This is not a completion-quality measurement. The lossless speculative-sampling theorem does not prove equivalence between two different target-logit implementations.

The sampler's six existing tests passed, including the statistical deterministic-proposal acceptance/residual test. `spec-bench --lookup` requires rollback checks and refuses scalar-path differences unless `--allow-path-differences` is explicitly supplied. That override was used only for the diagnostic table above. It does not enable any production behavior.

## DCOCO decision

[DCOCO](https://arxiv.org/html/2610.12570v1) improves acceptance by adding a learned contextual corrector to a parallel neural drafter. This experiment measures neither its acceptance nor its DeepSeek latency. The [released checkpoints](https://github.com/two-delta/DCOCO) target Qwen3-4B/8B, not DeepSeek-V4.1. Our pre-existing DSpark CPU reference is a different architecture and does not supply trained DCOCO correction weights.

Proceed first with a consistent, cheaper small-block target path. A candidate budget to test is at most 90 ms for an eight-token verification pass, at most 10 ms drafting, and at least five committed tokens per round: together those assumptions yield 50 TPS. No such gain is booked. A real GPU drafter must also account for experts displaced from GPU1's nearly full VRAM and measure a complete coding task with its current context and sampling settings.

## Receipts and recovery

`summary.json` records weighted aggregates, seeds, fixture and binary hashes, and the initial-revision caveat. Sanitized JSONL files retain all timing and rollback observations while omitting private continuation IDs and generated outputs. The first two controls use an uncommitted initial harness revision; v2 adds explicit CLI correctness gates outside the measured loops. The boundary/lookup run uses the committed v2 source. See [benchmark usage](../../../docs/speculative-t16.md).

The user's llama.cpp process on port 8080 was stopped by request. Its exact invocation and working directory are preserved at `/home/lxe/ai/ds41/codex-speculative-20261025/llama-before.json` and `restart-llama.sh` on `.102`. It was not restarted. Both GPU controllers completed, restored CPU power, and released `GPU_LOCK`; no GPU job or inference server was left running by this experiment.

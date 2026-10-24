# T16 correctness and oracle verification on host 102

October 24, 2026. Model: shipped T16 v1 GGUF, Q8_0 dense weights, IQ4_NL Engram table, F16 projections. No weight or CUDA-kernel change was made. This directory records validation of deployed af2c154 against the earlier f1f53e4 reference, and the cost of verification before implementing a GPU drafter.

## Default-build correctness

Both binaries evaluated the same first 1,024-token held-out sequence, producing 1,023 next-token predictions. Adaptation was disabled, CPU workers were set to eight, and both placements contained exactly 1,568 GPU0 experts and 2,794 GPU1 experts. Rank-3 deferral was enabled for the complete-logit comparison.

| Quantity | Reference | Current defaults |
|---|---:|---:|
| KL against teacher top-256 | 0.160876 | 0.160876 |
| Top-1 agreement | 0.832845 | 0.832845 |
| NLL | 2.312076 | 2.312076 |
| Raw logits | 1,023 × 129,280 F32 | 1,023 × 129,280 F32 |
| Changed F32 bit patterns | — | **0 / 132,253,440** |
| Maximum absolute logit difference | — | **0** |

Each 529,013,760-byte little-endian F32 output has SHA-256 `fbce05f045b49ad822fc54119f0b91178531a0333b2e4dcefcfb04fe4c1d54e2`. This clears the current default integration on this sample; it does not certify experimental flags or all possible inputs. In a separate no-deferral check, both unmodified binaries reported KL 0.137649, top-1 0.837732 and NLL 2.285205.

As a diagnostic control, enabling only `DSX_QB_ROPE=1` in the current build reproduced the historical failing arm exactly: KL **0.170395**, top-1 **0.822092**, NLL **2.318534** and perplexity **10.1608**. Thus the optional fused rotation is sufficient to reproduce the old shift in this setup. It remains disabled; no production setting was changed.

The complete outputs came from a CLI-only `DSX_EVAL_DUMP=PREFIX` hook, applied identically to both source revisions. It writes `PREFIX.seqN.f32` after model execution and timing. Instrumented binaries embedded the original deployed/reference CUDA fatbins byte-for-byte, avoiding a change of CUDA compiler as a confound. Their quality metrics also reproduced the unmodified binaries. The historical window-35 experimental arm remains a separate result.

## Oracle verification timing

The fixture is the original cube agent's 2,031-token prompt, rendered through `DeepseekV41Encoding` and the GGUF's embedded tokenizer. Context capacity is 131,072. GPU0/GPU1 resident counts are 1,239/2,757, matching production. T16 weights and placement ranking are fixed; `DSX_ADAPT=0` makes repeated paths comparable. Timing uses 14 CPU workers, 125 W CPU long-term cap, 180 W short-term cap and 230 W GPU caps. These settings differ from the earlier live Pi run, so use the within-window comparisons below.

One 128-token greedy continuation was generated with deferral off, then every arm consumed those exact IDs. Two rounds reverse path order. The forced-accept verifier calls `verify_spec` and `keep` and includes rewindable snapshot and commit costs. It assumes every proposed token is accepted and contains no draft-model computation. It is an upper bound, not a sampled speculative-generation benchmark.

| Path | Deferral off, tok/s | Rank 3 configured, tok/s |
|---|---:|---:|
| Ordinary `step` | 27.80 | **34.41** |
| Verify 1 position | 24.73 | 30.84 |
| Verify 2 positions | 30.99 | 30.95 |
| Verify 3 positions | 33.92 | 33.97 |
| Verify 4 positions | 36.60 | 36.61 |
| Verify 6 positions | 39.51 | 39.54 |
| Verify 8 positions | 42.53 | 42.52 |

Rates are total tokens divided by total time across both rounds. Multi-position verification currently bypasses expert deferral; configuring rank 3 therefore does not preserve the single-token execution semantics inside those blocks. Ordinary replay matches its serial reference exactly. Verification changed one top-1 position out of 128 against the no-deferral reference and five against rank 3 at widths 2–8. Probability-distribution differences require their own comparison; changed raw logits alone do not quantify a quality loss.

Rollback checks verify a known token followed by arbitrary suffixes and retain only that first token. Discarded suffix IDs 11 and 4242 produce bit-identical retained logits across eight successive positions, for widths 2, 4 and 8, in both deferral configurations. This is a bounded causal-state check, not exhaustive rollback coverage.

At rank 3, perfect width-4 verification has only 6.4% headroom over ordinary decode; width 8 has 23.6%. Under a simplified constant-conditional-acceptance model, they require acceptance above 95.9% and 93.8%, respectively, merely to break even before drafting, communication and lost target-expert residency. Widths 2/3 cannot break even even with perfect acceptance. At 90% conditional acceptance, width 8 would yield about 30.27 tok/s before those additional costs. Actual draft acceptance has not been measured.

## GPU1 capacity and production cache policy

GPU1 runs its resident experts locally; its x4 host link does not stream those weights every token. The bounded 64-token decode trace recorded **1,640,320 bytes of GPU1 transfer activity per output token**, with maximum individual transfer 20,480 bytes. This matches IDs, routing weights, activations and results. Kernel busy time was 3.189 ms per token, about 11.6% of the CUDA activity span. This is a time-union measure, not SM occupancy. GPU0 was busy in kernels for 16.111 ms per token. Copy records do not include mapped-memory accesses made within kernels.

The capacity probes moved target experts from GPU1 to RAM, without changing their T16 weights:

| GPU1 expert storage | Ordinary rank-3 decode | Perfect verify 4 | Perfect verify 8 | CPU pick share in ordinary decode |
|---|---:|---:|---:|---:|
| ~23.3 GB | 34.41 | 36.61 | 42.52 | 38.82% |
| 19.0 GB | 32.98 | 34.77 | 40.28 | 42.32% |
| 17.0 GB | 32.35 | 34.02 | 39.25 | 43.97% |

All rates are tok/s on the same 128 IDs, static placement, two reversed-order rounds. GPU0's measured resident count varied from 1239 to 1245/1246 in the reduced-GPU1 probes; these are approximate placement penalties, not perfectly frozen-GPU0 A/Bs. The reduced stores held 2243/2004 experts versus 2757 initially. Both alternatives retained all other experts in RAM (101.6/103.7 GB pinned, zero file-only experts). SSD expert reads were therefore unnecessary at these capacities. Moving more experts to the CPU under rank-3 deferral also changed the output distribution: mean full-vocabulary KL from the full-capacity serial reference was 0.00702/0.00774. These are distribution-drift measurements, not teacher KL or task-quality scores.

The production cache policy was measured separately on one forward-only stream of the same 128 IDs. With `DSX_ADAPT=8`, rank 3 and the 110 GB arena, it reached **39.29 tok/s**, with **25.03%** of picks computed on CPU and 1024 swaps. No output positions were replayed to warm that measurement. Its event-instrumented counterpart measured 37.73 tok/s and exposed about **5.11 ms/token CPU wait**, versus 10.63 ms in the static-cache profile. Attention, Engram and normalization accounted for 12.71 ms/token, including q projections 3.14 and output projections 5.46. Instrumentation changes timing and can perturb adaptive placement; the profiled rate is not an untraced speedup.

Cold static oracles alone consequently do not establish the best possible verification cost under a warmer cache. A matched warm-cache oracle is measured separately: exact-continuation warming, then an explicit benchmark-only freeze of placement before all timed arms. That experiment is optimistic future-informed placement, not a live-agent result.

| Matched warm, frozen placement | Ordinary rank-3 decode | Perfect verify 4 | Perfect verify 8 |
|---|---:|---:|---:|
| Two-round weighted rate, tok/s | **44.32** | **53.36** | **60.12** |
| CPU pick share | 15.69% | 15.65% | 15.65% |

The ordinary replay matches its regenerated frozen reference bit-for-bit. Both verifier widths change two top-1 positions out of 128 and have mean full-vocabulary KL 0.003074 from that reference. The freeze control flushes pending swaps and disables further adaptation only when explicitly invoked by the benchmark; it does not change normal serving.

This establishes headroom under optimistic cache placement and perfect acceptance, not a delivered agent speedup. At 90% constant conditional acceptance, the width-4 cycle has only **2.63 ms** available for all exposed drafting, communication and target-residency costs before losing to matched ordinary decode. Width 8 already loses at that acceptance without any added cost. Its zero-cost break-even conditional acceptance is about 91.1%; width 4's is 87.7%. Actual prefix-length distributions must replace this simple model in a real drafter experiment.

## Decision

Keep rank-3 deferral and adaptive expert caching as the reference. The default integration passes the bounded correctness check; the optional q_b rotation remains rejected. The verifier needs its own numerical/quality gate and sampled acceptance semantics before production use. A full GPU DSpark implementation is not justified by these receipts yet: cold placement leaves little headroom, and the warm-placement gain is sensitive to acceptance, draft overhead and displaced experts. The old negative result is therefore conditional on the execution path, not evidence that GPU speculation can never help this hardware.

Prioritize verifier equivalence and causal cache behavior. If speculation is pursued next, start with high-confidence repeated-code lookup, which adds little VRAM pressure, then measure a compact drafter's accepted-prefix distribution against the actual T16 target before building a full GPU draft pipeline. Do not book the warm 60.12 tok/s oracle as achievable production throughput.

## Verification distribution checks

With deferral disabled, width-4/8 verification still differed from ordinary decode: one changed top-1 position out of 128, and mean full-vocabulary KL from serial logits of **0.002707/0.002410**. With rank 3 as the serial reference, the corresponding means were about 0.005105, with five changed top-1 positions. These comparisons neither demonstrate a task-quality regression nor establish lossless sampled decoding. Verification numerics and the intended target semantics require a separate gate before a production speculative sampler is added.

## Harness and receipts

Build the validation tools with:

```sh
CUDA_ARCHS=86 cargo build --locked --release -p dsx --features cuda \
  --bin dsx --bin oracle-bench --bin validation-encode
```

`validation-encode MODEL.gguf REQUEST.json OUTPUT.ids` renders and tokenizes a saved text-only OpenAI request using the serving template and the model's embedded tokenizer. `oracle-bench --help` describes fixed-token, placement, rollback and profiler options. `--continuation FILE` keeps IDs identical across deferral or placement changes. `--dump-reference` and `--compare-reference` allow full-vocabulary distribution comparisons across placements. `--serial-probe` measures one forward-only stream under the selected adaptive policy. `--warm-cache-oracle` warms on the exact continuation, flushes swaps, freezes expert placement, and regenerates the serial reference before timing. `--trace-path step` or `--trace-path verify4`, with one round, brackets the selected timed path using CUDA profiler start/stop APIs; use Nsight capture-range `cudaProfilerApi` to exclude initialization. `bench/analyze_nsys.py` summarizes a SQLite export using interval unions rather than double-counting overlapping kernels.

These tools allocate the model and large pinned arenas. Stop serving only inside an owned, coordinated GPU-lock window, retain the existing watchdog and memory cap, and restore the original serving configuration and power settings afterward. The historical scripts are receipts, not an unattended job queue.

Raw artifacts remain under `/home/lxe/dsx-validation-20261024` locally and `/home/lxe/ai/ds41/codex-validation-20261024` on host 102: per-arm commands and environment, binary and original-kernel hashes, telemetry, complete quality logits, per-call verification times and rollback diagnostics. Controller startup errors and restoration events are retained in the event logs. The measured GPU1 capacities are feasible in RAM; using SSD instead would introduce a different I/O cost that was not benchmarked here.

[Machine-readable receipts](receipts.json) include source hashes, timing summaries, log hashes and restoration status. Final checks confirmed the original server executable MD5 `363ced3dada78d1271549592de700e3d`, healthy serving with zero pending requests, no GPU lock, both GPUs at 230 W and the captured CPU long-term setting restored. No thermal abort occurred. The saved Pi request counters also show zero prefix recomputation over all 35 follow-up requests, so prefix reprocessing is not an established bottleneck on that trace.

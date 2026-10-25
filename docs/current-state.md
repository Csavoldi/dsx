# NVIDIA runtime and measurement status

The default desktop model is T16 v1: DeepSeek-V4.1-Flash-1017 with T16 routed experts, Q8_0 dense weights and an
IQ4_NL Engram lookup table in one GGUF. The October 23, 2026 NVIDIA kernel integration is af2c154; 76a5d0a records
its serving configuration. This page separates that implementation from earlier model-release measurements and
unfinished proposals. Documentation changes after those commits do not imply a new measured runtime.

## Model package

| Component | Shipped representation |
|---|---|
| Routed experts | T16, types 1010/1011/1012; each matrix is `trellis \| suh \| svh` |
| Bit layout | gate/up/down 1.5/1.5/2 in layers 0–19; 2/2/2.5 in layers 20–39 |
| Quantizer | patched ExLlamaV3, Hadamard-128, column-major tile positions, sigma_reg=2 |
| Output gains | gate/up ×1.13 at K1.5 and ×1.04 at K2, baked into `svh`; chosen on the reported eval |
| Dense weights | Q8_0; separate F16 Engram projection matrices remain as shipped |
| Engram lookup table | IQ4_NL; the `q6k-engram` model branch provides Q6_K instead |
| Tokenizer | embedded in GGUF metadata |
| Vision | optional `vision.safetensors` beside the model |
| Placement ranking | `expert-rank.json`, passed through `--rank` |

The default GGUF is 251,592,261,632 bytes (~252 decimal GB). The Q6_K alternative is ~302 GB in four shards.
These file layouts were checked against the Hugging Face model API on October 24. The T16 model revision was
`8bf76fe667734aff5c4d7fcd88ed855a3191b244` at that check; future downloads can change independently of this repo.

The [format specification](t16-format.md) describes the v1 byte layout and a proposed container v2 with separate
scale tensors. The current loader and converter implement v1, not that proposal. T16 model release v2 refers to the
calib5 requant experiment, which is a separate version number. It completed but scored KL ~0.202 against v1's 0.196
with the same gains; it did not replace the default model. No improved agentic quality was established.

## Runtime defaults

`dsx-server` defaults to a 32K context, chunk size 4096, arena budget 0, and deferral off. Its two-3090 quick start
explicitly selects 128K context, a 110 GB arena and `--defer-rank 3`. The recorded reference serving configuration
uses those values, `--main-gb=-1.5`, `--aux-gb=-1.0`, and no experimental environment overrides.

| Control | Default | Meaning |
|---|---|---|
| `DSX_T16_GEMV` | staged kernel; `0` selects old kernel | cp.async T16 expert GEMV |
| `DSX_Q8R1`, `DSX_SK1` | enabled; `0` disables | single-token Q8 and split-K variants |
| `DSX_COMBINE_FUSED`, `DSX_FUSE2`, `DSX_FUSE3` | enabled; `0` disables | combine, dense/shared and attention epilogue fusions |
| `DSX_AUX_LATE` | enabled; `0` disables | enqueue aux GPU after main GPU layer-0 attention |
| `DSX_QB_ROPE` | off; `1` enables | experimental fused q_b rotation; inexact on sm_86 |
| `DSX_PAR_ATTN`, `DSX_FUSE4` | off; `1` enables | parallel attention branch and closing hc_post fusion; below the gain threshold |
| `DSX_ADAPT_EARLY` | off; `1` enables | earlier cache update; measured worse residency |
| `DSX_ENGRAM_FP8` | off; `1` enables | convert F16 Engram **projections** to exact FP8 weights at load |
| `DSX_ENGRAM_PERSIST` | off; `1` enables | persistent lookup-row gather threads; marginal measured gain |
| `DSX_PREFILL_CPU` | enabled; `0` disables | CPU assistance for short chunks |
| `DSX_PREFILL_CPU_MAX_N` | 512 | maximum chunk size eligible for CPU assistance |
| `DSX_ADAPT` | 8 | maximum adaptive expert replacements per token; `0` fixes residency |
| `DSX_THINKING_BUDGET` | unset | experimental hard reasoning-token cap 1–8192 for NVIDIA thinking requests; forces a close token, intentionally changes generation, and requires task-quality validation |
| `DSX_REASONING_SCORE` | unset | experimental native thinking score 1–100; changes only the leading V4.1 reasoning header; thinking-off requests are unchanged |
| `DSX_ADAPT_STABLE` | off; `1` enables | explicit layer/expert ordering for equal-score adaptive cache choices; experimental reproducibility control |
| `DSX_T16_PAIR_GU` | off; `1` enables | experimental paired gate/up T16 launches; see [screening receipts](../bench/results/2026-10-24-gauntlet/README.md#paired-gateup-screening) |
| `DSX_GPU_SAMPLE` | off; `1` enables | experimental GPU categorical sampling at temperature 1 and top-p at least 1, with zero hesitation penalty; other settings and non-finite logits use CPU sampling |
| `DSX_RUNTIME_STATS`, `DSX_LOOKUP_AUDIT` | off; `1` enables | per-request timing and causal copy-opportunity diagnostics; neither executes speculative decoding |

The lookup table and the projection matrices are different weights. Keeping F16 projections does not mean the
252 GB package stores the large lookup table in F16: that table remains IQ4_NL.

The reference hardware has an RTX 3090 on a ~24.6 GB/s x16 host link and another on a ~3.3 GB/s x4 link. The x4
card keeps a fixed expert set. The main GPU adapts its cache, and the CPU computes its assigned experts from RAM
while the GPUs work. [Placement](placement.md) explains backing copies, capacities and multi-GPU controls.

## Measurement provenance

All numbers below are measurements from their named configuration, not promises for a fresh server or other hardware.
The reference machine is two RTX 3090s, an i7-12700K and 128 GB DDR4. Sizes are decimal GB.

| Result | Build and workload | Source |
|---|---|---|
| KL 0.196, top-1 80.9%, perplexity 9.54 | T16 v1, IQ4_NL lookup table, prefill-path Lucebox 8×1024, deferral off | [model card](../cards/dsx-t16/README.md#quality) |
| KL 0.194, top-1 81.4% | same expert build, Q6_K lookup table, prefill-path eval | model card |
| 34.9 tok/s; 36.8 with defer=3 | earlier T16 release build, free generation after 3K prompt, 300 new tokens; GPU limits 250/280 W | [model card speed table](../cards/dsx-t16/README.md#speed) |
| ~529 tok/s prefill | 3K prompt after vectorized T16 dequant and hugepage arena fixes | [FINDINGS §8d](../FINDINGS.md#8d-where-the-time-goes-on-102-2026-10-22-t16-iq4_nl-build) |
| 33.86 → 37.03 tok/s (+9.4%) | old base vs integrated kernels **with optional FP8 projections**, teacher-forced 1×1023 decode steps, defer=3, adapt=8; means of two interleaved rounds | [FINDINGS §9b](../FINDINGS.md#9b-decode-small-kernels-launch-structure-and-the-dev-4-merge-2026-10-23), windows 30–34 |
| 36.7 → 41.0 tok/s | free generation, 120 after 300 tokens; candidate also includes optional FP8 projections; different generated text | FINDINGS §9b |
| 31.63 tok/s | final default af2c154, F16 projections, 128K context, 38-token cold serving reply | recorded serving smoke check; not a sustained A/B |

The integrated maximum above includes an option the default configuration leaves off. It is not a newly measured
37–41 tok/s guarantee for that default. A 128K context also leaves fewer experts on the main GPU than the 32K gen
benchmarks, and the adaptive cache takes time to follow a new conversation.

Free-running numeric changes alter generated text, routing and CPU pick share, so their tok/s cannot isolate a
kernel's speed. Use fixed teacher tokens, interleaved/order-swapped runs, matching power caps and repeat statistics.
Exactness comparisons additionally require `DSX_ADAPT=0` and equal resident expert counts. Changing GPU placement
can change rounding even when weight bytes are unchanged. Cooler 8-thread quality checks are not production-thread
timing measurements.

<a id="quality-and-the-open-decode-check"></a>

## Quality and decode validation

The published 0.196 quality figure is a prefill-path result with deferral off. Its output gains were selected on the
same eval, so it is not an independent quality estimate. Later decode checks use different sequence counts,
residency and deferral settings and must be reported separately.

A T16 fixed-token decode comparison on 2046 positions measured KL ~0.208 without deferral, 0.237 at rank 3,
0.260 at rank 2 and 0.306 at rank 1. Ranks 2/1 failed the additional +0.01 KL gate and remain rejected. Rank 3 was
retained for the reference serving configuration. Deferral keeps top-6 routing but injects low-ranked CPU outputs
one layer late, changing the residual computation; it is an approximation, even with exact expert selection.

[Window 35](../bench/results/2026-10-22-102/win35.log) fixed adaptation and equalized GPU0 residency at 1568 experts.
Base KL was 0.160876; the integrated arm was 0.170395, with top-1 0.832845 → 0.822092. FP8 and F16 projection arms
at equal residency gave the same result. Thus that window did not attribute the shift to FP8 storage.

The fused q_b rotation differed by one ulp in ~700/4096 rotated values on sm_86 despite a matching sm_120 bench.
It is disabled by default. The default-flags comparison was not completed before the final serving switch, so the
deployment report kept the KL caveat open.

The [October 24 validation](../bench/results/2026-10-24-102/README.md) subsequently compared complete logits from
the current defaults against f1f53e4, with adaptation disabled, rank-3 deferral and equal residency (GPU0 1568,
GPU1 2794). All **132,253,440 F32 values** matched bit-for-bit across 1023 positions and a 129280-token vocabulary;
both files have SHA-256 `fbce05f045b49ad822fc54119f0b91178531a0333b2e4dcefcfb04fe4c1d54e2`.
Both arms scored KL 0.160876 and top-1 0.832845. Separate unmodified-binary checks also matched without deferral:
KL 0.137649 and top-1 0.837732. These results clear the default integration on that sample, not every experimental
flag or all possible inputs. The older window-35 arm remains a distinct historical measurement.

A diagnostic control enabling only `DSX_QB_ROPE=1` reproduced window 35's KL 0.170395, top-1 0.822092 and
NLL 2.318534 exactly. The optional fused rotation is therefore sufficient to reproduce that shift in this setup;
the production default keeps it off.

The original FP8 projection kernel had a separate deterministic KL change from summation order. Its replacement,
`gemv_fp8b32_o`, preserves F16 arithmetic order; window 35 confirmed equal F16/FP8 KL at equal placement. That fix
is merged, but optional conversion remains off. Persistent gather threads removed the join wait but added only
~0.49% above FP8, near the kill threshold, so they remain off too.

## Recorded negative results

- Next-layer expert prefetch (`DSX_PREFETCH`) reduced CPU work but slowed decode: 33.1 → 25.3/22.2 tok/s at one/two
  copies per layer, or 37.4 → 28.3/20.7 with defer=3. DRAM reads, repack work, wasted copies and GPU0 contention
  outweighed the saved CPU work. [Budget analysis](budget-102.md#4-dsx_prefetch-against-this-budget-negative-result).
- Desktop DSpark verify passes of 2/3/4 tokens cost 51/69/77 ms versus ~33 ms for one plain step. The union of CPU
  experts grows with the window; the all-GPU AMD gains do not transfer to this desktop result.
- Early cache update worsened the CPU pick share from ~23% to ~26%. The proposed commit-timing diagnosis was not
  confirmed by its rework. The cause remains open and the option remains off.
- Forcing the x4 GPU adaptive can evict arena coverage onto disk and hurt prose speed. Decay/swap-count sweeps did
  not produce a useful consistent gain. The measured configuration keeps the x4 store fixed.

## Remaining work

The default-flags, fixed-residency comparison against the old base passed on the October 24 sample. Verification
still uses a different execution path from ordinary decode: multi-position blocks bypass deferral, and their
logits require separate correctness and quality checks. Oracle upper bounds and rollback diagnostics are recorded
in the October 24 results. Hardware scripts require a coordinated window and must not be run as unattended cleanup.

Subsequent proposals include profiling the adaptive-cache scan, combining T16 gate/up launch work and improving
multi-conversation prefix reuse. Lower-bit dense weights are estimates, not implemented gains. Requantization is
not currently scheduled. The gpu-unlock proposal was dropped. Old budget tables retain those proposals to explain
their assumptions, not to designate current jobs.

Before a benchmark, read actual power and thermal limits. The recorded October 23 window used 230 W GPU limits
and a 125/180 W CPU cap. An October 24 read-only check found the CPU long-term limit at ~4095.9 W, not 125 W, with
short-term still 180 W. The watchdog has previously stopped jobs at CPU ≥95°C. Large pinned allocations can also
kill serving through memory pressure. Do not assume old protection settings still apply.

## Documentation map

| Document | Scope |
|---|---|
| [README](../README.md) | download, quick start, supported hardware and API |
| [DESIGN](../DESIGN.md) | current NVIDIA execution and storage architecture |
| [placement](placement.md) | current residency accounting and tuning flags; dated older sweeps |
| [T16 format](t16-format.md) | shipped v1 bytes; clearly separated v2 proposal |
| [T16 experiments](t16.md) | chronological codebook, layout and real-expert research |
| [FINDINGS](../FINDINGS.md) | measured development history, with current summary and later integration |
| [budget](budget-102.md), [inflight analysis](t16-decode-inflight.md) | dated resource models and intermediate kernel profiles |
| [bench results](../bench/results/2026-10-22-102/README.md) | raw window scripts and receipts; not all large traces are in Git |
| [quantization tools](../quant/README.md) | current T16 entry points and historical scalar pipeline |
| [model cards](../cards/dsx-t16/README.md) | model-release metrics; local copies require separate publication to update the Hub |
| [AMD runner](../amd/README.md), [batched log](../amd/RUNNER.md) | separate HIP implementation and completed rental measurements |

Historical raw logs, third-party vendored docs and gauntlet task definitions are retained as records. They are not
rewritten to look like current measurements. Public model cards and GitHub files change only when these edits are
published; a local documentation commit does not update either service.

## Failure replay diagnostics

`oracle-bench --serial-probe --replay-sampling cpu|gpu --replay-seed N` records the initial prefill distribution and the distributions after a fixed continuation. It supports `--dump-reference` and `--compare-reference`, records sorted initial store membership and subsequent changes, and reports sampled choices. GPU replay explicitly downloads logits after sampling. Those readbacks and logs perturb execution timing; this mode is a correctness diagnostic, not production throughput evidence.

`DSX_ROUTE_AUDIT=PATH` writes buffered per-layer JSON records containing selected experts, routing weights and the CPU experts actually computed immediately or deferred. It is off by default. For compiler/kernel isolation, `DSX_KERNEL_IMAGE=/absolute/image.fatbin` selects an archived image at build time; kernel-dependent experimental flags must match the image. This retains the current host wrapper and therefore does not reproduce an entire archived serving binary. Ordinary builds compile the current CUDA sources as before.

`sampler-audit --logits FILE.f32 --seed N` measures actual GPU categorical weights against CPU FP64 weights on saved vocabulary rows. It reports KL, total variation, sampled choices and a continuous-uniform CDF interval disagreement estimate. Run it only inside an owned GPU window with serving stopped. The [failure-conditioned receipt](../bench/results/2026-10-24-gauntlet/fixed-failure-replay.json) separates this sampling diagnostic from model-logit comparisons and live task quality.

`DSX_TOKEN_AUDIT=PATH` optionally appends each request's raw prompt/sample token IDs, processed-token count, prefix-cache boundary and initial/final expert-store membership. The stopping or cancelled token can be sampled without entering KV state, so replays must honor `processed_output_tokens`. These receipts support text-only generation-history reconstruction; they do not include image contents. The runtime profile also records `prefix_replayed_tokens`, the matching prefix that had to be prefetched again after restoring an older snapshot.

For diagnostic prefill attribution, `oracle-bench --prefill-probe --rounds 2
--trace-path warm-prefill` brackets only the second round's full-prompt prefill
with CUDA profiler start/stop calls. Prefix preparation and forced decode remain
outside that range. Run under Nsight with the CUDA-profiler capture range; these
instrumented times are attribution evidence, not an untraced speed measurement.
`bench/analyze_nsys.py --tokens N --token-kind input` labels amortized prefill
activity per input token. The default normalization remains output tokens for
existing decode traces.

`DSX_REASONING_SCORE` is a prompt experiment, not a hard reasoning-token limit.
The recipe currently maps low/high/max to 50/75/100. A lower native score may
reduce planning cost, but its completion quality and latency require task tests.
The override accepts only 1–100 and replaces only the template-generated leading
header, preserving conversation text. With no override, prompt bytes are
unchanged. `validation-encode` honors the same process-local override so saved
requests can be encoded consistently with the serving window.

`DSX_THINKING_BUDGET=N` is an opt-in NVIDIA generation-policy experiment (1–8192
tokens). For a request whose prompt opens a thinking span, it emits the normal
sampled reasoning tokens up to the limit, then forces the thinking-close token
if the model has not closed naturally. Sampling still consumes its normal RNG
draw; after closure, answer sampling continues normally. Thinking-off requests
are unchanged. This intentionally changes generation and can reduce quality; it
requires actual coding-task validation. The raw-token audit records each forced
position, original sampled ID and chosen close ID, and runtime stats report the
applied budget and reasoning-token count. It is not a lossless optimization.

The [October 24 optimization ledger](../bench/results/2026-10-24-gauntlet/DECISIONS.md)
links measured gains, rejected configurations, task failures and harness fixes.
Experimental throughput results do not supersede the accepted serving configuration
or establish improved coding completion.

# DeepSeek-V4.1-Flash-1017 on a two-GPU desktop: findings

How fast can a 552B-parameter MoE run on a desktop with two RTX 3090s and 128 GB of DDR4, with exact routing, and
what limits it? This document collects the analysis, measurements and results behind dsx. Engine architecture:
[DESIGN.md](DESIGN.md). Hardware microbenchmarks: [bench/RESULTS.md](bench/RESULTS.md). Speed model:
[roofline.py](roofline.py).

## Summary

| | |
|---|---|
| Decode | ~29 tok/s on 2x RTX 3090 + 128 GB DDR4, exact routing (v4) |
| Prefill | 301 tok/s (3K tokens), 470 tok/s (8K tokens) (v4) |
| Quality | KL 0.373, top-1 74.4%, perplexity 10.68 vs the FP8 model's 8.56 (148 GB quant, v4b) |
| Limit | the experts the CPU computes from RAM (33-36 GB/s on DDR4-3200), then the dense half on GPU0 |

The levers that mattered, in order: keep every expert in VRAM or RAM (none on disk); rank VRAM residency by measured
expert hotness and let it follow the conversation; one CPU core per spinning thread; a parallel decode-attention kernel;
a quant whose bit budget is shared across layers, calibrated on the model's own reasoning; Hadamard-rotated experts with
refit scales.

## 1. The model

[DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017), MIT.

| | |
|---|---|
| Type | MoE: 384 routed experts + 1 shared per layer, 6 routed per token |
| Size | 552B backbone + 196B Engram tables + vision + 3 draft (DSpark / MTP) layers |
| Active | 16B parameters per decoded token |
| Layers | 40 (20 "encoder" + 20 "decoder"), hidden 5120, expert width 2304 |
| Attention | MQA sparse attention: 128-token window + top-512 compressed rows chosen by an indexer |
| KV cache | FP4, 890 bytes per token (128K tokens = 117 MB) |
| Checkpoint | 510 GB: dense FP8, experts FP4, Engram FP8 |

1017 has the same architecture as the October 16 V4.1 release, but every weight shard changed, the Engram tables
included. A V4.1 quant is not a 1017 quant.

| Part | Parameters | Placement in dsx |
|---|---:|---|
| Routed experts | 544B (15,360 × 35.4M) | quantized; VRAM, RAM, a few on disk |
| Dense (attention, shared expert, routers, embeddings, head) | ~8B | GPU0, Q8_0 |
| Engram tables | 196B, 2 × 384M rows | disk, read per token (~48 rows of 264 bytes per table) |

## 2. Target hardware

| | |
|---|---|
| CPU | i7-12700K (8P + 4E), AVX2 + AVX-VNNI, no AVX-512 |
| RAM | 4 × 32 GB DDR4-3200, dual channel: 51 GB/s peak, **39.5 GB/s measured** |
| GPU0 | RTX 3090, PCIe 4.0 x16: 24.7 GB/s host-to-device from hugepage memory |
| GPU1 | RTX 3090, chipset slot, PCIe 3.0 x4: 3.3 GB/s |
| Disk | NVMe, ~6 GB/s sequential |

Two properties shape everything: RAM bandwidth is the slow tier, and GPU1's link is too slow to stream experts, so it
holds a fixed set.

## 3. Performance model

**Decode.** Each token passes through 40 layers. The dense half runs on GPU0 every token (~14 ms on a 3090 after
optimization). The routed experts are 6 per layer, 8.5B parameters, ~2.1 GB per token at ~2 bits per weight. Each byte
costs according to where its expert lives:

| Tier | Bandwidth | Computed by |
|---|---:|---|
| VRAM | 936 GB/s | GPU |
| RAM | ~40 GB/s | CPU, in place |
| NVMe | ~6 GB/s | read, then CPU |

One NVMe read costs ~7× the same read from RAM, so the first rule is to keep every expert in VRAM or RAM. With two
24 GB GPUs and 128 GB of RAM that allows ~140 GB of experts: about 2 bits per weight.

**Routing is skewed on real text.** On the agentic calibration set, the hottest 10% of experts take 32-41% of routed
tokens, the hottest 25% take 55-61%, the hottest 50% ~80%, the coldest 10% 1.0-1.7%. Ranking VRAM residency by
hotness lets ~34 GB of VRAM serve over half of the traffic; RAM carries the rest.

**Prefill.** Every chunk of a long prompt uses nearly every expert, so the non-resident experts stream over GPU0's
PCIe link once per chunk (~100 GB at 24.7 GB/s ≈ 4 s). Larger chunks amortize it. Decoder layers only need the last
`(40 − L) × 128` tokens, so they run on that tail only.

**Long agentic contexts.** KV is small (117 MB at 128K), and attention reads a fixed window plus 512 rows, so long
context barely slows decode. A server that keeps each conversation's KV and only prefills new tokens makes follow-up
turns cheap.

## 4. Ecosystem (October 2026)

Runtimes that load V4.1 / 1017:

| Runtime | Status |
|---|---|
| llama.cpp | not merged ([PR #28696](https://github.com/ggml-org/llama.cpp/pull/28696)); forks run it |
| ik_llama.cpp | merged, with DSpark; open long-context issue |
| vLLM / SGLang | supported; no Ampere path |
| ExLlamaV3 (EXL3) | served through a vLLM-based kit, multi-GPU or DGX Spark |
| antirez/ds4 and derivatives | DS4 GGUFs on Mac, ROCm, CUDA |
| [Xolotl](https://github.com/Indras-Mirror/Xolotl) | one GPU + RAM + NVMe tiers, GSQ quant; ~26 tok/s on a 4090 with biased routing |

Published 1017 quants span 2-bit GGUFs (~150-375 GB with Engram), EXL3 2.8-3.5 bpw (196-240 GiB, perplexity within
0.4-1.8% of FP8), MLX, NVFP4 and VQ formats. Few publish quality numbers; few target 24 GB GPUs with 128 GB of RAM.

## 5. Engine results

Correctness: GPU decode matches a CPU reference implementation of DeepSeek's `model.py` to bf16 rounding (differences
only at router near-ties), and prefill matches token-by-token decode.

Decode on 2x RTX 3090 (3000-token agent prompt, exact routing):

| Change | Decode |
|---|---:|
| Baseline | 12.5 tok/s |
| One CPU per spinning thread (the CPU-expert coordinator shared a core with the GPU launch thread and saw routed picks up to 1.5 ms late) | 20.3 tok/s |
| Tiled split-K decode attention (276 → 21 µs per layer) | 31.5 tok/s |

Dense-half optimizations: repacked Q8 GEMV (22.6 → 10.6 ms per token), hyper-connection mixing (6.1 → 1.1 ms), partial
top-token selection instead of a full vocabulary sort, Engram row prefetch on its own threads (6.2 → 0.6 ms).

Prefill: a tensor-core attention kernel for long chunks (0.53 → 0.15 s per 4K chunk), deduplicated Engram reads, no
host stalls before the expert stream, and four staging buffers: 8K-token prefill 478 → 508 tok/s.

## 6. Expert placement follows the conversation

A VRAM set ranked on agentic text fits agentic text. Logging decode-time expert picks for three kinds of output:

| Text | Picks served from VRAM, static ranking | With an adaptive GPU0 cache | Best possible for the turn |
|---|---:|---:|---:|
| Prose essay | 15% | 68-72% | 90% |
| Python module | 43% | 73-74% | 84% |
| Agent / tool-call JSON | 47% | 78-79% | 86% |

GPU0's experts are therefore a cache over the RAM arena. After each token, the hottest non-resident experts (by
decayed usage, seeded from calibration counts) replace the coldest resident expert of the same quant type in place:
up to 8 per token (~72 MB over PCIe, overlapped with compute), committed between tokens. Short prose decode went from
26.0 to 29.9 tok/s on v1 with quality unchanged. The arena keeps copies of GPU0's experts (110 GB); the ~530 coldest
experts stay in the file.

## 7. Quantization

Each routed expert gets one of three tiers, chosen from its measured output error on held-out calibration text:

| Tier | Bits per weight | Grid |
|---|---:|---|
| PTQ1_0 | 1.75 | ternary, fp16 scale per 128 |
| Q2_0 | 2.25 | {-1, 0, 1, 2}, fp16 scale per 64 |
| Q3_K | 3.44 | 8 levels, 6-bit sub-scales per 16 |

Method: layer-by-layer calibration through DeepSeek's reference `model.py` on the FP8/FP4 checkpoint, GPTQ per
expert, then tier allocation by error per byte (Xolotl's GSQ pipeline). Calibration text: 128 × 2048 tokens of
coding-agent sessions and agent/code prompts through v2; from v3a, 160 × 2048 tokens with 53 sequences of the full
model's own reasoning traces. Dense weights are Q8_0; Engram stays unquantized.

What moves quality at a fixed size (Lucebox's 1017 eval, 8 × 1024 tokens against FP8 top-256 log-probs):

| Allocation, 148 GB | KL | Top-1 | Perplexity |
|---|---:|---:|---:|
| 2.0 bpw per layer (v1) | 0.462 | 71.5% | 11.66 |
| 2.0 bpw per layer, no limit on ternary | 0.463 | 71.8% | 11.93 |
| One budget for layers 2-36 (v2) | 0.395 | 74.1% | 10.93 |
| v2's tiers, calibration with reasoning traces (v3a) | 0.397 | 74.1% | 10.81 |
| + 128-block Hadamard rotation of every expert (v4) | 0.382 | 74.9% | 10.69 |
| + least-squares scale refit after GPTQ (v4b) | **0.373** | **74.4%** | **10.68** |
| Reference: 2.5 bpw (Lucebox ROCMFP) | 0.253 | 78.7% | 9.78 |
| Reference: Unsloth UD-IQ1_S (llama.cpp codebooks, ~122 GB without Engram) | 0.361 | 75.1% | 10.76 |

v3a changes what the model does more than these numbers show: its reasoning traces are ~1.2x those of a
near-lossless build on two hard math problems, against ~1.8x for v2. Rotation spreads each expert's outlier weights
before rounding (the engine rotates expert inputs to match); the refit re-solves each group's scale for GPTQ's codes
against the full Hessian, then GPTQ runs again. v4b's row was measured with the current prefill code, under which v4
scores KL 0.3816.

Sharing the budget across layers gives sensitive layers (3-13, 17, 32) 2.2-2.75 bpw and most decoder layers ~1.76.
Separate tiers for down and gate/up projections did not help. v2 is also faster: its smaller decoder-layer experts let
~200 more experts fit in VRAM (agent 30.0 tok/s; a 37.2 tok/s prose run had only 10.6% of picks on the CPU, against
25-31% on typical prose).

Quantizer speed: GPTQ's column loop and the scale search run as native CUDA kernels batched over experts, the H^-1
factor comes from one Cholesky of the reversed Hessian, and integer codes rather than fp32 values go to the writer.
One layer's experts: 159 → 70 s; a full model: ~2.5 h → ~1.5 h on two 3090s.

## 8. Other hardware: RTX 5090 + 128 GB DDR5

| | Static placement, 80 GB arena | Adaptive, 75 GB arena |
|---|---:|---:|
| Decode, prose | 12.0 tok/s | 7.6 tok/s |
| Decode, 3K-token agent prompt | 15.2 tok/s | 7.8 tok/s |
| Prefill, 3K / 8K tokens | 93 / 172 tok/s | 68 / 114 tok/s |

With one 32 GB GPU and ~90 GB of usable RAM, ~40 GB of experts are read from disk through the page cache, and that
sets the speed. Static placement wins there: the adaptive cache's RAM copies of GPU0's experts cost space the machine
doesn't have. Such a machine needs ~160 GB of RAM, or a disk tier with batched direct reads and prefetch.

## 9. Negative results

- **Speculative decoding (DSpark).** A verify pass of 2 tokens costs 55 ms (1.5 decode steps), of 3 tokens ~70 ms,
  after moving verify onto decode kernels (from 101 / 109 ms). Most of what remains is the CPU's expert work, which
  grows with the union of the tokens' experts (20.9 distinct experts per layer for 6 tokens vs 6.0 for one). With the
  draft's expected acceptance on a 2-bit model it would gain ~0-5% at best; it pays where every expert sits in GPU
  memory (1.2-1.7x on the MI325X build, [FINDINGS-AMD.md](FINDINGS-AMD.md)).
- **Routing bias toward VRAM-resident experts.** +4 tok/s at a bias of 0.2, for 2 points of top-1. Off by default.
- **EXL2 / EXL3.** ExLlama keeps all weights in VRAM and has no RAM tier. EXL3's trellis codes would improve quality per
  bit for GPU-resident experts, but decoding them on the CPU is likely compute-bound.
- **Experts on disk.** Each token needs ~2 GB of expert weights; serving 10% of it from a 6 GB/s drive adds ~30 ms.
- **Swapping on the x4 GPU** (`DSX_AUX_ADAPTIVE=1`). Decode on a 3K agent prompt 28.8 → 31.0 tok/s, on prose 29.1 →
  22.0: evictable experts need RAM copies, GPU1's 23 GB of them push ~2,200 cold experts out of the arena onto disk.
  Off by default (slow links keep a fixed set).
- **Cache tuning.** Usage decay 0.90-0.995 (`DSX_DECAY`) and 16 instead of 8 swaps per token: all within ±1 tok/s.
  Hit rate on this machine is set by capacity, not by the eviction rule.
- **CPU expert speed.** The CPU kernels read 30-35 GB/s. The memory controller moves ~42.6 GB/s during a plain read test
  (DDR4-3200 Gear 1, but 4 dual-rank DIMMs run 2N command rate and slow refresh); the v5-era benchmarks also lost a few
  GB/s to a concurrent job. Pinning the expert pool to one thread per physical P-core (`DSX_CPU_LIST`) was 1-2% slower
  than the default 14 threads on 7 P-cores with their hyperthreads.

## 10. Open problems

- Quality per byte: KL 0.373 at 148 GB against Unsloth's 0.361 at ~122 GB (codebook formats, more bits in late layers
  and down projections), and the gap to 2.5 bpw (0.253) at a size that fits 128 GB + 48 GB.
- A disk tier for RAM-limited machines: batched direct reads, one read per expert, next-layer prefetch (as in
  [colibri](https://github.com/JustVugg/colibri)).
- Prefill: the expert stream runs at ~75% of the PCIe floor.
- The dense half (~14 ms per token) as decode's next floor: its weight GEMVs run at 740-860 GB/s (q_b 800, o_a 740, LM
  head 860), and ~1,000 small kernels take ~3 ms; gaps between kernels are only ~1.2 ms per token.

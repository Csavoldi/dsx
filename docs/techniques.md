# Techniques

What dsx does to fit DeepSeek-V4.1-Flash-1017 into a desktop and run it fast, with measured effects, and how that
compares with other engines for this model family. Measurements: [FINDINGS.md](../FINDINGS.md).

## Current T16 implementation

The default model uses T16 v1 (1.5–2.5 bpw experts, Q8_0 dense, IQ4_NL Engram lookup table). Its trellis codebook,
column-major tile positions, damped LDLQ and baked-in output gains are described in [T16 experiments](t16.md).
The later runtime integrates staged T16 GEMVs and small-kernel/launch changes; [current state](current-state.md)
records the measured gains, defaults and the unresolved fixed-residency decode check. The output gains were chosen
on the reported evaluation set. The calib5 requant completed but did not lower its general-text KL.

The tables below retain the earlier scalar and Unsloth experiments. Their speed and quality numbers belong to
those builds, not the current T16 integration.

## Quantization (build v4b, 148 GB)

| Technique | Source | Measured effect |
|---|---|---|
| Per-expert tiers: ternary PTQ1_0 (1.75 bpw), Q2_0 (2.25), Q3_K (3.44); dense Q8_0; Engram unquantized | [Xolotl](https://github.com/Indras-Mirror/Xolotl) GSQ formats | basis of the historical scalar builds |
| One bit budget across layers 2-36, tiers picked by output error per byte | dsx (v2) | KL 0.462 → 0.395 at the same size |
| GPTQ per expert, Hessians from a layer-by-layer calibration pass through DeepSeek's reference `model.py` | [GPTQ](https://arxiv.org/abs/2210.17323); Xolotl pipeline | — |
| Native batched CUDA GPTQ, scale search and H^-1 | dsx | 159 → 70 s per layer |
| Calibration text with the full model's own reasoning traces | dsx (v3a) | reasoning length ~1.8x → ~1.2x of a near-lossless build |
| 128-block Hadamard rotation of every routed expert's input dimension | [QuIP#](https://arxiv.org/abs/2402.04396), [QuaRot](https://arxiv.org/abs/2404.00456) | KL 0.397 → 0.382 |
| Least-squares refit of group scales after GPTQ, GPTQ again, a final refit | dsx (v4b) | KL 0.382 → 0.373 |

KL: Lucebox's 1017 eval set, 8 x 1024 tokens against the FP8 model's top-256 log-probs.

IQ formats for the dsx-ud build (Unsloth UD-IQ1_M's experts, KL 0.270 at 142 GB):

| Technique | Source | Measured effect |
|---|---|---|
| ggml's IQ1_S, IQ2_XXS, IQ2_XS, IQ2_S as expert types: CPU kernels, GPU decode tiles, prefill dequant | [llama.cpp](https://github.com/ggml-org/llama.cpp) formats and tables | 28-39 GB/s on 16 CPU threads; 1.6-2.5x Q2_0's GPU tile time |
| IQ1_S's +-1/8 shift as one madd per 256 weights against per-16 activation sums | dsx | IQ1_S CPU 28 -> 35 GB/s |
| One GGUF from dsx's dense weights and another GGUF's routed experts (`quant/hybrid_gguf.py`) | dsx | KL 0.270 vs 0.275 for the same experts in llama.cpp |

## Historical scalar runtime (2x RTX 3090 + 128 GB DDR4)

| Technique | Measured effect |
|---|---|
| Hottest experts in VRAM, ranked by routing counts on calibration text | — |
| Adaptive main-GPU cache: up to 8 replacements per token by decayed usage | prose picks served from VRAM 15% → 70% |
| Automatic GPU roles (up to 6 GPUs) by measured host link; slow links keep a fixed set ([placement](placement.md)) | same speed as hand placement |
| CPU computes RAM experts in place from a hugepage arena (AVX2, AVX-VNNI, AVX-512 VNNI kernels) | early kernels: 30-35 GB/s; later RAM read tests reached ~42.4 GB/s |
| No host synchronization inside a token: per-layer flags in mapped memory, GPU streams wait with `cuStreamWaitValue32` | — |
| One core per spinning thread (CPU-expert coordinator off the launch thread's core) | 12.5 → 20.3 tok/s |
| Tiled split-K decode attention | 276 → 21 µs per layer |
| Repacked Q8 GEMV for the dense half | 22.6 → 10.6 ms per token |
| Fused hyper-connection mixing | 6.1 → 1.1 ms per token |
| Engram rows prefetched on helper threads | 6.2 → 0.6 ms per token |
| int8 tensor-core kernels for VRAM experts, up to 8 tokens per call | — |
| Prefill: decoder layers only on tokens later positions can attend to; tensor-core attention; experts streamed through 4 staging buffers; other GPUs compute their own experts | 8K prefill 478 → 508 tok/s |
| Rotated experts in prefill: activations rotated instead of weights | 3K prefill 217 → 300 tok/s |
| Decode kernel fusions: router top-k in registers, mixing reduce + finish, KV norm + RoPE + FP8 + store | +1% |
| Verify passes (speculative decoding) on decode kernels | 2 tokens 101 → 55 ms |
| Opt-in expert deferral (`--defer-rank 3`, after [KTransformers](https://madsys.cs.tsinghua.edu.cn/publication/ktransformers-unleashing-the-full-potential-of-cpu/gpu-hybrid-inference-for-moe-models/SOSP25-chen.pdf)): the CPU's experts ranked 3-5 enter the residual one layer later, with their own layer's output weights | decode 29.9 → 35.9 tok/s; KL 0.369 → 0.383 |
| Prefix reuse across turns; vision weights streamed from RAM | — |

## Historical comparison with other engines

Snapshot from the original October 2026 investigation; it describes dsx-2bit and contemporaneous external sources,
not a current feature audit of those projects. "?" = not verified. vLLM and llama.cpp entries describe their
general capabilities, not a V4.1-specific audit.

| | dsx | Xolotl (on Strata) | llama.cpp (V4.1 PR #30817) | vLLM | DeepInfer |
|---|---|---|---|---|---|
| Target | one desktop, CPU + GPU | desktops, CPU + GPU, Windows | anything, CPU + GPU | datacenter GPUs | datacenter clusters (NVIDIA, Ascend) |
| Expert format | per-expert ternary / 2-bit / 3-bit | same tiers | k-quants, IQ codebooks (e.g. Unsloth's mixes) | FP8, MXFP4, GPTQ / AWQ | MXFP4 from the QAT checkpoint |
| Bit allocation | per expert, error per byte | per expert, with ternary caps | per tensor: importance matrix + recipes | per model | — |
| Calibration | layer-streamed, model's own reasoning traces | layer-streamed, agent text | importance matrix on chat text | — | — |
| Rotation | Hadamard | tried, measured worse | — | external tools | — |
| Scale refit | yes | designed, not built | — | — | — |
| Expert placement | automatic, hottest first, up to 6 GPUs | VRAM / RAM tiers | manual (`-ot`, `--n-cpu-moe`) | all in VRAM (`--cpu-offload-gb` is not expert-aware) | all in HBM, expert parallel + EPLB |
| Cache follows the conversation | yes | ? | — | n/a | n/a (EPLB rebalances every 10K steps) |
| CPU computes experts | yes, at RAM bandwidth | CPU pool or misses over PCIe | ggml CPU backend | — | — |
| No host sync within a token | yes | ? | partial | CUDA graphs | CUDA graphs, async scheduler |
| Speculative decoding | verify path only (does not pay with CPU experts) | DSpark (draft keeps its 64 hottest experts), prompt lookup | separate draft model | MTP | DSpark with confidence-scheduled verification |
| Dense weights | Q8_0 | Q6_K at load | Q5_K / Q6_K in Unsloth's mixes | FP8 | FP8 |
| KV cache | FP4 compressed, FP8 window | FP4 option | ? | FP8 | FP4 compressed, FP8 window |
| Engram tables | official shards, prefetch threads | disk + pinned hot rows | IQ4_NL inside the GGUF | ? | host RAM |
| Batching | one user | one user | multi-slot | continuous batching | continuous, batch-invariant |
| 2x RTX 3090 + 128 GB | ~29 tok/s decode, 301 tok/s prefill | 14.7-22 tok/s (their build, RTX 3090 on Windows) | 12.3 tok/s decode, 109 prefill (Unsloth IQ1_M) | does not fit | does not fit |
| Quality (Lucebox KL vs FP8) | 0.373 at 148 GB | other metrics | Unsloth UD-IQ1_S 0.361 (~122 GB), UD-Q2_K_XL 0.114 (~224 GB) | — | the checkpoint's own precision (reference) |

The DeepInfer row is [deepseek-ai/DeepInfer](https://github.com/deepseek-ai/DeepInfer); the Xolotl row is
[Indras-Mirror/Xolotl](https://github.com/Indras-Mirror/Xolotl) at 9747b9e3. The llama.cpp numbers on 2x RTX 3090 and
the KL of Unsloth's GGUFs were measured with dsx's eval set ([cards/dsx-2bit](../cards/dsx-2bit/README.md)).

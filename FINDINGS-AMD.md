# DeepSeek-V4.1-Flash-1017 on one AMD MI325X: findings

Goal: run the whole model (minus Engram tables) from the HBM of one MI325X with a runner written for this model and
this GPU, then measure decode at 128K and 256K context, prefill, and quality against FP8. Work log and results below;
the architecture and quant background is in [FINDINGS.md](FINDINGS.md).

## 1. The box

| | |
|---|---|
| GPU | AMD Instinct MI325X (gfx942, CDNA3): 304 CUs, 256 GiB HBM3E (274.9 GB), 6.0 TB/s peak, 256 MB Infinity Cache |
| Host | 20 vCPUs, 157 GiB RAM, 720 GB boot disk, 5 TiB scratch NVMe |
| Software | ROCm 7.14 (prerelease image), Ubuntu 24.04 |
| Price | $3.80/h (DigitalOcean `gpu-mi325x1-256gb`) |

Measured on the rented box (ROCm 7.14, `mbench`):

| | |
|---|---|
| Empty kernel in a HIP graph | 1.53 µs (grids up to 1216 blocks); 3.3 µs at 4864 blocks, 10 µs at 19456 |
| Stream launch, empty kernel | 2.28 µs |
| Dependent load latency | 290 ns (32 MB working set) to 340 ns (2 GB) |
| Streaming read, 16 B nontemporal loads | 5.45 TB/s (91% of peak) at 304 x 256 threads, 4 loads in flight |
| Agent-scope fence + atomic per block | 80 µs per 1216-block launch (vs 2.2 µs without): never fence per block |
| NVMe scratch, parallel page-fault reads | ~1.7-2.4 GB/s (checkpoint load: 143 s) |

## 2. What has to fit

| Part | Parameters | Native size | Where it goes |
|---|---:|---:|---|
| Routed experts | 544B | 289 GB (MXFP4, 4.25 bpw) | HBM, re-quantized to ≤ 3.6 bpw |
| Dense (attention, shared experts, routers, hyper-connections, embeddings, head, Engram projections) | ~7.6B | ~7.5 GB (FP8 + bf16) | HBM, as in the checkpoint |
| Engram tables | 196B | 203 GB (FP8 + E8M0 scales) | NVMe scratch + page cache; 48 rows of 264 bytes per token |
| DSpark draft layers | 3 × (attention + 128 experts) | ~6 GB | HBM |
| KV + indexer keys at 256K | | ~0.5-1 GB | HBM |

The original expert format (MXFP4) needs 289 GB and does not fit in 274.9 GB next to everything else. With ~8 GB of dense
weights, ~6 GB of draft layers and a few GB of activations and caches, the expert budget is about 250 GB: **3.6-3.7 bits
per weight**. The Engram tables do not fit in host RAM either (157 GiB); they stay on NVMe, read through the page cache.
Their rows depend only on the token ids, so a token's reads start the moment it is sampled and layer 1 (the first Engram
layer) runs ~0.1 ms later.

## 3. Per-token cost on this GPU

Weights read per decoded token (one token, no speculation):

| | Bytes per token |
|---|---:|
| Dense FP8 per layer: wq_a 6.6M, wq_b 41.9M, wkv 2.6M, wo_a 33.5M, wo_b 41.9M, shared expert 35.4M, router 3.9M (bf16), hc 2M | ~168 MB × 40 = 6.7 GB |
| Head (129280 × 5120), Engram wkv (2 × 157M), indexer/compressor | ~0.9 GB |
| Routed experts: 6 × 35.4M × 40 = 8.49B weights | 3.4 GB at 3.2 bpw; 3.8 GB at 3.6 bpw |
| **Total** | **~11 GB** |

At ~4.8 TB/s achieved (80% of peak; pocket-engine's GEMVs reach 4.8-5.4 TB/s on this GPU) that is **~2.3 ms** of pure
weight streaming per token. Two thirds of it is the FP8 dense half, not the experts.

The rest is fixed cost: ~40 layers × (attention, indexer, router, experts, hyper-connection mixes). A small kernel in a
HIP graph costs ~1.5-2.5 µs on gfx942, and every dependent trip to HBM ~0.4 µs. With ~12 kernels per layer that is
~1-1.5 ms per token. Long context adds the indexer: at 256K tokens, layers 2/8/14 score 128K compressed keys and layer
20 scores 256K (32 heads × 128 dims; ~5 GFLOP and ~40-170 MB of keys depending on the key format), plus top-512
selections. That is ~0.1-0.2 ms per token on MFMA.

**Estimate: 3.5-4.5 ms per token, 220-280 tok/s single-stream decode at 128-256K** if the per-layer work is fused well.
That is above the 120-180 tok/s first estimate in the brief, which assumed larger fixed costs.

Speculative decoding (DSpark, 5-token blocks): verifying k tokens costs the same dense bytes (the GEMVs become
16-row MFMA tiles at no extra weight traffic) but the union of routed experts grows (dsx measured 20.9 distinct experts
per layer for 6 tokens vs 6.0 for one), so a 6-token verify reads ~7.5 + 12 GB ≈ 4 ms. With ~3 accepted tokens per pass
that is ~1.7x.

Prefill: ~16B active parameters × 2 FLOP per token = 32 GFLOP per token; at ~500 TFLOP/s achieved on FP8 / int8 MFMA
that is ~15K tok/s; at 30-40% of the 2.6 PFLOP/s FP8 peak, 25-30K tok/s.

## 4. Ecosystem (October 2026)

No runtime is known to run 1017 on a single MI300X/MI325X:

| Runtime | gfx942 | Single MI325X | Notes |
|---|---|---|---|
| vLLM ROCm + AITER | yes | no: TP4 by default, TP2 at minimum KV | MXFP4 experts via Triton (no FP4 MFMA on gfx942); "mono decode" PR #60731 (open): MI325X TP4 217 → 503 tok/s per user with DSpark, short prompts |
| SGLang ROCm | unconfirmed | no evidence | |
| llama.cpp PR #28696 (open) | not tested on HIP | | GGUF Q2_K-Q8_0 exist (246-473 GiB) |
| antirez ds4, ds4fa | Strix Halo only | | ROCmFP block formats (2.5 / 3.5 / 4.25 bpw codebooks) came from ds4fa |
| Lucebox `luce_server` | gfx1100/1151/1201 shown, not gfx942 | | ROCMFP2S 158 GiB: KL 0.265; ROCMFP23S 179 GiB: KL 0.194 |
| ExLlamaV3 | no DeepSeek-V4.1 class upstream | | EXL3 2.9 / 3.5 bpw: 196 / 240 GiB without Engram |

For scale: V4-Flash-0731 (the older release) on one MI300X with vLLM nightly ran 67 tok/s single-stream (153 with
DSpark-7). None of the 1017 quants that fit 256 GB has a gfx942 runtime, so the baseline here is our own runner.

## 5. Plan

1. Measure the box: HBM bandwidth for GEMV shapes, graph launch cost, MFMA throughput for 16-row tiles.
2. Runner (`amd/`, Rust + HIP): everything in HBM; one HIP graph per decode step with all position-dependent values
   read from device memory; GEMVs on MFMA (FP8 weights × FP8 activations, exactly the checkpoint's numerics); routed
   experts gathered by device-side router output; indexer scoring on MFMA with device top-k.
3. First end to end: dsx's 2-bit quant (148 GB) fully in HBM.
4. Quant: ~3.5 bpw experts allocated by measured error, measured on Lucebox's eval set.
5. DSpark speculation.

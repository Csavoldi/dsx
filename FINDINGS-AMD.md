# DeepSeek-V4.1-Flash-1017 on one AMD MI325X: findings

This document records the original model release and early single-session experiments. Later batching and
agentic-serving work is described in [the AMD guide](amd/README.md) and [profile](amd/box/AGENTIC_MI325X.md).
Rental totals and proposed next steps below belong to those earlier runs; the [rental ledger](amd/box/RENTALS.md)
records the subsequent sessions and their closure.

## Original release measurements

| | |
|---|---|
| Decode, single stream | 114.1 tok/s at 4K context, 111.5 at 128K, 110.4 at 256K (8.8-9.1 ms per token, one HIP graph) |
| With DSpark drafts (greedy) | 1.4-1.7x on code, 1.2-1.4x on math, ~1.0x on prose (section 6) |
| Prefill | 1,640 tok/s (4K and 16K prompts), 1,160 (128K) |
| Quality vs FP8 (Lucebox 1017 eval, 8 x 1024 tokens) | KL 0.049, top-1 90.6%, perplexity 8.71 (FP8: 8.56), on the published shards (0.047 on another run) |
| Model in HBM | 271.6 GB: dense FP8 as in the checkpoint (7.2 GB), 8,992 experts exact MXFP4 + 6,368 re-quantized to a 3-bit codebook (260.6 GB), bf16 embeddings and head |
| Cost at $3.80/h, one stream | $9.3 per million generated tokens, $0.64 per million prompt tokens |
| Published | [lxe/DeepSeek-V4.1-Flash-1017-dsx-mxfp4](https://huggingface.co/lxe/DeepSeek-V4.1-Flash-1017-dsx-mxfp4), 6 GGUF shards, 270.5 GB |

For comparison on that eval: an earlier dsx 2-bit build (148 GB) measured KL 0.395; Lucebox ROCMFP2S (158 GiB) 0.265, ROCMFP23S
(179 GiB) 0.194. With 256 GiB of HBM there is room to keep 59% of the experts in the checkpoint's own FP4 (the model
was trained with them) and give the coldest 41% (by measured routing frequency) a fitted 3-bit codebook.

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

## 5. Tuning notes

- Decode GEMVs: the k-split is chosen per launch so it has ~2048 waves (at least 4 weight units per wave). 8192 waves
  (the first default) cost 0.5 ms per token: 107 vs 114 tok/s.
- Prompt top-k: the threshold bin of the 4096-bin histogram was found by one thread walking the bins through LDS,
  ~150 µs per query block. A block-wide suffix scan takes it to ~7 µs, and prefill at 16K went from 1,154 to 1,640 tok/s.
- Run-to-run differences: float atomics (k-split GEMVs, scatter-adds) sum in varying order, which flips a few
  near-tied expert choices. The published build measured KL 0.049 and 0.047 on two runs (an earlier placement with
  8,804 MXFP4 experts: 0.051).

## 6. Speculative decoding (DSpark)

The checkpoint's draft stack (`mtp.*`: 3 window-only layers with 128 experts each, top-3, drafting a block of 5
tokens at once, plus a Markov head) runs on the GPU: the draft layers are ordinary layers on the prompt path with
block-bidirectional attention over a window of the main model's projected residual (`amd/src/dspark.rs`). The main
model verifies the newest token plus the 5 drafts in one pass of 6 positions: decode-style weight streaming with the
positions as MFMA columns (`kernels/verify.h`), routed experts grouped on the device, state committed for the
accepted prefix only. `DSXA_DSPARK=3` loads it (6.5 GB of HBM; 1,613 cold experts go from MXFP4 to 3-bit to make
room: KL 0.056 instead of 0.049); the server then uses it for greedy requests.

| 300 tokens, greedy | Drafts accepted per pass | vs plain decode |
|---|---:|---:|
| Code (parser + tests) | 3.0-4.0 | 1.4-1.7x |
| Math word problem | 2.8-3.0 | 1.2-1.4x |
| Explanation, translation | 1.4-2.8 | 1.0-1.2x |
| Short story | 1.0 | 0.94x (falls back to plain decode) |

A pass costs ~23-26 ms (draft 1.5 ms, verify of 6 positions 18 ms, commit and window update) against 8.6 ms per
plain token, so it pays above ~2 accepted drafts; below that the loop decodes 15 tokens plainly and probes again.
Raw runs: [amd/results/dspark-spec.log](amd/results/dspark-spec.log).

Where the verify pass goes (6 positions, ~18 ms): routed experts ~7-9 ms (6 tokens touch ~25 distinct experts per
layer, 4x the bytes of one token, streamed at ~2 TB/s), dense GEMVs ~3.7 ms, and ~6 ms of small per-token kernels
(attention, hyper-connection mixes, norms). Replaying the pass as a HIP graph changes nothing (it is GPU-bound).
Tuning that mattered: one k-unit in flight per wave instead of two in the expert kernel (more waves resident; 19.8 →
18.2 ms), k-split expert GEMVs with outputs zeroed by the grouping kernel, a batched head.

The expert GEMV is also decode's largest cost: per launch, every block spends ~8 µs picking the experts from the
router logits and ~9 µs staging the activation before ~15 µs of streaming (`gtime`). Producing the routing and the
FP8 activation once, and blocks that stream several tiles each, is the next step for both decode and verify.

## 7. Historical status and proposals

Done: the runner (`amd/`), the MXFP4 + 3-bit build (published as 6 GGUF shards), decode 110-114 tok/s at 4K-256K,
prefill 1,160-1,840 tok/s, the OpenAI / Anthropic server, DSpark speculation (opt-in). Rental: one MI325X,
~9 hours (~$34 at $3.80/h).

Next, in order of expected gain:

1. Expert GEMVs at HBM speed: route once per layer (the router's last block) and FP8 activations produced once,
   so expert blocks start streaming immediately; several tiles per wave. Decode spends ~65 µs per layer here at
   ~1.7 TB/s; at ~4 TB/s decode would be ~135 tok/s, and verify passes ~4 ms shorter.
2. Verify pass: fold the per-token prompt kernels (mix, norm, rope, window copy) into fewer launches with more
   blocks per token; target ~12 ms for 6 positions, which would make DSpark ~1.5-2x on most text.
3. DSpark draft layers in the export (they are read from the official checkpoint today).

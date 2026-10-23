# dsx-amd batched runner: plan

Goal (lxe, 2026-10-23): a Volta / pocket-engine style batched inference runner for DeepSeek-V4.1-Flash on one
MI325X, with maximum throughput for one session and for many sessions at up to 256K context. Budget $100 of
DigitalOcean time (amd/box/RENTALS.md).

Built inside dsx-amd: its loader, weight formats, prompt path and batched-decode kernels are validated against the
teacher (Lucebox KL equal to decode), so the runner adds a scheduler, memory planning and faster kernels on top.

## Floors (MI325X: 6 TB/s HBM, 304 CUs, 256 GB)

- B = 256: ~377 of 384 experts read per layer per step, so a step streams nearly the whole 3-bit expert set
  (~210 GB, ~42 ms at 5 TB/s). Indexer keys at 256K add ~11 GB per step as FP4. Ceiling ~5-6K tok/s.
- B = 1: ~10 GB per token (dense FP8 6.7 GB, experts 3.5 GB) = ~2 ms; dsx-amd decodes in 8.7 ms (541 kernels,
  products at ~1.5-2 TB/s). The gap is latency and slow matrix-vector kernels.

## Lessons taken (Volta, Volta Scout, pocket-engine)

- Megakernels and persistent per-layer kernels lost on both V100 and MI325X; use HIP graphs per step shape and
  fuse selectively. No `__threadfence` / acquire-release hand-offs between blocks (gfx942 flushes L2).
- Matrix-vector at 1-4 rows: wvSplitK style (a wave owns whole rows over all of K, 4+ non-temporal 16-byte loads in
  flight per lane, in-wave reduction) reaches 4.8-5.4 TB/s (pocket-engine skinny.cuh / fp8w.cuh).
- Issue every load before any branch; size grids so no CU runs a second block; budget ~1.5 us per graph node.
- Compressed KV and indexer keys in native FP4 (exact: the model rounds them to FP4): ~0.9 KB/token,
  0.23 GB per session at 256K (bf16 would be 0.85 GB).
- Continuous batching over slots, prompt chunks mixed into decode steps, prefix reuse, graphs keyed by
  (batch bucket, context bucket), drafts padded to multiples of 8.
- hipBLASLt for 5-256 rows (tuned per shape); FP8 on gfx942 is FNUZ.

## Phases

1. Memory: FP4 slot state (compressed rows 256 B + 16 E4M3-ish scales; indexer keys 64 B + 4 E8M0 scales),
   a planner that sizes slots x context and re-quantizes experts to fit.
2. Indexer on MFMA over FP4 keys (FP4 -> FP8 is exact; the 32-wide scale block matches the FP8 MFMA k of 32).
3. Scheduler / server: continuous batching, chunked prefill into slots, prefix reuse, OpenAI-style HTTP.
4. Single session: wvSplitK-style FP8 / codebook matrix-vector kernels for decode, fewer graph nodes.
5. Prefill throughput at long prompts (dense on hipBLASLt, grouped expert GEMM, indexer).

Measure each step on the box (beval KL, bbench / long-context benches), commit by path, push.

## Next items: budgets (2026-10-23, from the B=8 distinct-session profile: 22.1 ms per step)

1. Dense products for 1-16 tokens. Measured 4.75 ms per step (wo_b 0.85, router+shared 0.95, wo_a 0.77, q_b 0.77,
   q_a/kv 0.56, shared down 0.44, head 0.41) for ~8.0 GB of FP8 / BF16 weights: ~1.7 TB/s. Floor at 5 TB/s: 1.6 ms.
   Bottleneck: bytes in flight. vgemv_d runs at 13% occupancy, 6% VALU busy; ~1.7 waves per SIMD x 2 KB in flight
   is ~14 KB per CU against ~40 KB needed at ~2 us HBM latency. Deeper prefetch (VDU 4) and more waves did not help;
   unconditional loads were slower.
   Predicted gain: 2.5-3 ms per step (~12%) with a kernel that keeps >= 40 KB per CU in flight and no k-split
   zeroing. Confirming measurement first: a standalone microbenchmark of the wo_b shape (5120 x 8192 FP8, 8 tokens)
   on the MI325X. Kill if it does not reach 3.5 TB/s.
   Result (2026-10-23, amd/results/dgemv-bench-2026-10-23.txt): killed. At 8 tokens the best variants reach
   2.94 TB/s (q_b, 2 tiles per wave sharing activation loads) and 2.53 TB/s (wo_b), against 2.77 / 2.41 for the
   current structure; loading whole k-slices first or partial sums without atomics do no better. Throughput falls
   with tokens (wo_b 3.78 / 2.41 / 1.47 TB/s at 1 / 8 / 16 tokens) for reasons these variants do not remove, and
   small products (wq_a 6.5 MB, sh_w1 11.8 MB) stay at 1-1.7 TB/s: launch ramp and tail dominate them, which only
   fusing products would fix.
2. Small kernels. prep_t 1.67 ms (240 launches), hc_step 1.07, hc_mix_t 1.01, rope / kv / attn_out / route / bgroup
   ~1.5: ~5.3 ms in ~600 launches for under 100 MB of traffic (floor < 0.1 ms; ~1.5-3 us per launch minimum).
   Predicted gain: ~2 ms by staging the activation (norm + FP8 quantization) inside the dense products' blocks, as
   the decode path's gemv_jobs does (removes the 160 prep_t launches before dense products), and merging hc_mix_t
   into hc_step. Confirm with the same profile (launch count and per-kernel time). Kill if the fused products lose
   more than they save.
3. Expert kernels (9.4 ms at B=8): 59% VALU busy, 38% occupancy. Gains need fewer VALU ops per weight (codebook
   decode ~1.75 ops per weight, scales ~0.5 after packed math). No item yet that passes the budget rule.

## Batch plan after rental 5 (2026-10-23 04:20 PT)

- Item 2 revised budget: hc_mix_t is 1.01 ms in 83 launches (~12 us each), mostly a serial 20-iteration Sinkhorn on
  one wave; folding it into hc_step's last block per token moves that work, it does not remove it. The attention/FFN
  prep_t (80 launches, ~7 us each) is the better fold. Revised prediction for the fold: ~0.6 ms of 22 ms at B=8.
  Confirming measurement first (amd/bench/launch_bench.hip): the eager per-launch floor on this stream. Build the
  fold only if an empty / one-load small kernel costs >= 4 us per launch; below that, removing ~160 launches saves
  < 0.5 ms (kill).
- prep_t mode 0: the 80 partial sums were summed by a serial loop in every thread; now one load per lane and a wave
  sum. Predicted ~2 us per launch (0.15 ms per step). Measure by interleaved kernel-reload A/B (bsweep B=1/8/12).
- Cheaper expert decode: the 3-bit codebook loop is 171 VALU per 64 weights per lane (~96 unpacking the 3-bit
  indices from bit planes, ~48 scales); field unpacking cannot go much below ~1.5 instructions per weight with VALU
  bit operations, and MXFP4 is no cheaper (162 per 64, 30% more bytes). No format change passes the budget rule now.

## Gap to floor (2026-10-23 04:30 PT)

Floor = bytes per token or step at 5 TB/s (practical HBM rate; 6 TB/s peak). Experts at the load's mix (~14.4 MB per
3-bit expert, 18.8 MB MXFP4; RES 24: 1953 MXFP4 + 13407 3-bit), dense FP8 6.7 GB, head BF16 1.3 GB. Experts touched
per layer by B sessions under uniform routing: 384 * (1 - (1 - 6/384)^B). Attention state at 256K: single session
(bf16 lane caches) ~0.19 GB per token, batched (FP4 slots) ~0.058 GB per session per step.

| mode | measured | floor (bytes) | gap | source of measured |
|---|---|---|---|---|
| 1 session, 4K | 8.66 ms (115.5 tok/s) | 2.3 ms (~11.6 GB) | 3.8x | rental 2/3 `bench 4096` |
| 1 session, 128K | ~8.8 ms (~113 tok/s, est.) | 2.35 ms | 3.7x | interpolated 4K/256K |
| 1 session, 256K | 8.95 ms (111.8 tok/s) | 2.4 ms (~11.8 GB) | 3.7x | rental 3 `bench 262000` |
| B=8, 256K, plain | 15.6 ms (64 tok/s each) | 6.9 ms (~34.5 GB) | 2.3x | rental 3 `lspec` 258K prefix |
| B=8, 256K, speculation (agentic) | 92 tok/s each | n/a (more tokens per expert read) | | rental 3 `lspec` |
| B=64, 256K | 52.3 ms (1,223 total) | 30.5 ms (244 experts/layer, ~152 GB) | 1.7x | rental 3 `bbench 64 262144` (before later kernel work) |
| B=128, 228K | 76.4 ms (1,676 total) | 41 ms (332 experts/layer, ~201 GB) | 1.9x | rental 3 `bbench 128 232000` (1595 2-bit experts) |

Ranking by the gap each next item closes:
1. Single-session expert products (decode path, gemv_jobs): gate/up 39.6 us + down 23.9 us per layer for ~86 MB
   (6 experts x 3 matrices) = ~1.4 TB/s; at 5 TB/s they would take ~17 us. Closing that saves ~1.9 ms of 8.7 ms
   (+28%: ~147 tok/s). Mechanism: 1,728 waves for gate/up is ~1.4 waves per SIMD with ~3 KB in flight each,
   ~17 KB per CU against ~40 KB needed. Confirming measurement (kernel reload + `waves`, no code): sweep X1U
   (prefetch depth 4 / 6 / 8) x TARGET_WAVES (2048 / 4096 / 8192) on `prof 4096`; build a dedicated kernel only if
   one point moves the expert time >= 20%. Kill otherwise.
2. Single-session small kernels: hc_step 16 us x 83, attention 41 us x 40, and ~540 graph nodes; the per-launch floor
   (launch_bench, rental 6) sets how much fusing can recover.
3. B=8 fold of prep / hc_mix into hc_step (~0.6 ms of 22 ms at 4K, ~3%), gated on the same launch floor.
4. B >= 64: experts read at ~3.7 TB/s and dominate; closing to 5 TB/s is ~25% of the step, but the 3-bit decode is
   ALU-bound (171 VALU per 64 weights); no format change passes the budget yet.

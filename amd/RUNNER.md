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
2. Small kernels. prep_t 1.67 ms (240 launches), hc_step 1.07, hc_mix_t 1.01, rope / kv / attn_out / route / bgroup
   ~1.5: ~5.3 ms in ~600 launches for under 100 MB of traffic (floor < 0.1 ms; ~1.5-3 us per launch minimum).
   Predicted gain: ~2 ms by staging the activation (norm + FP8 quantization) inside the dense products' blocks, as
   the decode path's gemv_jobs does (removes the 160 prep_t launches before dense products), and merging hc_mix_t
   into hc_step. Confirm with the same profile (launch count and per-kernel time). Kill if the fused products lose
   more than they save.
3. Expert kernels (9.4 ms at B=8): 59% VALU busy, 38% occupancy. Gains need fewer VALU ops per weight (codebook
   decode ~1.75 ops per weight, scales ~0.5 after packed math). No item yet that passes the budget rule.

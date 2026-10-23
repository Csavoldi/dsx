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

## Demo: 8 agentic sessions at 256K context, ~92 tok/s each (staged by lxe for later)

Measured on rental 3 (2026-10-23 ~03:00 PT): 8 sessions share a ~258K-token prefix of source files, each asks for a
different file rewritten in full; prompt-lookup speculation gives 91.5 tok/s per session (732 total) against 64.2
plain; 12 sessions 74.3 (891 total) against 56.8. The 258K prefix prefills at ~2,460-3,100 tok/s (~105 s).

Reproduce (commit at or after this section):
1. Rent `gpu-mi325x1-256gb` in nyc2, image `gpu-amd-base` (amd/box/RENTALS.md has the API steps; ~3 min to active).
2. On the box: `bash setup.sh` (amd/box/setup.sh: ROCm dev, Rust, Python), then the fetch script
   (~/.cache/v2work/amd_fetch2.sh locally: dsx-mxfp4 + checkpoint config/tokenizer/Engram shards 47-48, with
   restarts on stalls; then `hf download deepseek-ai/DeepSeek-V4.1-Flash-1017 config.json`, which the include list
   misses). rsync this repo to /root/dsx and `cargo build --release -p dsx-amd` (~35 s).
3. Start the dev runner with room for the slots at 256K:
   `DSXA_RESERVE_GB=24 DSXA_WEIGHTS=/data/models/dsx-mxfp4/DeepSeek-V4.1-Flash-1017-dsx-mxfp4-00001-of-00006.gguf
    ./target/release/dsx-amd dev --model /data/models/v41-1017 --rank /data/models/dsx-mxfp4/expert-rank.json
    --ctx 262200` (loads in ~2.5 min).
4. `amd/box/devcmd.sh "lspec 8 240 512"` prints plain and speculative tok/s per session and the first session's
   output (it should be the requested file with `// NOTE:` comments); `lspec 12 240 512` for 12 sessions.
For a client-facing demo use `dsx-amd serve --slots 8 --batch-ctx 262144 --ctx 262200` (same environment), the
OpenAI-compatible server on port 8090 with continuous batching and prefix reuse.

## Single-session expert products: sweep and counters (rental 6, 2026-10-23 04:22-04:44 PT)

Sweep (amd/results/single-x1u-waves-2026-10-23.txt, `prof 4096`, two alternations): prefetch depth X1U 4 / 6 / 8 is
flat (gate/up 39.5-40.5 us per layer at TARGET_WAVES 2048); more waves are slower (48-49 us at 4096, 54-56 us at
8192, from the k-split). The default stays best; the sweep item is killed.

Counters (amd/results/counters-2026-10-23/, rocprofv3 on gemv_jobs in the eager decode step; the gate/up launch is
the 864-block grid, 3,456 waves, 88 VGPRs, 38.9 us; down is the 1,120-block grid, 23.1 us):
- Achieved bandwidth: FETCH_SIZE 61.4 MB per gate/up launch in 38.9 us = 1.58 TB/s (down: 43.1 MB / 23.1 us =
  1.87 TB/s).
- VALU work: SQ_INSTS_VALU 7.37 M per launch = 2,133 per wave; a wave covers 16 rows x 2,560 k = 640 weights per
  lane, so 3.3 VALU instructions per weight (the column-rotated single-token tile: 3-bit unpack, FP8 MFMA operand
  prep, 16-column scale flush). At 4 cycles per wave64 VALU instruction on 1,216 SIMDs that is 11.5 us of pure ALU
  time per launch, exactly the time 57.6 MB takes at 5 TB/s. VALUBusy 27% (~10.5 us of 38.9) is consistent.
- Memory pipeline: SQ_INSTS_VMEM_RD 328,800 per launch = ~187 bytes per wave-level load: the weights are fetched as
  3 separate dword loads per lane per 128-k unit (NDW = 3) plus predicated 8-byte activation loads. MemUnitStalled
  ~0: the memory unit is not the limit.
- Residency: per wave lifetime SQ_WAVE_CYCLES / SQ_WAVES = 15.7 K cycles (~7.5 us), against a 82 K-cycle launch:
  ~2.2 waves resident per CU on average (OccupancyPercent reports 24.5%). SQ_WAIT_INST_ANY is 11% of wave cycles.
- Limit: each wave runs its 20 units back to back with ~430 VALU cycles per unit, so the launch is set by per-wave
  ALU latency at low residency, and by the ALU floor (11.5 us) even at full residency. The lever is fewer VALU
  instructions per weight in the single-token path, not more bytes in flight (which the X1U sweep confirmed).

Next step for whoever resumes (single session, the largest gap: 3.7x):
1. A single-token expert tile with fewer VALU instructions per weight: load each lane's weights as one dwordx3
   (or a repacked 16-byte unit) instead of 3 dword loads, decode the 3-bit indices with fewer bit operations
   (bit-plane layout repacked at load so one shift + one and_or yields each byte group), and flush the 16-column
   scale accumulation less often (per 2 units). Budget: from 3.3 to ~2.0 VALU per weight puts the ALU floor at
   ~7 us and, with ~4 waves resident per CU, gate/up at ~15-20 us (from 39 us): ~0.8-1.0 ms per token (+10-12%).
   Confirming measurement: a standalone single-token tile microbenchmark (like dgemv_bench) on one 2304 x 5120
   3-bit expert, against the current tile. Kill if it does not reach 3 TB/s.
2. Raise residency: 3,456 waves for 864 blocks gives ~11 waves per CU over the launch but only ~2 resident at once;
   check the dispatcher limit (one block per CU per cycle on 304 CUs, 864 blocks) with a block size of 512 or 1024.
Not measured (bug, fixed offline): the prep_t mode-0 wave-reduction A/B. All four bsweeps failed in load_slot with
"pack_fp4_rows: 144 groups without an exact FP4 scale": load_slot packed pos/ratio + 1 rows, one past the rows a
prefill writes, and the preceding `prof` runs had filled the caches with random values. It now packs pos/ratio
rows (and skips empty packs); the change in prep_t (d20c4c1) is unmeasured.

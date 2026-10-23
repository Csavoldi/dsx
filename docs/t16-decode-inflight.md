# T16 decode GEMV: what caps it (5090, ncu, 2026-10-23)

Kernel: `t16d_gemv_K2` (t16.cuh), one warp per (pick, pair of 16-column tiles, k-split s), 4 warps a block; a decode
layer call (6 picks) launches 432 blocks for gate / up (S = 4) and 720 for down (S = 3). S stays: it fixes the order
of the float sum over 256-groups, and the output must stay bit-identical (`gpu_t16 t16_decode_bits`).

Measured (sudo ncu, `gpu_t16 t16_decode_speed`, K 1.5 gate / up, 64 experts so nothing stays in L2):

| | original | cp.async (b9434a0) |
|---|---|---|
| time / DRAM read | 34.5 us, 401 GB/s (23% of peak) | 21.0 us, 640 GB/s (36%) |
| L2 hit / L2 read | 10%, 418 GB/s (streaming; no re-reads) | 9%, 687 GB/s |
| sector use of the weight loads | 90% (LDG.32 x 3 a lane) | 100% (LDGSTS 16 B) |
| warps per SM / issue use | 10.0 / 18% | 10.0 / 34% |
| top stall | long scoreboard 11.5 | long scoreboard 4.3 |

DRAM load latency, one thread chasing a random cycle through 1 GiB (`gpu_latency`): **320 ns** unloaded. Little's law
for full bandwidth: 1.79 TB/s x 320 ns / 170 SMs = **3.4 KB in flight per SM** (~6 KB at a loaded ~600 ns).

- **Original:** 401 GB/s x 320 ns / 170 = 0.75 KB per SM in flight, ~75 B per warp: less than one iteration's 96 B.
  The SASS interleaves each iteration's three LDGs with the previous iteration's dp4a, so a warp has about one
  iteration of loads in flight; with 10 warps per SM that caps it. Fix: more bytes in flight per warp.
- **cp.async:** a warp has the next group's 1.5 KB in flight while it computes this one (10 warps: up to 15 KB per SM,
  4x Little's need), and long-scoreboard stalls fell 11.5 -> 4.3, so bytes in flight no longer cap it. Throughput now
  scales with warps per SM: the same kernel with more picks per launch (T16_SPEED_T) reaches 630 / 860 / 1006 / 1111
  GB/s at 6 / 12 / 24 / 48 picks (36 / 49 / 57 / 64% of peak; 10 / 20 / 40 warps wanted per SM, 69 registers allow ~29).
  Splitting the dp4a chain into four changed nothing (60.3 vs 61.3 us a layer call), so it is not one dependency
  chain; it is too few warps to hide a group's ~1,100 instructions of LDS -> funnel shift -> IMUL -> PRMT -> IDP latency.
  Fix: more warps per launch without changing S.

Next (targets warps per SM): split each group's 16 tile rows between two warps of a block; each sums its 8 rows into an
exact integer partial, the second hands its partial over in shared memory behind a 64-thread named barrier, the first
adds and does the float step as before. Prediction: the 12-pick numbers, t16d_gemv_3 21 -> ~15.6 us (~860 GB/s, 49%),
a K 1.5/2 layer call 60 -> ~48 us on the 5090. On the 3090 (82 SMs: 21 warps per SM now, ~29 resident at most) the
gain is smaller. Kill: below 60% of DRAM peak on .102 after it, or no teacher-forced decode gain there.

**Tried, negative: two warps per tile pair (8 rows each).** Exact (bit hash unchanged), with one `__syncthreads` a
group to hand the second warp's integer partial over. Named barriers (`bar.sync id, 64`) first capped it at one
block per SM (ncu `launch__occupancy_limit_barriers` = 1, and the driver then sized shared memory at 8 KB): 145 us a
layer call. With `__syncthreads` (idle warps loop along): occupancy 21 -> 41%, but t16d_gemv_3 21.0 -> 25.4 us (640 ->
535 GB/s), long-scoreboard stalls 4.3 -> 12.6, warp-inst per byte 0.69 -> 0.79; a layer call 60.3 -> 62.2 us. Halving
a warp's rows halved its bytes in flight, so the SM's total stayed the same, and the split added instructions. So
warps per SM alone is not the cap; next: per-instruction stall sampling of the cp.async kernel.

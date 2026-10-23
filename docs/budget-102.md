# Decode resource budget on .102

Where a decode token's time and bytes go on .102 (2x RTX 3090, i7-12700K, 128 GB DDR4), from measurements, and what
each candidate lever can win by that budget. Every proposal states its predicted gain, the measurement that confirms
it and the result that kills it. Build: T16 IQ4_NL (a3cf51b + prefill cap 512), 3000-token prompt, 600 decode tokens,
greedy, 2026-10-23.

## 1. Measured inputs

| Quantity | No deferral | `DSX_DEFER=3` (production) | Source |
|---|---:|---:|---|
| Token time | 30.8 ms (32.5 tok/s) | 27.9 ms (35.9 tok/s) | DSX_EVPROF / gen |
| DRAM read | 13.4 GB/s = **412 MB/token** | 17.8 GB/s = **496 MB/token** | perf uncore_imc data_read, last 12 s |
| DRAM write | 0.3 GB/s | 0.3 GB/s | same |
| CPU picks | 19.9% (~48/token) | 22.4% | gen's decode line |
| CPU worker forward | 11.15 ms/token | 5.29 immediate + ~7.8 deferred | EVPROF split / gen |
| DRAM read rate while the CPU computes | ~37 GB/s | ~37 GB/s | uncore over the forward |
| RAM read ceiling, 14 threads | 42.4 GB/s | | cpubench small (sc102) |
| Expert size (T16) | 7.42 MB (layers 0-19), 9.63 MB (20-39) | | t16::matrix_bytes |

GPU0 stages per token (DSX_EVPROF, ms; no deferral / `DSX_DEFER=3`):

| Stage | No deferral | DEFER=3 | Detail |
|---|---:|---:|---|
| 0 attention segment | 13.09 | 13.41 | engram 1.12/1.25, hc+norm 0.80/0.82, q 3.23/3.28, kv 0.54/0.56, compress+indexer 0.69/0.71, sparse_attn 1.08/1.11, o 4.74/4.77, hc post/pre+norm 0.89/0.91 |
| 1 router | 0.77 | 0.80 | |
| 2 publish + GPU0's experts | 3.08 | 3.33 | ~2.1 ms of kernels: t16d_gemv_4 60 x 11.8 us, gemv_3 40 x 15.3, gemv_5 20 x 10.3, prep 120 x 2.0, post_down 40 x 3.8 (nsys) |
| 3 shared expert | 2.54 | 2.59 | Q8_0 at ~650 GB/s |
| 4 wait for the CPU's experts | 8.01 | 4.34 | |
| 5 wait for GPU1 | 0.56 | 0.57 | |
| 6 combine | 0.30 | 0.35 | |
| 7 combine -> next layer | 0.16 | 0.16 | |
| outside the layers (head, logits, adapt step, host) | ~2.3 | ~2.4 | token time minus the stages; not split yet |

Dense GEMVs: q_b / o_b at 766-796 GB/s (84% of the 3090's 936 GB/s). Busy clocks: GPU0 median 1,740 MHz (P2, 250 W
cap), GPU1 1,950: clocks are not limiting (FINDINGS 8d). GPU0 runs ~1,700 kernels per token; the host enqueues them
ahead at ~5 us each (~8.5 ms of the main thread's time per token) and waits in cuStreamSynchronize the rest.

**Which token.** The stage table comes from a DSX_EVPROF run (a3cf51b + prefill cap 512, 600 tokens, with EVPROF's
timed events on): 27.9 ms/token with DEFER=3. Production-equivalent decode without instrumentation (pf102, 9be924f,
400 tokens, DEFER=3, prefetch off) measures 26.8 ms (37.35 tok/s). The ~1.1 ms difference is most likely EVPROF's own
cost (8 timed events per layer, read back every token), plus different builds and text; it has not been measured
separately. Gains below are in ms, which apply to the 26.8 ms production token; percentages are of 26.8 ms.

## 2. Budget per resource (production, `DSX_DEFER=3`, 27.9 ms/token under EVPROF)

| Resource | Capacity | Used per token | Idle | When it is idle |
|---|---|---|---|---|
| GPU0 SMs / VRAM | 936 GB/s | ~21.2 ms busy in the layers + ~2.4 outside | ~4.3 ms (stage 4) | while it waits for the CPU, inside MoE |
| GPU0 link (x16) | 24.6 GB/s | ~70 MB swaps (~2.8 ms) + ~2 MB exchange | ~90% | always; **every byte it moves from the arena is also a DRAM read** |
| GPU1 SMs | | its resident experts (stage 5 waits 0.57 ms) | mostly | outside its per-layer expert calls |
| GPU1 link (x4) | 3.3 GB/s | ~1.6 MB activations (~0.5 ms) | ~98% | always; too slow for expert bytes |
| DRAM | 42.4 GB/s | 496 MB (CPU experts ~430, swaps ~70) | ~55% on average | only while the CPU's expert pool is idle; with deferral the CPU also computes during the next layer's attention |
| CPU expert pool (14 threads, P-cores 0-6) | | ~13.1 ms busy | ~53% | between layers; it is RAM-bound when it runs |
| Main thread (CPU 14) | | ~8.5 ms of launches | ample | runs ahead of GPU0 |

Floors: the CPU's bytes alone need 496 MB / 42.4 GB/s = 11.7 ms per token, which hides inside GPU0's ~23.6 ms if
overlapped. GPU0's own work is the binding resource in production: with today's kernels and placement the ceiling is
~23.6 ms/token (~42 tok/s) even if the CPU wait went to zero. Without deferral the CPU is binding (8.0 ms of wait).

## 3. The critical path of one layer (production)

attention (0.335 ms, GPU0, VRAM-bound GEMVs) -> router (0.02) -> publish -> [GPU0 experts + shared expert, 0.148 ms]
in parallel with [CPU: immediate picks, ranks 0-2, ~0.235 ms per expert, RAM-bound] -> wait (0.108 ms average) ->
GPU1 wait -> combine. The CPU computes the layer's deferred picks (ranks 3-5, ~0.195 ms per layer on average) during
the next layer's attention, which leaves ~0.14 ms of slack per layer there.

Consequences for levers:
- Attention, router, combine and the work outside the layers are on the critical path **one for one**.
- Stages 2 and 3 (GPU0's experts, shared expert) overlap the CPU's immediate experts. Speeding them up shortens the
  token only on layers where GPU0, not the CPU, ends last: about half of them (a layer has no immediate CPU pick with
  probability ~0.78^3 = 0.47). Count roughly half of any stage 2/3 saving.
- Shrinking attention also shrinks the window the CPU's deferred work hides in; with ~0.14 ms of slack per layer,
  attention can lose up to ~40% before deferred work becomes critical.

## 4. DSX_PREFETCH, against this budget (negative result)

Copying layer L+1's predicted CPU experts into GPU0 (branch pf-prefetch) made decode slower in every configuration
(no deferral 33.1 -> 25.3 / 22.2 tok/s at 1 / 2 copies per layer; with DEFER=3 37.4 -> 28.3 / 20.7). Quality was
unchanged (m=2 decode eval KL 0.2048 vs 0.2082), so the loss is all cost:
- **DRAM:** a copy reads the same bytes the CPU would. One copy per layer adds ~35 x 8.5 = ~300 MB per token, about half
  of it wasted (predictions not picked), to a 412-496 MB budget that is already read at ~37 GB/s whenever the CPU
  runs. Measured: at m=1 the CPU's share of picks fell 0.71x but its busy time only 0.85x, so each remaining CPU expert
  ran ~20% slower (~32% with deferral).
- **Link time:** 300 MB at ~23 GB/s is ~13 ms, the whole attention segment; at m=2 (~520 MB) it overflows it.
- **GPU0 SMs:** the token grew ~9 ms more than the CPU's savings. The repack ran a ~9.4k x 3-block grid per copy and
  pf_copy kept 24 blocks resident, both alongside attention, plus a router GEMV and ~7 launches per layer.

What the literature agrees on (KTransformers SOSP '25, Fiddler ICLR '25, Pre-gated MoE ISCA '24, MoE-Infinity,
ProMoE, HOBBIT, AdapMoE): at batch 1 with a CPU that computes experts, moving weights does not pay. Fiddler's rule
`compute on the GPU only if cpu_lat > gpu_lat + transfer_lat` keeps batch-1 experts on the CPU; ProMoE and HOBBIT lost
or broke even against CPU compute in llama.cpp decode; AdapMoE's prefetch added nothing once a cache existed;
KTransformers moves no weights and overlaps instead (expert deferral, +33% decode). None models the DRAM contention
between CPU expert reads and PCIe copies, which is what this setup hits. Copies pay only when the expert stays resident
and is reused across tokens, which GPU0's adaptive cache already does.

## 5. Levers, ranked by predicted gain on production decode

| # | Lever | Predicted gain | Confirming measurement | Kill criterion |
|---|---|---|---|---|
| 1 | q_b / o_b weights below Q8_0 (e.g. 6-bit): 8.0 ms of the critical path at 84% of peak bandwidth, so bytes are the only big lever there | -25% bytes = **~2.0 ms (7%)** | decode eval KL / top-1 on the Lucebox eval, then EVPROF q + o | KL up > 0.005 over 0.2082, or q + o < 1.5 ms faster |
| 2 | `DSX_DEFER=2` re-measured on the T16 build (8b measured "no faster" on the older 2-bit build, whose CPU was slower): deferred work per layer would grow to ~0.26 ms, still under attention's 0.335 | wait 4.34 -> ~3.0 ms: **~1.3 ms (~5%)** | interleaved A/B DEFER=3 vs 2, 2 rounds, plus the decode eval | < 2% faster in both rounds, or KL up > 0.01 |
| 3 | Engram's 1.12-1.25 ms: its rows depend only on the tokens, so if the stage waits on host-gathered rows they can be fetched at the start of the token | up to **~1.1 ms (4%)** if transfer-bound | nsys of the engram stage: kernel time vs waiting | engram kernels' own time >= 80% of the stage |
| 4 | Small-kernel fusion in hc / norm / rope (1.7 ms, ~1 ms of in-stage gaps per token in 8d) | **0.5-1.0 ms (2-3.5%)** | nsys: gaps plus redundant reads in these stages | < 0.4 ms of gaps and redundant work to remove |
| 5 | GPU0's decode-expert chain (stage 2: ~120 GEMV launches at ~45% of bandwidth, 2.1 ms of kernels in 3.3) | 3.3 -> ~2 ms in the stage, **~0.6 ms on the token** (overlap) | EVPROF stage 2 and stage 4 | stage 4's wait grows by the same amount (no net) |
| 6 | Shared expert at 650 -> ~790 GB/s | 0.45 ms in the stage, **~0.2 ms on the token** | EVPROF stage 3 | < 0.15 ms on the token |
| 7 | The ~2.4 ms outside the layers (head, logits download, adapt step) | unknown | EVPROF / nsys split of the token's ends | |
| 8 | Engram (stage 0a, 1.12-1.25 ms): engram.wkv is 25,600 x 6,144 FP8 E4M3 with 32x32 E8M0 scales in the checkpoint; dsx stores F16 (315 MB/layer) and its GEMV already runs at ~875 GB/s (360 us x 2 layers). E1: keep it FP8 (gemv_fp8b32, scales expanded to 1x32 rows; E8M0 scales are powers of two, so the products are the same). E2: the host's join on the row-gathering thread idles GPU0 ~0.4 ms before layer 1 only; gather layer 1's 24 rows first on persistent threads (now 16 thread spawns a token, layer 14's rows gathered before the thread returns) | E1 ~0.36 ms (1.3%) + 315 MB of GPU0 VRAM; E2 up to ~0.4 ms (1.5%) | E1: nsys engram_kv <= 200 us per call, decode eval KL within 0.001; E2: DSX_PROF's "host: engram join wait" per token | E1: gemv_fp8b32 < 650 GB/s on this shape; E2: join wait < 0.15 ms/token |
| 9 | Split q_b / o_b between GPU0 and GPU1 (tensor-parallel halves; whole layers on GPU1 cannot help batch-1 decode, since the GPUs would run in turn). Each split matrix puts a GPU0 -> GPU1 -> GPU0 round trip R over the x4 link (host-staged, stream memory ops) on the critical path, twice per layer. Per layer, q (~81 us) gains only if R < ~40 us (half its work) and o (~118 us) only if R < ~59 us. The aux-expert handoff, the same kind of round trip, returns GPU1's results ~14 us after GPU0's own 148 us of experts; with GPU1's expert compute that bounds R at roughly 80-110 us. Half of q / o is ~3.1 GB of Q8_0 weights, which evicts ~370 of GPU1's ~2,181 resident experts: ~3 more points of picks on the CPU, ~+0.9 ms of wait | per token 40 x (100 - 2R) us minus ~0.9 ms of CPU wait: at R ~ 90 us **-4.1 ms (loses)**; break-even at R ~ 39 us; best case (R ~ 15 us) **+1.9 ms** | R from the existing EVPROF aux events (GPU1's per-layer wait / htod / run / dtoh next to GPU0's publish) or a ping-pong bench on .102 | R >= 40 us |

Not proposed, measured negative: copying predicted CPU experts into GPU0 (section 4); DSpark speculative decoding
(verify passes of 2 / 3 / 4 tokens cost 1.55 / 2.08 / 2.34 decode steps, because the CPU's work grows with the union of
the window's experts; FINDINGS 9); routing bias toward VRAM-resident experts (+4 tok/s for 2 points of top-1).

A lever that moves bytes off the CPU without adding DRAM traffic would need either reuse (an expert copied once and
then hit on later tokens: the adaptive cache) or fewer bytes per CPU expert (HOBBIT's low-precision copies for
low-gate picks; RAM is nearly full, so that means a lower-bit quant of the CPU-resident experts, gated on the eval).

## 6. Prefill and server turns: gpu-unlock

gpu-unlock (HW Tuning's REPORT.md, Task 2) lifts GeForce's half rate on FP16 / BF16 tensor MMA with FP32
accumulation (GA102 whitepaper: 142 dense TFLOPS with FP16 accumulate, 71 with FP32). Users report FP32-accumulate
GEMMs on a 3090 / 3090 Ti going from ~60-87 to ~90-118 TFLOPS. At .102's 230 W cap that is ~1.5x on compute-bound
GEMMs; 2x (the whitepaper ratio) is the ceiling.

What it touches in dsx, from the sm_86 SASS of every dsx kernel and the kernel names in the nsys traces: cuBLAS's
`ampere_s16816gemm_fp16_*` / `s1688gemm_fp16_*` (FP16 in, FP32 accumulate: prefill's dense and dequantized-expert
GEMMs) and dsx's `sparse_attn_tc` / `sparse_attn_tc_fast` (`HMMA.16816.F32.BF16`, prefill attention). The routed-expert
kernels (`moe_gu_fast`, `moe_down_fast`, `xgemv`) use int8 `IMMA`, which was never limited. **Decode does not change:**
the 64 decode steps of Dev 1's dec64 trace (2.28 s) run none of these kernels on either GPU, and decode's attention is
`sparse_attn_tile` (no MMA).

Measured exposure in Dev 1's existing nsys traces (.102, T16 build). Each GPU0 phase is split at its last cuBLAS GEMM,
so the decode steps after it are excluded; "link idle" = GPU0's host-to-device copies idle (the part of a kernel no
expert stream can hide):

| Trace | Prefill wall | GPU0 busy / link busy | FP32-acc HMMA on GPU0 (of it with link idle) | on GPU1 |
|---|---|---|---|---|
| lp3000: 3000-token prompt, hugepage arena (3 GPU0 phases) | 4.94 s (5.2 s with host gaps) | 3.67 / 3.09 s | 1.43 s, 39% of GPU0 busy (0.38 s) | 0.42 s |
| srvturn, turn 2 (1.89 s prefill; likely the ~300-token turn of 8d) | 1.89 s | 0.94 / 1.37 s | 0.44 s (0.11 s) | 0.16 s |
| srvturn, turn 1 (likely the ~60-token turn) | 0.79 s | 0.46 / 0.60 s | 0.19 s (0.03 s) | 0.09 s |

Bounds at speedup s on these kernels (saved = time x (1 - 1/s)): the low end counts only GPU0's link-idle HMMA time,
the high end all of GPU0's HMMA time (true if GPU0's compute, not the link, bounds the overlapped stretches; the link
busy time stays a floor). GPU1's share is left out, since its results are not known to be on the critical path.

| # | Lever | Predicted gain | Confirming measurement | Kill criterion |
|---|---|---|---|---|
| 10 | gpu-unlock (needs a full power-off and lxe's OK; .102 now runs kernel 7.0 and driver 615.71.10, the versions it was tested on) | 3000-token prefill: s = 1.5 **0.13-0.48 s (2.5-9%)**, s = 2 0.19-0.71 s (4-14%). ~300-token turn: s = 1.5 **35-146 ms (2-8%)**, mostly link-bound (link busy 72% of the prefill), so the low end is likelier. ~60-token turn: ~0: its GEMMs have M ~ 60, under the ~76 FLOP/byte where a 3090's FP32-accumulate GEMM turns compute-bound. Decode: 0 | cuBLAS FP16 -> FP32 GEMM TFLOPS at a prefill shape (M = 3000, K = 5120) before and after (one process per config, 230 W); then the lp3000 and srvturn traces again: s16816 kernel time and prefill wall time; decode teacher-forced tok/s unchanged | GEMM speedup < 1.2x at 230 W, or the 3000-token prefill < 2% faster |

HW Tuning's earlier 0-5% assumed prefill was bound by the expert stream. The traces cut against that for long prompts:
in lp3000 GPU0 computes with its link idle for 1.47 s of 4.94 s, and 39% of GPU0's kernel time is FP32-accumulate HMMA.

## 7. Free math per weight byte in decode, and the bits each tier can drop to

Batch-1 decode does ~2 FLOPs per weight while a 3090 needs ~76 FLOPs per byte to be compute-bound, so the tensor
cores only help with many tokens per weight read (prefill, batching). What decode can use is the ALU and issue slots
its kernels leave idle while waiting on memory: spent on decoding fewer bits per weight, they replace bytes. This
section measures that budget per tier.

**Method.** GPU: ncu (sudo on the workstation, 2026-10-23) on the 5090 for the real decode kernels from the existing
benches (gpu_q8 q8_variants 32768 x 1280; gpu_t16 t16_decode_speed, 6 experts of 64 per call, K 1.5 / 2; gpu_engram_fp8):
warp instructions executed per DRAM byte read, issue-slot use, stall reasons. Instructions per byte carry over to the
3090: the T16 kernels' sm_86 and sm_120 SASS differ by <= 5% (gemv_q8r's sm_86 SASS is 1.56x larger, so its 3090
figure is uncertain). The 3090 issues 82 SMs x 4 schedulers x 1.74 GHz (8d's busy clock) = 571 G warp-instructions/s;
at a practical 880 GB/s (gemv_f16 reaches 871) that is **0.65 warp-instructions (20.8 thread-instructions) per byte**,
the most a kernel may execute and still run at full bandwidth. CPU: cpubench t16 on the i7-13700 (Raptor Lake P-cores,
the same AVX2 / AVX-VNNI units as the 12700K's Golden Cove), 1 thread and 2 threads on one P-core, scaled by the clock
(~4.7 vs ~5.0 GHz all-core; 0.94) against .102's RAM share per P-core: 42.4 GB/s / 7 pool cores = **6.06 GB/s**.

| Kernel (tier) | 5090: GB/s (% of DRAM peak) | issue use | top stall | warp-inst per byte | 3090 issue use at 880 GB/s |
|---|---|---|---|---|---|
| gemv_q8r (dense q / o, Q8_0, 8.5 bpw) | 1280 (73%) | 27% | long scoreboard 21 | 0.297 (10.1 thread-inst per weight) | 46% (up to ~70% if sm_86 runs 1.56x the instructions) |
| gemv_f16 (engram, 16 bpw) | 1510 (88%) | 10% | long scoreboard 82 | 0.109 | 17% |
| gemv_fp8b32 (engram E1, 8 bpw) | 1405 (81%) | 17% | long scoreboard 14 | 0.175 | 27% |
| t16d_gemv_3 (GPU experts, gate / up K 1.5) | 391 (22%) | 18% | long scoreboard 12; occupancy 21% | 0.66 (4.0 thread-inst per weight) | **102%** |
| t16d_gemv_4 (GPU experts, down K 2) | 763 (43%) | 31% | long scoreboard 10; occupancy 34% | 0.55 (4.4 thread-inst per weight) | **84%** |

| CPU T16 kernel, per P-core (2 threads) | 13700 | at .102's clock | .102 RAM share per core | binds |
|---|---|---|---|---|
| K 1.5 | 5.8 GB/s = 30.9 G weights/s | 29.0 G weights/s | 32.3 G weights/s | **compute** |
| K 2 | 7.2 GB/s = 28.8 G weights/s | 27.1 | 24.2 | RAM (12% compute spare) |
| K 2.5 | 7.7 GB/s = 24.6 G weights/s | 23.1 | 19.4 | RAM (19% spare) |
| UD IQ1/IQ2 codebook mix (~1.6-2.1 bpw) | 4.8 GB/s | | | compute (slower than T16) |

**Lowest bits per weight with no slowdown** (below it the decode, not the bytes, sets the time; b_min = thread-inst
per weight x 8 / thread-inst per byte available):

- **GPU-resident experts:** T16's trellis costs 4.0-4.4 thread-instructions per weight, so b_min = **1.5-1.7 bpw** on
  the 3090: K 1.5 is already at the issue limit. Bits are not today's limit, though: the T16 GEMVs run at 22-43% of
  bandwidth on the 5090 (8d: ~45% on .102) with low occupancy and long-scoreboard stalls, i.e. too few loads in flight
  (432-720 blocks of 4 warps per call). Fewer bits gain ~nothing until that is fixed (lever 5).
- **CPU-resident experts:** compute equals .102's RAM share at **~1.7 bpw** (48.5 / b G weights/s of RAM vs ~28 G
  weights/s of T16 decode per core). K 1.5 is compute-bound already, so fewer bits gain nothing on the CPU; the
  ~12-19% spare at K 2 / 2.5 is the only free math there. No known decode is cheaper per weight on AVX2 than T16
  (the IQ codebooks measured 4.8 GB/s per core).
- **Dense q_b / o_b (now Q8_0, 8.0 ms of the critical path):** the one tier with real headroom (46% issue at full
  bandwidth). A trellis at 4.4 thread-inst per weight stays bandwidth-bound down to ~1.7 bpw; a scalar LUT format
  (IQ4_NL / NF4-like) at ~6-8 thread-inst per weight down to ~2.3-3 bpw; gemv_q8r's own density (10.1) to ~3.9 bpw.

**Decode tok/s from the dense tier** (production 26.8 ms/token, 37.3 tok/s before tonight's power caps; bytes of q / o
scale the 8.0 ms, the kernel staying at today's ~84% of bandwidth): 6.5 bpw (Q6 planes) -1.8 ms (+7%); 4.5 bpw
(IQ4_NL-like) -3.8 ms, **23.0 ms = 43.5 tok/s (+16%)**; 3 bpw trellis -5.1 ms, **21.7 ms = 46 tok/s (+23%)**.
Expert tiers: ~0 from fewer bits (GPU: parallelism-bound; CPU: compute-bound).

**Schemes inside the budget** (each gated on the Lucebox decode eval's KL and top-1 against the run-to-run spread):
dense q / o: trellis (QTIP / EXL3 family, i.e. T16 at 3-4 bpw), non-uniform 4-bit LUT scalars (IQ4_NL, NF4), HQQ /
Q4_K-style group scalars, E8-lattice codebooks (QuIP# E8P, ~2 bpw, bit-op decode plus a 256-entry table); GPU experts
below 1.5 bpw: vector codebooks with small shared-memory tables (AQLM 2x8, VPTQ) at ~1-2 thread-inst per weight, only
after the T16 kernel reaches bandwidth; CPU experts: none cheaper than T16 known on AVX2 (vpshufb tables need <= 16
entries per lookup; IQ1/IQ2 grids measured slower).

| # | Lever | Predicted gain | Confirming measurement | Kill criterion |
|---|---|---|---|---|
| 11 | Dense q_b / o_b from Q8_0 to a 4.5 bpw LUT scalar or a 3-4 bpw trellis (row 1 extended: the issue budget allows it) | 4.5 bpw **-3.8 ms (+16%)**; 3 bpw trellis **-5.1 ms (+23%)** | ncu on .102 (sm_86): the new GEMV's warp-inst per byte <= 0.5 at >= 750 GB/s on 32768 x 1280 / 5120 x 8192; EVPROF q + o; teacher-forced decode tok/s | decode eval KL up > 0.005 (or beyond 2x the measured run-to-run spread), or the GEMV < 700 GB/s on .102 |
| 12 | Experts below ~1.7 bpw (either tier) | ~0 on decode speed (see above) | | not proposed for speed |

Still to measure on .102 (no new code; one window): the same three ncu runs (sm_86 instruction counts and issue use,
especially gemv_q8r's), and `CPUB_THREADS=2,14 cpubench t16` (~20 s, inside the thermal limits) for the per-core
compute rate at the 12700K's clock under the 125 W cap.

## 8. Lever status when DSX work paused (2026-10-23, 04:35 PT)

Branches (local in Dev 4's worktrees, committed, not pushed): `engram-fp8` 29a4506 (E1, E2), `t16-inflight` 93979a9
(T16 GEMV), `window-dev4` 58ede8d (github/main 3f14dde + both, plus the window script), `budget-102` (this doc).
All .102 results below are Dev 1's windows; all 5090 numbers are sudo ncu / event timings on the workstation.

| Lever | State | Evidence | To pick up |
|---|---|---|---|
| 1 / 11. Dense q_b / o_b below Q8_0 | not started | section 7: 46% issue use at full bandwidth, room down to ~4 bpw (LUT scalar) or ~1.7 bpw (trellis) | quantize q_b / o_b, kernel, decode eval KL; kill as in row 11 |
| 2. DSX_DEFER=2 | **killed** (Dev 1) | decode eval KL (8 threads, 2046 tokens): defer 3 0.2370, 2 0.2600, 1 0.3062 vs none ~0.208; speed came from free-running gen, not valid for numeric changes | -; note production defer=3 itself costs ~0.029 KL (flagged to lxe) |
| 3 / 8. E1: engram wkv as FP8 | **pending one .102 window** | window 24 (teacher-forced, 1023 steps): +1.6% tok/s (28.99 -> 29.48) but KL 0.139519 -> 0.141207, traced to gemv_fp8b32's summation order; fixed by gemv_fp8b32_o (29a4506): gemv_f16's arithmetic on the FP8 bytes, 0 of 25600 outputs differ in any bit on the checkpoint's layer-1 wkv, 104 us vs 198 us on the 5090 | `window-dev4`, `EVAL_ARGS="<window 24's>" bash bench/w4_102.sh`, step 4 arm e1: KL must equal base exactly, ship if >= +1% |
| 3 / 8. E2: persistent engram gather | **parked** (Dev 1) | window 24: join wait 0.77 ms -> 0, +0.49% over E1 (kill line 0.5%) | re-test only after E1 ships (DSX_ENGRAM_PERSIST=1) |
| 4. Small-kernel fusion | Dev 1's (branch small-kernels, window 26) | | Dev 1's results |
| 5. T16 decode GEMV (GPU experts) | **pending one .102 window** | bit-identical (gpu_t16 t16_decode_bits 1148e32a0aaa9120); 5090 layer call (6 experts) 95.4 -> 54.2 us (K 1.5/2), 88 -> 62.3 us (K 2/2.5); t16d_gemv_3 23 -> 53% of DRAM peak, issue-bound at 3.9 thread-inst per weight (docs/t16-decode-inflight.md on t16-inflight: Little's-law analysis, stall sampling, negative tries). Decode gain predicted +0-2% (GPU0 waits ~4.3 ms on the CPU under defer 3); prefill / verify / GPU1 gain more | same window, steps 1-2 (DSX_T16_GEMV=0 vs 1, ncu sm_86) and step 4 arm t16; kill: < 60% of DRAM on .102 or no decode gain |
| 6 / 7 | not started | | |
| 9. q / o split onto GPU1 | not proposed | break-even round trip R ~ 39 us, estimated 80-110 us | measure R (ping-pong over x4) first |
| 10. gpu-unlock | **waiting on lxe** (enables it 10-24 with the thermal fix; no power-off before) | section 6: 3000-token prefill 2.5-9%, ~300-token turn 2-8%, decode 0 | cuBLAS FP16 -> FP32 GEMM before / after at M 3000, K 5120; re-trace lp3000 and srvturn |
| Section 7 sm_86 counts | **pending the same window** | 5090 ncu only; gemv_q8r's sm_86 SASS is 1.56x larger | w4_102.sh steps 2-3 (sudo /usr/local/cuda-13.4/bin/ncu; cpubench t16 at 2 / 14 threads) |
| DSX_PREFETCH | **killed** (section 4, FINDINGS 9) | decode 33.1 -> 25.3 / 22.2 tok/s | - |
| Build audit (sm_86 / sm_120) | **dropped** at the pause | partial: no assert / printf in any kernel; no hot decode kernel spills or uses local memory except router_topk (32 B frame); dequant_f16 (prefill) has a 128 B frame with 273 STL in its SASS; gemv_q6g8 spills (unused); the CALLs in rmsnorm, t16d_gemv, sparse_attn_tc, router_topk were not yet traced (likely --prec-div slow paths) | resume from these |

`bench/w4_102.sh` (window-dev4) was checked locally for steps 1-3 (kernels, ncu, cpubench); step 4 needs .102's model.
Per the pause, no .102 window was requested to run.

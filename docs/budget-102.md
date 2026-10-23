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
| 9 | Split q_b / o_b between GPU0 and GPU1 (tensor-parallel halves; whole layers on GPU1 cannot help batch-1 decode, since the GPUs would run in turn). Each split matrix puts a GPU0 -> GPU1 -> GPU0 round trip R over the x4 link (host-staged, stream memory ops) on the critical path, twice per layer. Per layer, q (~81 us) gains only if R < ~40 us (half its work) and o (~118 us) only if R < ~59 us. The aux-expert handoff, the same kind of round trip, returns GPU1's results ~14 us after GPU0's own 148 us of experts; with GPU1's expert compute that bounds R at roughly 80-110 us. Half of q / o is ~3.1 GB of Q8_0 weights, which evicts ~370 of GPU1's ~2,181 resident experts: ~3 more points of picks on the CPU, ~+0.9 ms of wait | at R ~ 90 us: **-2 to -4 ms (loses)**; break-even needs R ~ 30 us; the best case (R ~ 15 us) is ~+2 ms after the CPU cost | R from the existing EVPROF aux events (GPU1's per-layer wait / htod / run / dtoh next to GPU0's publish) or a ping-pong bench on .102 | R >= 40 us |

Not proposed, measured negative: copying predicted CPU experts into GPU0 (section 4); DSpark speculative decoding
(verify passes of 2 / 3 / 4 tokens cost 1.55 / 2.08 / 2.34 decode steps, because the CPU's work grows with the union of
the window's experts; FINDINGS 9); routing bias toward VRAM-resident experts (+4 tok/s for 2 points of top-1).

A lever that moves bytes off the CPU without adding DRAM traffic would need either reuse (an expert copied once and
then hit on later tokens: the adaptive cache) or fewer bytes per CPU expert (HOBBIT's low-precision copies for
low-gate picks; RAM is nearly full, so that means a lower-bit quant of the CPU-resident experts, gated on the eval).

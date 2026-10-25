# DigitalOcean MI325X rentals

`gpu-mi325x1-256gb`, nyc2, image `gpu-amd-base`, $3.80/h. Times Pacific. The droplet and its account SSH key are
deleted at the end of every session.

| # | start | end | hours | cost | purpose |
|---|---|---|---|---|---|
| 1 | 2026-10-21 20:58 | 21:28 | 0.5 | $1.90 | dsx-amd baseline |
| 2 | 2026-10-21 21:46 | 2026-10-22 00:24 | 2.63 | $10.00 | batched decode, router fix, profiling |

Budget for the batched runner (lxe, 2026-10-23, relayed by Supervisor): $100 of new spend, counted from rental 3.

| # | start | end | hours | cost | running total (from #3) | purpose |
|---|---|---|---|---|---|---|
| 3 | 2026-10-23 00:23 | 03:20 | 2.95 | $11.21 | $11.21 | runner: FP4 slots, scheduler, prefill, small-batch kernels, speculation |
| 4 | 2026-10-23 03:27 | 03:30 | 0.05 | $0.19 | $11.40 | dense few-token product microbenchmark (RUNNER.md item 1) |
| 5 | 2026-10-23 03:53 | 04:11 | 0.30 | $1.14 | $12.54 | batched-step graphs vs eager, KL, speculation with graphs |
| 6 | 2026-10-23 04:22 | 04:44 | 0.37 | $1.39 | $13.93 | launch floor, single-session expert sweep + counters, then droplet destroyed (DSX paused by lxe) |
| 7 | 2026-10-23 08:22 | 10:10 | 1.80 | $6.84 | $20.77 | serving demo: API + Volta-style dashboard tunneled to 9090/9091, Pi gauntlets, sampled speculation, O_DIRECT startup (RUNNER.md) |

| 8 | 2026-10-24 15:06 | (running; as of 2026-10-24 21:01 PDT) | ~5.91+ | ~$22.45+ | ~$43.22+ | one MI325X, six-hour agentic optimization; final gauntlet 4/4 in 198.20 s, four-active median 106.53 TPS/session; SSH/API/Pi retained for lxe; receipts in `/home/lxe/Documents/Codex/2026-10-24/from-our-work-on-dsx-or/outputs/amd-six-hour` |

**Rental 8 retention:** lxe explicitly requested that SSH and the model/demo remain available. This rental stays active after the bounded optimization work, at $3.80/hour; the general end-of-session deletion rule above does not apply to this rental.

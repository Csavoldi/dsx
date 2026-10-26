# DigitalOcean MI325X rentals

Recorded October 21–24, 2026: `gpu-mi325x1-256gb`, nyc2, image `gpu-amd-base`, $3.80/h. Times Pacific.
The price and availability are historical, not a current provider quote. The droplet and its account SSH key are
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
| 8 | 2026-10-24 15:06 | 2026-10-24 22:41 (deletion confirmed) | ~7.57 | ~$28.77 | ~$49.54 | agentic inference optimization and demo; droplet 612312165 and rental SSH key deleted; source and results archived locally |

**Rental 8 closed:** released at lxe’s request. DigitalOcean returned 404 for both the droplet and its rental SSH key, and the account inventory contained no GPU droplets. The local SSH tunnel is closed. Cost is an elapsed-time estimate through the deletion recheck, not a final invoice.

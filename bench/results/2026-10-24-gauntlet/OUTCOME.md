# Six-hour T16 coding-performance outcome

The 2,048-token reasoning configuration completed a reviewed cube in 179.386 seconds
and tetrahedron in 225.366 seconds, both with normal agent exits and correct shaded
Braille rendering at three terminal sizes. A third octahedron attempt spent its
300-second budget on setup and planning without writing a renderer. These are
**two successful completions out of three fresh-seed trials**, not a reliable
success-rate estimate or proof that the reasoning cap caused the improvement.

The scheduled window is 2026-10-24 23:54:18 through 2026-10-25 05:54:18 UTC.
T16 v1 weights remain fixed. Candidate changes remain experimental, and the
accepted production configuration is restored after every hardware window.

## Measured results

| Change | Result | Practical interpretation |
|---|---|---|
| Bounded reasoning, no compact guidance | [Cube 179.4s; tetrahedron 225.4s; octahedron 300.2s timeout](budget2048.json), [octahedron receipt](budget2048-octahedron.json) | A useful pilot configuration with a real direction/completion failure; no default rollout |
| Persistent repair feedback | [Saved-history HTTP replay 20.489→3.539s](feedback-cache-api-replay.json), 82.729% less request latency | Fixes avoidable prefix loss; not an 82.7% whole-task speedup |
| Smaller GPU1 prefill share | [Warm full prompt 462.261→472.444 tokens/s](aux-prefill-w4-screen.json), +2.203% | Incremental 64/256-token results are nearly unchanged; no coding-task gain demonstrated |
| Paired gate/up launches | [Controlled oracle +1.835%](paired-screen.json) | Kernel screening gain, not isolated agent completion gain |
| GPU sampling | [Controlled oracle +2.231%](gpu-sampler-screen.json) | Saved-distribution checks and failure replays support correctness under their recorded conditions |

The two successful tasks decode at about 39.7 tokens/s and measure 608–638 new-prompt
prefill tokens/s over their particular request histories. These rates must not be
compared directly with a fixed full-prompt benchmark. Their first passing
artifacts appear at 144.179 and 184.195 seconds; the remaining time includes the
agent's display check and final summary.

The 512-token cap produces code but no passing cube. Native score 10 alone also
fails to produce a renderer in 300 seconds; that pilot is excluded from comparisons
because a stale proxy recorded its two requests in an older directory. The
recovered observations and concrete rendering defects are preserved. Numerical
replays cannot turn those failed tasks into successful completions.

## Bottlenecks and next experiments

The [incremental traces](incremental-prefill-profile.json) show substantial
main-GPU weight copies and T16 dequantization costs. GPU1 already computes resident
experts: its kernel busy fraction is about 59% for 64-token and 32% for 256-token
increments in these captures. Its 3.3 GB/s host link is not its VRAM read rate.
A fused prefill design or better weight-copy scheduling needs a new measured
wall-time and task-quality test. Previous direct/int8 and prefetch failures remain
explicit rejection criteria in the [decision ledger](DECISIONS.md).

A GPU drafter was not implemented. Current causal acceptance/cost evidence does
not justify its overhead; future-informed verification throughput is a ceiling,
not a demonstrated speculative-decoding speedup.

The optional completion-prefix mode preserves Pi system/tool headers while
blocking further tool execution. Unit tests pass, but the final octahedron never
reaches acceptance, so its live latency benefit remains unverified. It is disabled
by default. HTTP cache-hit usage fields also remain unimplemented; the engine's
raw prefix boundaries and measured prefill times are the authoritative counters.

## Reproduction and verification

The [gauntlet guide](../../geometry/README.md) gives the exact workflow and a
server preset. The measured host uses 14 CPU workers, a 125 W CPU long-term cap,
230 W GPU caps, a 118 GiB process guard and no swap. All requests use fresh recorded
seeds, temperature 1 and top-p 1. Captured sources and hashes identify each guard
version; source/display reviews and raw token audits accompany the outcomes.

Final verification is recorded in [the audit receipt](final-audit.json). It checks
the restored server executable/configuration, fixed model and rank identities,
original CPU power setting, idle health, owned-process/lock cleanup and repository
state. The heartbeat is paused at the deadline. All useful implementation changes,
negative results and receipts are committed and pushed on `codex/dsx-correctness`.

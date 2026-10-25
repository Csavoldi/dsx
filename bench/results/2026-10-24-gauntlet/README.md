# T16 geometry coding performance

The six-hour improvement window runs from 2026-10-24 23:54:18 UTC through 2026-10-25 05:54:18 UTC. These initial pilots use the original af2c154 serving binary, fixed T16 v1 expert weights, deferral rank 3, and the existing adaptive expert cache. No serving improvement has been adopted yet.

| Pilot | Agent time | Output tokens | Decode tokens/s | New-prompt prefill tokens/s | Reviewed outcome |
| --- | ---: | ---: | ---: | ---: | --- |
| Longer cube brief, reasoning off | 374.03 s | 11,223 | 38.41 | 373.54 | Failed: incorrect shading/overlap and frame pacing |
| Short cube brief, reasoning off | 361.92 s | 12,183 | 37.95 | 337.12 | Timed out; final artifact mechanically runs but lacks differentiated face shading |
| Short tetrahedron brief, reasoning low | 353.44 s | 12,123 | 37.54 | 446.27 | Completed, but face culling leaves missing visible regions at some rotations |

Rates divide summed engine token counters by summed engine durations. They include the actual variable coding trajectories. Shapes, prompts, and seeds differ, so these pilots do not establish a reasoning-mode winner or a kernel gain. The second and third runs use temperature=1.0, top_p=1.0, and fresh recorded request seeds. The low-reasoning pilot spent its initial 8,192-token response limit mostly on reasoning; it later produced and checked an artifact.

The initial mechanical judge accepted a running renderer too readily. Source and display review caught failures in the actual task requirements. Version 2 tightened frame pacing; version 3 followed the user's preference for a less restrictive brief and practical animation range. The [current version 4 prompt](../../geometry/PROMPT.md) explicitly requires filled, distinctly shaded braille faces, correct overlap, comfortable margins, and one successful self-check before stopping. It leaves the algorithm to the agent and removes the mesh-export requirement. Results across those contracts must remain separate.

`pilot-receipts.json` binds prompts, summaries, submitted code, reviews, SSE timing records, and server logs by SHA-256. Full raw receipts remain under `/home/lxe/dsx-gauntlet-20261024`. The interrupted longer tetrahedron run is excluded from completion comparisons.

The first implementation candidate combines gate/up preparation and GEMV into paired grid planes, removing two launches per T16 expert layer/store. It is opt-in through `DSX_T16_PAIR_GU=1`. Hardware output fingerprints match in 12 cases covering all three gate layouts and T=1/2/4. Timing is still in progress; this device check alone does not establish a useful speedup or completed coding quality.

A separate opt-in `DSX_LOOKUP_AUDIT=1` diagnostic observes causal copy proposals in serial coding output without changing sampling or executing verification. It will measure whether prompt-copy speculation has enough opportunity to justify implementation. Unit tests pass, but no live copy-acceptance result or speculative gain has been measured yet.

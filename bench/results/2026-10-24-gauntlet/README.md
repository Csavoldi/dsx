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

The first implementation candidate combines gate/up preparation and GEMV into paired grid planes, removing two launches per T16 expert layer/store. It is opt-in through `DSX_T16_PAIR_GU=1`. Hardware output fingerprints match in 12 cases covering all three gate layouts and T=1/2/4. The completed timing comparison is recorded below; functional gauntlet validation remains pending.

A separate opt-in `DSX_LOOKUP_AUDIT=1` diagnostic observes causal copy proposals in serial coding output without changing sampling or executing verification. It will measure whether prompt-copy speculation has enough opportunity to justify implementation. Unit tests pass, but no live copy-acceptance result or speculative gain has been measured yet.

## Paired gate/up screening

The completed ABBA comparison measured **44.0167 → 44.8244 tokens/s (+1.84%)** across 1,024 forced tokens per path. Every arm used the same compiled binary, 1,239 GPU0 and 2,757 GPU1 resident experts, 14 CPU threads at a temporary 125 W long limit, and 230 W GPU caps. Each process warmed placement with the exact future continuation and then froze it. This isolates a useful kernel candidate under an optimistic cache; it is not measured coding-task acceleration.

Three arms shared the same 4,821 CPU picks per round and full-model logit fingerprint. The first paired arm finished warm-up with 4,867 CPU picks and a different fingerprint, even though expert counts matched; its result remains included in the aggregate. The paired device tests independently matched all 12 layout/token-count cases. [Screening receipts](paired-screen.json) include every arm, hashes, routing counts, hardware maximum, and restoration events. The original serving binary, unchanged model identity, and original CPU power limit were independently verified after the window; the owned GPU lock was released.

## Live serving and completion controls

The initial paired-serving cube trial timed out; its completed requests measured 39.16 decode tokens/s, but the artifact failed rendering checks. The following octahedron trial was interrupted when the controller observed 92°C, above its 91°C threshold. Stock serving, model identity, and original power were restored and the lock released. That entire octahedron trial is excluded from comparisons because serving changed during it. Subsequent windows cap long CPU power at 125 W for both arms and stop the local trial immediately on a serving-window abort.

Live profiling measured CPU sampling around 0.6–0.7 ms/output token, adaptive-cache host work around 0.26 ms/token, and snapshots around 0.3 ms/request. Those figures identify sampling as a more useful next target than snapshots. The copy audit took about 8 µs/token in the first paired cube run and matched a mean 2.49 of three proposed future tokens across 880 completed blocks. That run contained repeated code repairs, so the result must not be presented as acceptance on an efficient successful task or as a speculative speedup.

The optional acceptance-stop workflow certifies shading and frame clearing, waits for the agent's successful capture of the unchanged renderer, and then disables tools and asks for a final summary. A first reviewed cube completion took **214.49 seconds**, 10 tool calls and 6,679 generated tokens, with normal exit and zero unfinished requests. It passed all three terminal sizes and independent source/display review. It used unpaired kernels, temperature 1.0/top-p 1.0, a fresh request seed, and a 125 W CPU cap. This is one successful workflow result, not a controlled percentage improvement over the differing earlier failed seeds. The following octahedron trial timed out at 301.18 seconds without differentiated face shading, so this first bounded batch has one successful task out of two. The controller preserved that failure instead of issuing an acceptance stop.

## GPU categorical sampling screening

`DSX_GPU_SAMPLE=1` samples eligible temperature-1/top-p-1 requests on the main GPU and returns a token instead of downloading the entire logit vector. Three kernels replace the measured CPU distribution construction and sampling cost. Scratch is allocated after expert placement; other sampling settings and non-finite logits use the existing CPU path, preserving its RNG position on fallback. The flag remains off by default. GPU weights use FP32 exponentials and double sums, while CPU weights use FP64 exponentials, so rare CDF rounding differences remain possible.

On the reference hardware, six alternating CPU/GPU pairs consumed the same 128 forced tokens per pair, with a fresh seed per pair and matching seeds within each pair. Aggregate throughput was **44.5468 → 45.5407 tokens/s (+2.23%)**. Every arm had the same routing counts, all 768 sampled token draws matched, and there were no fallbacks. Device support, non-finite fallback, frequency and full-vocabulary tests passed; all 12 existing T16 layout/token-count fingerprints remained unchanged. The isolated sampling call measured about 0.69 ms on CPU and 0.078 ms on GPU.

These measurements use paired T16 launches and placement warmed with the exact future continuation, then frozen. They establish a screening result rather than a coding-task speedup or universal sample equivalence. Peak CPU temperature was 72°C under the temporary 125 W cap. Stock binary, idle health, original power and released lock were independently checked afterward. [Sampling receipts](gpu-sampler-screen.json) bind the raw logs and every timing round. Functional gauntlet validation is in progress.

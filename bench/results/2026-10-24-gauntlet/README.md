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

The first live task with paired T16 launches and GPU sampling passed source/display review and finished normally in **109.23 seconds**, with 5 tool calls, 3,259 generated tokens, **40.94 decode tokens/s** and **641.52 new-prompt prefill tokens/s**. The cube used per-dot depth and different face densities, passed three terminal sizes without trails, and the agent captured its display before the acceptance stop. There were no GPU-sampling fallbacks. Signal handling restored the cursor as requested; its early exit omitted restoring autowrap, which is recorded as a cleanup limitation. [Cube receipt](sampler-cube-receipt.json) binds the source, review and timing logs. Its seed and generated solution differ from the earlier 214.49-second cube, so the task-time difference is not a controlled kernel improvement. GPU sampling is timed inside `step_s`; the small remaining `sample_s` counter does not mean device sampling is free. The same-binary CPU-sampling comparison remains pending.


The following GPU-sampling tetrahedron trial timed out after **302.00 seconds** and failed review, leaving this first candidate batch at one success out of two. It generated 10,769 tokens at 39.94 decode tokens/s and made 17 tool calls. About 269.63 seconds went to decode and 31.33 seconds to prefill. Early revisions raised Python type errors, and later repairs left incorrect triangle-orientation arithmetic and a shade-mask membership test that suppressed visible faces. The final submitted renderer produced zero dots at four sampled 80×24 rotations. In-memory diagnostic substitutions produced visible geometry, but the submitted artifact was preserved and remains a failure. [Failure receipts](sampler-tetrahedron-receipt.json) record the exact substitutions, dot counts, hashes and attribution limits. These independent sampled tasks do not show that GPU sampling caused the coding failure. Stock serving, model identity, original power and released lock were verified after the batch.

The runner now writes `progress.txt` while a task runs, distinguishing streamed code/tool arguments, waiting for a first model delta, timeouts and completion. This observer reads existing logs without sending inference requests. The terminal display remains the agent's animation rather than a generation indicator.

The paired-launch CPU-sampling control cube also timed out (300.83 seconds, 38.73 decode tokens/s); its smallest-size shading check failed. The next tetrahedron was interrupted at the user's request to reserve the server and is excluded from completed-task counts. The accepted stock server was restored and independently checked before the manual reservation. [The quality investigation](quality-investigation.md) preserves the negative completion outcomes, distinguishes measured divergence from missing evidence, and defines failure-conditioned replays. No experimental serving configuration has been adopted.

## Stable adaptive ordering and deferral controls

The failure-conditioned investigation found equal-score adaptive evictions that
changed CPU deferral participation even between two runs of the baseline.
Explicit layer/expert tie-breaking removed that observed drift: original-image,
paired/GPU-sampling and repeated controls matched all 16,677,120 F32 logits and
the complete cache/routing traces on the tested prefix. This is a reproducibility
fix, with no claim of improved coding quality. See the [quality investigation](quality-investigation.md).

The first live batch with stable ordering and **deferral disabled** completed a
reviewed cube in **167.61 seconds**, 4,724 tokens and 10 tool calls, at **34.68
decode tokens/s** and **440.97 engine-cache-miss prefill tokens/s**. Its original
judge-5 result was a false failure: returning the cursor home at both ends of a
frame created empty intervals that the parser counted as redraws. Judge 7
excludes control-only intervals and passes the unchanged renderer at all three
sizes. A [retrospective check](judge-v7-retrospective.json) applied the same fix
to seven earlier artifacts, retaining their previous artifact outcomes.

The no-deferral tetrahedron timed out after **300.09 seconds**, 8,930 emitted
tokens and 13 tool calls, at **33.49 decode tokens/s** and **407.06 prefill
tokens/s**. Its final faces used ASCII characters instead of braille, and it
fails the corrected judge too. This batch therefore has one reviewed completion
in two tasks. These fresh sampled tasks do not prove that deferral caused or
prevented a quality change. [Batch receipts](stable-defer-none.json) include
source/display review, original and corrected checks, exact raw-token audit,
temperatures and verified stock restoration.

The token audit matched every recorded prompt length and available completion
usage count. Processed output counts matched model-step counters. The cancelled
last tetrahedron response emitted one token that was not processed into KV;
that explains its one-token difference from the engine-generated total. The
830-token prefix replay occurred on the first tetrahedron request, where only a
shared prefix of the previous cube conversation matched; no follow-up within
either task replayed an already cached prefix. The rank-3 task batch is the next
control, keeping the same candidate binary and task settings with fresh seeds.

The stable **rank-3** batch then timed out on both tasks: cube **301.52 seconds**
at **39.98 decode / 415.00 prefill tokens/s**, and tetrahedron **300.81 seconds**
at **39.91 decode / 369.12 prefill tokens/s**. Both fail the corrected judge and
source/display review. The cube clips the terminal boundary and associates
several faces with incorrect normals. The tetrahedron's culling uses inconsistent
face winding; its unchanged renderer is blank at several sampled rotations.
[Rank-3 receipts](stable-defer-3.json) preserve these failures, raw-token checks
and verified stock restoration. These two groups use different fresh seeds and
solutions: one completion in two no-deferral tasks versus zero in two rank-3
tasks is insufficient to attribute a quality difference to deferral. Faster
output is not a successful task completion, and no candidate serving defaults
have been adopted.

## Causal copy proposals on actual generated IDs

[Offline replay receipts](causal-lookup-replay.json) evaluate widths 4/8 and suffix
lengths 8/12/16/24/32 on all four stable-batch trajectories. Replaying width 4,
suffix 8 reproduces every live attempt, completed block, matched-prefix histogram
and token count exactly. Each proposal uses only tokens already present in the
prompt and processed output; an unfinished proposal is not counted as accepted.

A conditional cost model combines these observed matching lengths with the
[historical warm, future-informed verifier costs](../2026-10-24-102/receipts.json).
It assumes zero drafting, communication, rollback and displaced-expert costs,
and excludes prefill and tools. On the successful no-deferral cube, the best
screened configuration predicts just **0.35% decode-trace gain** (width 4,
suffix 16); suffix 8 predicts a small loss. Failed, repetitive repair traces
predict up to 2.17–4.64%, which does not improve their failed completion status.
These are conditional estimates, not measured speculative acceptance, delivered
speedups, or upper bounds for another drafter, cache policy or verifier. They do
not justify implementing GPU speculation ahead of the prefill and workflow
experiments. `bench/geometry/analyze-lookup.py` reproduces this CPU-only analysis.

## GPU1 streaming during prefill

An off/on/on/off comparison tested `DSX_AUX_STREAM=1` on the x4 GPU while
retaining its fixed resident experts. All four arms used the same binary,
complete expert membership (1,240 main / 2,750 auxiliary), fixed adaptation,
T16 weights, 14 CPU workers at 125 W, and the accepted GPU power caps. Each arm
ran three rounds of a full 2,031-token prompt and 256-/64-token increments,
followed by the same 64 forced decode tokens. Prefix construction was excluded
from incremental timing. The first round and warm rounds are reported separately
in the [complete receipt](aux-prefill-screen.json).

| Warm path | Streaming off, tok/s | Streaming on, tok/s | Change |
| --- | ---: | ---: | ---: |
| Full 2,031-token prefill | 461.85 | 423.86 | −8.23% |
| 256-token increment | 159.15 | 159.63 | +0.30% |
| 64-token increment | 108.71 | 105.12 | −3.30% |
| Forced decode after full prompt | 37.37 | 37.35 | −0.05% |
| Forced decode after 256 increment | 37.69 | 37.37 | −0.85% |
| Forced decode after 64 increment | 37.53 | 37.60 | +0.19% |

This rejects the tested streaming split on this host. GPU1 continued computing
its resident experts in both arms; the flag adds a share of experts copied from
RAM for prefill. An idle fraction during decode does not establish spare
capacity during prefill. The x4 link's theoretical transfer contribution did not
translate into a gain. A bounded profile will distinguish copy, compute and
scheduling costs before trying another split. No auxiliary adaptation or SSD
expert storage was enabled. Fixed-cache/forced-token rates are not live agent
performance, and changed prefill summation order also changed fingerprints and
some subsequent routing counts. Stock binary, model metadata, original power,
idle health and released lock were verified afterward.

## Empty-cell capture normalization

A CPU-only check of a saved successful cube follow-up replaced 3,174 U+2800
empty Braille cells with ASCII spaces in its single captured-display tool result.
All rows, columns, nonblank dots, other messages and request settings remained
unchanged. The embedded GGUF tokenizer and serving template encoded **7,365 →
5,855 prompt tokens**, saving **1,510 tokens (20.50%)**. [The receipt](capture-normalization-screen.json)
binds both requests and token-ID files. This is a prompt-size measurement;
prefill latency and completion-quality gains remain to be tested in the labeled
live workflow. The raw capture is preserved.

## Corrected-judge workflow control

The next three-shape batch used judge 7 during the task, the same version-4
prompt, a 180-second time limit, stable adaptive ordering, paired launches,
GPU sampling and rank-3 deferral. Capture normalization remained off. It completed
**zero of three tasks**: cube 181.33 seconds / 40.93 decode tokens/s, tetrahedron
180.63 seconds / 39.90, and octahedron 181.51 seconds / 40.00. These shorter-budget
outcomes should be compared with the following matching workflow groups, rather
than combined with the earlier 300-second/judge-5 success rates.

The failures are concrete. The cube initializes depth to positive infinity and
only accepts greater inverse depths, leaving every frame blank. A diagnostic
substitution restores dots, but its bit packer still ignores shading. The
submitted octahedron similarly computes shading without applying it to visible
density or color, leaving a uniformly filled silhouette.

The tetrahedron shows shaded geometry, then loses output during the resize
sequence and exits without restoring the cursor. It makes stdin nonblocking
while stdin and stdout share the PTY's open-file description. A diagnostic copy
using select-based input readiness passes all three sizes and exits cleanly;
the unmodified repeat fails again. Both diagnostic and original source/output
hashes are preserved in the [batch receipt](workflow-control.json). Diagnostic
repairs do not change the submitted artifacts or their failed completion status.
These tests explain code-level failures without attributing them to a particular
inference approximation. A matching no-deferral batch follows to gather more
actual completion evidence before deciding whether deferral's speed is useful.

## Prefill attribution and the second deferral control

A bounded [Nsight capture](aux-prefill-profile.json) attributes the rejected
streaming configuration. GPU0 host-to-device traffic falls from 68.78 to 62.95
GB, but GPU1 peer-copy traffic rises from 0.68 to 6.34 GB. GPU0's combined kernel
and copy busy fraction falls from 96.0% to 81.2% of the capture span. GPU1's
kernel busy fraction is only 20.6% without streaming and 22.2% with it; its extra
traffic, rather than a large increase in compute, consumes the x4 path. The
capture span grows from 4.44 to 4.85 seconds. These are instrumented attribution
measurements at matched placement, not replacements for the untraced ABBA rates
or measurements of SM occupancy. A smaller auxiliary work share would need its
own measurement before use.

The matching no-deferral, judge-7, 180-second task batch also completed **zero
of three tasks**, at 34.79 / 34.65 / 34.71 decode tokens/s for cube, tetrahedron
and octahedron. The cube exposes another checker limitation: synchronized-update
controls were unsupported in version 7. Version 8 certifies the unchanged
artifact, though source/display review records its horizontally displaced,
compressed layout. The tetrahedron is too small and loses shading at the smallest
size; the octahedron converts every non-space ASCII shade into an enabled Braille
dot, collapsing its face densities. [Batch receipts](workflow-nodefer.json)
preserve these distinctions and every original timeout. Neither matching batch
establishes a completion advantage from removing deferral.

The final octahedron request was unfinished when the run stopped, so its raw
engine token receipt and final usage are unavailable. Counters cover the ten
completed requests and must not be presented as totals for all eleven starts.
The saved requests also show that a functional-feedback user note appeared in
request 10 and disappeared in request 11. The latter queued 16,749 prompt tokens
and never produced a first delta before the deadline. That changing prefix can
force a full prefill because the engine retains only its latest prompt snapshot.
Version 8 now attaches feedback to stored tool results to preserve history;
actual request/token verification of the new path is the next validation step.

## Persistent feedback and bounded reasoning trials

The [exact prefix analysis and live feedback receipts](feedback-prefix-repair.json)
confirm the transient-note problem. Dropping the old user note moves the shared
prefix before the retained snapshot: the16,749-token request must prefill from0.
Keeping that note would require only636new tokens. This comparison uses exact
model tokenization and the cache restore rule; it is not a timed replay. Judge8
now stores functional feedback in tool results. The live trial retains every
observed note in subsequent requests. It still records cross-task prefix replay
and24replayed tokens on the final octahedron request, so persistence does not
imply that every request has zero replay.

The [blank-cell normalization batch](workflow-space.json) and
[compact-guidance/judge8 batch](workflow-v8-compact.json) both complete zero of
three tasks within their180-second budgets. Their source and saved-output reviews
identify concrete defects: lost shading, clipped or tiny shapes, invalid face
connectivity, and a dot-grid indexing exception. The second batch combines
several workflow changes and cannot isolate their individual effect. Decode
rates remain about40–41tokens/s. No task-completion improvement is booked.

A missing tmux pane exposed a cleanup bug after the first batch: the owned proxy
was left listening, and a subsequent native-score10 pilot sent its requests
through that listener. New runs verify proxy root/PID/nonce and always terminate
their owned proxy even if pane cleanup fails. Run summaries exclude requests
started outside the agent lifetime while retaining late endings of owned
requests. The two misplaced requests are preserved and recovered in the
[native-score10 receipt](native-score10-recovered.json), with exact prompt-token
matching to the candidate engine audit. That pilot is excluded from comparisons;
its300-second timeout and11,615processed tokens without a renderer remain useful
negative observations about score10 alone.

`DSX_THINKING_BUDGET` adds an optional hard limit on thinking tokens. Forcing the
thinking-close token intentionally changes generation and needs task-quality
validation; it is disabled by default. Forced positions and original samples are
recorded in the token audit. A shorter thought stream by itself does not count as
a successful coding task.

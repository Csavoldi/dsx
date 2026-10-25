# Coding quality and distribution divergence

The failed tetrahedron is evidence of poor task completion in the tested configuration. Its immediate defects were incorrect triangle-orientation arithmetic, shade-mask selection and ineffective debugging. Those defects do not establish why the model generated them. A regression in inference behavior has not been ruled out, and the current candidate has not been adopted for normal serving.

## Measured task outcomes

All rows use fixed T16 v1 weights, rank-3 deferral, adaptive cache 8, reasoning off, temperature/top-p 1, fresh request seeds, the version-4 task prompt and version-5 acceptance checks. The CPU long limit was 125 W. Shapes and generated trajectories differ; these small groups cannot establish equal quality or a statistically supported regression.

| Serving configuration | Shape | Task seconds | Reviewed completion |
| --- | --- | ---: | --- |
| Unpaired launches, CPU sampling | Cube | 214.49 | Pass |
| Unpaired launches, CPU sampling | Octahedron | 301.18 | Timeout; failed shading checks |
| Paired launches, GPU sampling | Cube | 109.23 | Pass |
| Paired launches, GPU sampling | Tetrahedron | 302.00 | Timeout; blank frames and code defects |
| Paired launches, CPU sampling, same candidate binary | Cube | 300.83 | Timeout; failed shading check at the smallest size |

The candidate groups have one reviewed success in three completed trials; the unpaired group has one in two. This is a negative reliability signal that must be retained, with substantial sampling and task differences. The following CPU-sampling tetrahedron was interrupted when the user reserved the server; it is excluded from completed-task counts. User-requested interruption is not another model failure.

## What the numerical checks establish

Earlier validation compared deployed af2c154 with f1f53e4 on 1,023 teacher-forced positions under fixed placement and found zero changed F32 values out of 132,253,440. That clears those default builds on that sample, not the new experimental flags or live adaptive behavior. The [earlier report](../2026-10-24-102/README.md) records this evidence.

Paired T16 kernels matched 12 device cases. Three of the four timing arms shared the same full-model logit fingerprint and routing counts. One paired warm-up ended with different CPU picks and a different fingerprint, despite equal expert counts. Cache state therefore needs to be controlled rather than inferred from resident counts.

GPU sampling matched all 768 sampled token choices on the tested forced model outputs, and all 512 synthetic full-vocabulary draws. Matching samples does not prove unchanged distributions: the GPU uses FP32 exponentials while the CPU uses FP64. The original failed trajectory's warmed expert-cache state has not been reconstructed. The controlled fixed/adaptive failure-prefix replays below measure the new execution paths on that history without claiming an exact replay of its original cache state.

Rank-3 expert deferral changes the forward pass: CPU experts ranked 3 or lower enter the following layer's residual, using their original layer's output weights. An earlier held-out sequence measured teacher top-256 KL 0.137649 without deferral and 0.160876 with rank 3, a difference of 0.023227. That is an existing approximation, not a measured new regression. Adaptive placement can change which experts run on CPU and therefore which contributions are deferred. Faster kernels or synchronization can interact with this policy even when an isolated kernel is equivalent.

## Prepared failure replays

Three exact saved requests have been encoded with the embedded T16 tokenizer and serving template: the initial task (2,106 tokens), the first TypeError repair (10,433), and an explicit blank-frame observation (17,859). Every encoded length matches its recorded serving prompt count. [The manifest](quality-replay-manifest.json) binds the request and token-ID files. Preparation used CPU tokenization only; it did not issue inference requests.

The user subsequently released the manual-server reservation and authorized server experiments. The planned comparison was to compare accepted defaults, candidate with both flags off, paired-only, and paired-plus-GPU-sampling on these same prefixes. Generate each continuation once with the accepted baseline and force those IDs in every arm. Measure full-vocabulary KL, top-1 changes, logit fingerprints, CPU/deferred-expert participation, actual cache placement and sampled CDF decisions. First freeze placement to isolate arithmetic; then test adaptation 8 to capture the real cache/deferral interaction. Compare deferral off versus rank 3 separately with T16 fixed. Preserve exact model, compiler, binary and placement receipts.

Functional completion remains the adoption criterion. Distribution diagnostics should identify a cause for a failed task or rule out a specific execution change; they cannot certify coding quality on their own. Further live task trials must use fresh seeds, the same task contract and bounded self-check/stop behavior. The temporary manual reservation has ended; the diagnostic windows below restored and verified stock serving afterward.

## Fixed-placement failure replay

The first failure-conditioned replay completed on the 10,433-token request immediately after the model received its TypeError traceback. It includes the initial prefill distribution and 128 forced continuation positions, totaling 16,677,120 F32 values per arm. Under identical fixed store membership, the archived CUDA image, current defaults, paired launches and GPU-sampling path produced bit-identical logit arrays. All 129 sampled choices matched, and an independent repeated baseline matched too. Actual immediate/deferred CPU expert partitions matched captured membership in all 5,120 layer records per arm. [The receipt](fixed-failure-replay.json) includes every comparison, source/binary provenance, raw hashes and restoration verification.

A separate audit read the GPU sampler's actual weights on those saved distributions. Mean CPU-to-GPU KL was 2.10e-15; maximum total variation was 5.00e-9, with zero changed draws in the tested 129 paired draws. The largest continuous-uniform CDF interval disagreement estimate was 3.70e-6; this measures sensitivity of equal-seed inverse-CDF choices, rather than a task-quality loss. Double accumulation and finite RNG precision introduce further boundary details. These measurements narrow the isolated arithmetic and sampling hypotheses on this sample without certifying functional quality.

This fixed cache is colder than the failed live agent's cache and uses explicit 10.5/23.3 GB expert budgets (1,241/2,750 resident experts). Its CPU participation was 48.5% of routed picks, and the mean fraction of routing weight delayed by deferral was 19.7%, with a 42.5% 95th percentile across layers. Those are controlled-replay values, not measured fractions in the original failed run. The adaptive-cache comparison below addresses cold-prefix replay; reconstructing a prefix's KV state alone does not reconstruct the expert cache warmed by previous generated responses.

## Adaptive-cache replay and equal-score evictions

With adaptation 8, the repeated archived-image baseline differed from its first run: mean full-vocabulary KL 0.000100407, worst-position KL 0.00707080, and 15,384,316 changed F32 values. Top-1 and all 129 tested sampled choices stayed unchanged. Current defaults, paired-only and GPU-sampling arms matched the repeated baseline bit for bit. Thus this sample reproduces numerical drift within the baseline itself, rather than showing an additional arithmetic regression from the optional kernels or sampler. [Adaptive receipts](adaptive-failure-replay.json) preserve every contrast, actual CPU partitions, cache changes and restoration evidence.

The first cache difference appeared after the second forced token. The runs promoted the same experts but evicted different residents: (6,358) versus (17,335). Both had calibration count 4,375 and neither had been selected. Before the first numerical difference, another equal-score choice evicted (2,364) in one run and (0,185) in the other; both had calibration count 4,401 and neither had been selected. At forced-token index 9, layer 0 selected the same expert IDs and weights in both runs, but expert 185 was GPU-resident in one and CPU-deferred in the other. The first differing logit row was 10.

The adaptive eviction code reads residents from a hash map and sorts them by score alone. Equal scores therefore have no stable eviction order. Rank-3 deferral converts a residency difference into a different forward pass. This explains a concrete source of seeded numerical non-reproducibility, while leaving the cause of the original coding failure unproven. `DSX_ADAPT_STABLE=1` adds explicit layer/expert tie-breaking without changing score priorities; it is opt-in and the repeated-run validation below tests that intervention. Real task quality and throughput still need to be checked before adoption.

## Stable eviction validation

With `DSX_ADAPT_STABLE=1`, the archived-image control, paired/GPU-sampling candidate and repeated control matched on all 16,677,120 logit values and all 129 sampled choices. Their full cache-change and CPU-routing traces also matched, with zero partition mismatches. The test reused the previous adaptive run's continuation, expert budgets and prefix. [Stable-control receipts](stable-failure-replay.json) bind the outputs and verified stock restoration. This intervention removed the observed tie-order divergence on the tested prefix; it does not establish better coding quality or universal timing independence.

The completed task trials compare rank-3 deferral with deferral disabled, retaining fixed T16 weights and fresh seeds. Both 180-second/judge7 groups complete zero of three tasks; the no-deferral cube also exposes a checker limitation, documented in workflow-nodefer.json. Raw-token/cache-boundary receipts preserve actual sampled IDs. Later persistent-feedback trials and the excluded native-score10 recovery are described in this directory's README. These outcomes do not establish a quality advantage from disabling deferral.

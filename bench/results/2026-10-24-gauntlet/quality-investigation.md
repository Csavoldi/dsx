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

GPU sampling matched all 768 sampled token choices on the tested forced model outputs, and all 512 synthetic full-vocabulary draws. Matching samples does not prove unchanged distributions: the GPU uses FP32 exponentials while the CPU uses FP64. Full-vocabulary KL has **not** been measured for the failed coding trajectory or the new sampler path under live adaptation.

Rank-3 expert deferral changes the forward pass: CPU experts ranked 3 or lower enter the following layer's residual, using their original layer's output weights. An earlier held-out sequence measured teacher top-256 KL 0.137649 without deferral and 0.160876 with rank 3, a difference of 0.023227. That is an existing approximation, not a measured new regression. Adaptive placement can change which experts run on CPU and therefore which contributions are deferred. Faster kernels or synchronization can interact with this policy even when an isolated kernel is equivalent.

## Prepared failure replays

Three exact saved requests have been encoded with the embedded T16 tokenizer and serving template: the initial task (2,106 tokens), the first TypeError repair (10,433), and an explicit blank-frame observation (17,859). Every encoded length matches its recorded serving prompt count. [The manifest](quality-replay-manifest.json) binds the request and token-ID files. Preparation used CPU tokenization only; it did not issue inference requests.

After the user releases the manual-server reservation, compare accepted defaults, candidate with both flags off, paired-only, and paired-plus-GPU-sampling on these same prefixes. Generate each continuation once with the accepted baseline and force those IDs in every arm. Measure full-vocabulary KL, top-1 changes, logit fingerprints, CPU/deferred-expert participation, actual cache placement and sampled CDF decisions. First freeze placement to isolate arithmetic; then test adaptation 8 to capture the real cache/deferral interaction. Compare deferral off versus rank 3 separately with T16 fixed. Preserve exact model, compiler, binary and placement receipts.

Functional completion remains the adoption criterion. Distribution diagnostics should identify a cause for a failed task or rule out a specific execution change; they cannot certify coding quality on their own. Further live task trials must use fresh seeds, the same task contract and bounded self-check/stop behavior. No GPU benchmark, automated inference request or serving restart is authorized while the manual reservation remains in effect.

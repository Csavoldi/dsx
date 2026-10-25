# Agentic coding on one MI325X

`serve-agentic-mi325x.sh` is an opt-in serving profile for DeepSeek-V4.1-Flash-1017 on one MI325X 256 GB. Four concurrent coding sessions are the primary workload; the server admits up to eight. It expects the existing target checkpoint, native MXFP4 export, expert ranking, and the original checkpoint's MTP shards. It binds both API and admin service to loopback.

## Profile

Run `bash amd/box/serve-agentic-mi325x.sh` from a release build on the GPU host. Override `DSX_MODEL_DIR`, `DSXA_WEIGHTS`, and `DSX_RANK` for another checkpoint location. `DSXA_BIN`, `DSX_PORT`, `DSX_ADMIN_PORT`, and `DSX_TEMPERATURE` are optional overrides. This profile was measured on gfx942 with ROCm 7.14; other hardware needs its own measurements.

The profile reserves 8.001241088 GB for the native four-bit MTP draft and 15.998758912 GB for the remaining serving buffers, keeping the original combined 24 GB target-placement budget. The measured placement was 3,378 MXFP4 and 11,982 CB3 experts, with no CB2 experts. Only attention wq_b, wo_a, and wo_b receive BF16 dense copies, totaling about 9.4 GB. These copies retain the larger-row BLAS path without duplicating every dense matrix.

MTP proposals use the checkpoint's trained five-token block. Every emitted proposal is verified by the target. At positive temperature with top_p=1, proposals are sampled from q, accepted with min(1,p/q), and rejected proposals recover from normalized positive(p-q). The GPU sampling kernels preserve the CPU sampler's 30-temperature-unit logit cutoff. Truncated distributions use the existing CPU path. This is probabilistic speculative sampling; acceptance is measured, never forced.

`DSXA_VERIFY_ROWS=31` limits total target verification rows. With five desired proposals it permits lengths 5/5/5/5/5/4/3/2 at one through eight decoding sessions. This avoids a measured performance drop when this engine's verification batch crosses 32 rows. The exact cause of that boundary is unresolved. The candidate enables up to six prompt-lookup proposals with a five-token matching context. All lookup proposals remain target-verified; the row budget also caps lookup length. Its benefit depends on repeated editing content. Set DSXA_LOOKUP_LIMIT=0 to disable it.

New prompts use 512-token chunks while existing sessions decode. Four decode rounds run between chunks. With no decoders, prefill uses 2,048-token chunks. This policy favors ongoing output at a modest cost to a newly arriving long prompt. HIP batch graph replay stays disabled because it regressed these speculative workloads.

## Measurements and limits

A cold 33,271-token arrival while three sessions decoded took 24.34 s to first token with this cadence, compared with 22.18 s at 256-token chunks and no extra decode rounds. Background output during prefill increased from 70.12 to 146.80 aggregate TPS; maximum observed stream gap increased from 0.18 to 0.25 s. This is a mixed-arrival measurement, not a four-session steady decode rate.

Direct decode rates use output-token counter differences only while all requested sessions are decoding and no prefill is active. End-to-end wall throughput includes prompt processing and is reported separately. Dashboard moving averages can retain prefill history and must not be substituted for the direct counter window. Replaying fixed shape-code requests screens throughput; it does not execute generated tools and cannot establish task completion quality.

## Live tuning and verification

The script enables an optional `/tuning` route on the loopback admin port. POST controls are validated before any mutation. Controls include draft length and enablement, verification-row budget, lookup length/context, prefill chunk sizes/cadence, GPU sampling, probabilistic draft rejection, attention output fusion, and graph replay. Changing controls invalidates cached graphs. These controls apply to active work and should be changed between benchmark rounds for reproducible comparisons. Do not expose the admin port publicly.

Retained kernel changes include HC premix and scratch reuse, the Engram history-tail copy, attention merge/output preparation, finer attention splits, and paired routed/shared SwiGLU preparation. Oracle tests cover the fused kernels and GPU sampling/rejection; scheduler, lookup, and row-budget tests run on the AMD host. Task completion remains the primary quality decision. Performance estimates from memory bandwidth are conditional lower bounds, not proof of a model-wide theoretical maximum.

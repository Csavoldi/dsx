# T16 prefill experiments

These experiments target 1,000 newly processed prompt tokens per second on the dual RTX 3090 host. T16 model bytes, rank-3 decode deferral and the 256K context allocation remain fixed. All new kernel and buffer options are disabled by default.

The saved 3K/8K prompts are the same fixtures used before these changes. Each full-prompt measurement resets model state and processes every input token; warm measurements exclude CUDA's first-use loading cost. Incremental measurements count only newly processed tokens. Larger chunks change buffer allocation and can displace resident experts, so their decode cost must be reported separately.

## Implemented options

- `DSX_T16_WARP_HAD=1` replaces block-wide Hadamard barriers with warp shuffles and registers in the same butterfly order.
- `DSX_T16_SPECIALIZED_DQ=1` specializes weight expansion for the three T16 bit widths and two layouts. The initial screen shows little benefit beyond the Hadamard change.
- `DSX_PREFILL_REUSE_BUFFERS=1` shares query, attention output and temporary storage at separate stages, saving about 1.91 GB at an 8K allocation. Query expansion follows indexer scratch use. Tiny Engram GEMVs retain separate source storage.
- `DSX_T16_FUSED_POST_GU=1` combines gate/up Hadamard post-processing and SwiGLU, removing two launches per routed expert. Full-model timing is in progress.

## Verification

The CUDA audit compares 84 cases with zero changed values: all T16 weight formats/layouts in both projection dimensions; Hadamard transforms including in-place input; three attention implementations with aliased query/output; and fused gate/up post-processing at multiple shapes, gains and clipping limits. Its raw output is [kernel-audit.jsonl](kernel-audit.jsonl).

Run the audit with `cargo run -p dsx --release --features cuda --bin t16-prefill-audit` on an available CUDA GPU. These are isolated kernel checks, not a coding-task success claim.

At fixed expert placement, buffer reuse also matches every compared full-model logit for the full 3K prompt and 8/1-token increments: 387,840 unchanged values. Full-model performance and server validation receipts will be added after the remaining measurements finish.

The benchmark binary supports `--prefill-probe --prefill-full-only` for full prompts and `--prefill-probe --prefill-new-tokens 3000,8,1` for selected increments. Use `DSX_ADAPT=0`, a saved continuation and `--dump-reference` / `--compare-reference` to hold placement and compare final logits. Record complete expert membership and forced-continuation fingerprints as well as counts.

# T16 prefill checkpoint

The integrated runner processes an 8,000-token prompt at **1,007.84 tokens/s** across two warm passes on two RTX 3090s, an i7-12700K and 128 GB DDR4. T16 weights and F16 Engram projections remain unchanged. The context allocation is 262,144 tokens and the prefill chunk is 8,192 tokens. Benchmark screens cap the CPU package at 125 W and use 14 CPU threads. GPU caps read back as 230 W per card after the checkpoint; the controllers do not change GPU caps.

Each timed pass resets model state and processes every prompt token. The initial pass measured 928.49 tok/s; the warm passes measured 1,008.50 and 1,007.18 tok/s. Warm-up excludes first-use kernel loading, not prompt processing. No prompt KV is reused. These numbers describe this saved workload; they do not imply 1,000 tok/s for smaller prompts or cached tool updates.

## Changes and measurements

| Configuration | Warm 8K prefill, tok/s | GPU0 / GPU1 resident experts |
|---|---:|---:|
| Original 4K chunk | 868.07 | 1,161 / 2,757 |
| 8K chunk | 981.32 | 829 / 2,705 |
| Add warp Hadamard and fixed-bit expansion | 993.05 | 829 / 2,705 |
| Add buffer reuse | 994.72 | 1,048 / 2,705 |
| Add fused gate/up post-processing | 1,006.43 | 1,048 / 2,705 |
| Main integration, same final profile | **1,007.84** | 1,048 / 2,705 |

The original and final profiles use the same prompt and 256K context allocation in separate screens. The combined improvement is about 16%. The final fusion also has a closing control: without that fusion, the bracketing controls measured 997.73 and 996.14 tok/s at identical expert placement. Its two warm 64-token forced continuations measured about 40.4 decode tok/s. These short continuation checks are separate from a complete agent task and from the older 41 tok/s free-generation result with FP8 projections.

Larger batches amortize repeated expert transfers and expansion. The larger allocation initially displaces resident experts. Buffer reuse recovers about 1.91 GB: query, attention output and temporary storage share a buffer at separate stages. Query expansion follows indexer scratch use, and tiny Engram GEMVs retain a separate source. Warp transforms remove block barriers; fused gate/up post-processing removes two launches per routed expert. Fixed-bit expansion alone adds little benefit beyond the warp transform in the 3K screen.

## Enable the profile

All four options are off by default. For the measured profile, prefix the normal server invocation with:

```bash
DSX_T16_WARP_HAD=1 \
DSX_T16_SPECIALIZED_DQ=1 \
DSX_PREFILL_REUSE_BUFFERS=1 \
DSX_T16_FUSED_POST_GU=1 \
./target/release/dsx-server \
  --model dsx-t16/DeepSeek-V4.1-Flash-1017-dsx-t16.gguf \
  --rank dsx-t16/expert-rank.json \
  --ctx 262144 --chunk 8192 --arena-gb 110 \
  --main-gb=-1.5 --aux-gb=-1.0 --defer-rank 3 \
  --host 0.0.0.0 --port 8090
```

These settings include rank-3 decode deferral, which remains an approximation. The new prefill operations preserve the arithmetic tested below. This profile leaves 8.9 GB of experts on GPU0 and 22.9 GB on GPU1 on the reference host. Other hardware and context allocations require new measurements.

## Verification and receipts

The [CUDA audit](kernel-audit.jsonl) contains 84 exact cases: all T16 bit widths/layouts in both projection dimensions; Hadamard transforms including in-place input; three attention implementations with aliased query/output; and fused gate/up post-processing across shapes, gains and clipping limits. Run it on an available GPU with:

```bash
cargo run --release -p dsx --features cuda --bin t16-prefill-audit
```

At fixed expert placement, buffer reuse matches 387,840 full-model logits for a 3K prompt and 8/1-token additions. Fused post-processing matches all 129,280 final logits and the 64-token forced-continuation fingerprints. The [main integration check](integration-check.json) also matches the experimental runner's final logits byte for byte. These checks establish agreement on the tested inputs; they do not replace FP8-reference quality evaluation or resolve the earlier open decode check in [runtime status](../../../docs/current-state.md).

[Main timing output](main-prefill.log), [screen summaries](screens.json) and [deployment verification](deployment.json) retain the measurements, commands, flags, binary hashes and model-file identity checks. Raw logs, full resident membership and F32 references are archived under `/home/lxe/ai/ds41/codex-prefill-20261025` on the measurement host and mirrored at `/home/lxe/dsx-prefill-20261025/remote`. Prompt fixtures are identified by hashes in the receipt metadata; they are retained outside the source repository with the agent-session artifacts.

The main runner includes a full-prompt benchmark. Stop the inference server before running it so the benchmark has the GPUs available:

```bash
DSX_ADAPT=0 DSX_CPU_THREADS=14 \
DSX_T16_WARP_HAD=1 DSX_T16_SPECIALIZED_DQ=1 \
DSX_PREFILL_REUSE_BUFFERS=1 DSX_T16_FUSED_POST_GU=1 \
cargo run --release -p dsx --features cuda --bin prefill-bench -- \
  --model dsx-t16/DeepSeek-V4.1-Flash-1017-dsx-t16.gguf \
  --rank dsx-t16/expert-rank.json --ids prompt8000.ids \
  --ctx 262144 --chunk 8192 --rounds 3
```

The fixed-placement full/incremental oracle screens additionally use the `oracle-bench` harness on the `codex/prefill-1000` experiment branch. The new main benchmark reports every round, zero cached tokens and the weighted warm rate.

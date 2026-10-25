# DSX Nanbeige Rust and CUDA runner

This branch runs the dense Nanbeige4.3-33B model in DSX's Rust and CUDA runtime on an RTX 5090. It supports native NVFP4 weights and standard EXL3 `mul1` trellis weights, an OpenAI-compatible API, the embedded DSX web UI, and Pi tool calls. Inference does not launch vLLM, PyTorch, or llama.cpp. The EXL3 conversion and independent reference evaluations use ExLlamaV3 and PyTorch separately.

The fork started at DSX commit `53819a1781527e9e4f2c09ded29ba666dc9fc84f`. It retains DSX's GPU boundary, safetensors reader, sampler, FIFO engine worker, cancellation handling, and web UI. Nanbeige's dense Llama execution is separate from DeepSeek's MoE, MLA, Engram, and expert streaming paths.

## Weights

The NVFP4 path uses `primitive-ai/Nanbeige4.3-33B-NVFP4` revision `4171a7d6eabfffc37b9d89629fb75d28fd5c2c5d`. The 22,070,410,112-byte pack was checked against its published hashes. Each projection retains its own inverse global scales, packed E2M1 values, and E4M3 scales per 16 weights.

The EXL3 pack is `lxe/Nanbeige4.3-33B-EXL3-2.5bpw`, converted from the original BF16 checkpoint at `c1093c501cc6460427c0ac634db7223d0450c0ee`. Its 364 decoder matrices use 2.5 bpw trellises, the standard `0x83DCD12D` codebook, and the upstream tensor layout. It is distinct from DSX T16's custom codebook. Embedding and norms remain BF16; the shipped head is FP16. The two weight shards total 14,402,849,061 bytes. Calibration used 32 rows of 512 tokens, with converter patches and hashes included in the model repository.

## Build and run

```bash
CUDA_ARCHS=120 cargo build -p dsx --features cuda --release \
  --bin dsx-nanbeige --bin dsx-nanbeige-server

hf download lxe/Nanbeige4.3-33B-EXL3-2.5bpw --local-dir models/nanbeige-exl3
DSX_NB_BATCH=4096 DSX_NB_ATTN8=2 DSX_NB_PART_SIZE=16384 \
DSX_NB_EXL3_NARROW=1 DSX_NB_EXL3_EXPERIMENT=pf4 \
  ./target/release/dsx-nanbeige-server --model models/nanbeige-exl3 \
  --ctx 262144 --kv-bits 3 --host-embed --head-bits 4 --lookup 3 \
  --host 127.0.0.1 --port 8095
```

For NVFP4, provide that checkpoint directory instead and omit `DSX_NB_EXL3_NARROW`. `DSX_NB_DP4A=1` selects the experimental integer-dot-product NVFP4 decode kernel. Its first-layer validation was bit-identical to native cuBLASLt on the tested input; full-model sampling can differ because reductions and verification shapes differ.

Thinking defaults to off. Requests can enable it through `chat_template_kwargs.enable_thinking`. The original checkpoint's Jinja template renders history and XML tool calls. The API supports `/health`, `/props`, `/v1/models`, and `/v1/chat/completions`, including streaming, reasoning separation, tool results, and measured cached-token counts.

## Context and cache

`--ctx 262144` reserves capacity for 256K tokens. The 3-bit KV codec uses signed 128-wide Hadamard rotations, eight Lloyd-Max centroids, and a fitted BF16 norm, using 50 bytes per head per token. The canonical KV allocation occupies 10.15625 GiB. A BF16 recent window preserves the latest 2,048 keys and values. Its 2,592 slots also cover a 512-token prefill batch and 32-token cutoff alignment. Snapshot copies use the exact GPU payload size, excluding host allocation padding. This codec does not implement the complete QJL TurboQuant algorithm.

`--host-embed` gathers exact embedding rows from the CPU-mapped table. `--head-bits 4`, `6`, or `8` quantizes the output head at load time; zero preserves the 16-bit head. These are runtime choices and do not change the published EXL3 files.

Prefix reuse checks the longest common token prefix. Short rewinds reuse GPU cache; longer rewinds restore a pinned-RAM snapshot of the BF16 recent window. Exact prompt retries restore saved logits as well as KV state. The cache audit verified matching warm, cold, and retried outputs after a 600-token generation and rollback. API usage reports cached tokens separately, so cache hits do not inflate prefill throughput.

Prompt lookup proposes up to eight tokens from repeated text and verifies them with the target model. The coding workload had low acceptance and little benefit; repeated copying had high acceptance. Changing verification shapes can change floating-point rounding and free-running output, even with greedy sampling.

## Current performance

The `nanbeige` branch is intended for integration into DSX `main`; it currently builds a separate `dsx-nanbeige-server` binary. The deployed EXL3 configuration uses four-row trellis prefetching, lookup width 3, 3-bit KV, a 4-bit runtime head, and CPU embedding lookup. Verification-batch graphs remain off.

| Request | Input tokens | Output tokens | Populated context after generation | Decode tok/s | Accepted / proposed lookup tokens |
|---|---:|---:|---:|---:|---:|
| Rust sorted-slice merge | 49 | 512 | 561 | 72.65 | 79 / 141 |
| PostgreSQL monthly ranking | 56 | 512 | 568 | 66.98 | 34 / 81 |
| Probability explanation | 64 | 512 | 576 | 70.25 | 49 / 116 |

The aggregate rate was **69.88 tok/s**, calculated from 1536 output tokens divided by total decode time. All runs allocated **262144-token capacity**. A cold **3968-token** prefix took **1.93977 seconds**, or **2045.60 prefill tok/s**. These capped responses were not graded for task completion. The 100 tok/s general-decode target remains unmet. Raw inputs, acceptance counts, and limits are in [the current receipt](validation/nanbeige/continue-prefetch-summary.json).

The matched 128-target precision check measured perplexity 12.70363 before prefetching and 12.70233 afterward, mean KL 0.0018954, and 96.875% top-1 agreement. This compares runtime configurations with the same weight files; it does not establish a quality improvement. Source-versus-quant drift is recorded separately below.

## Earlier performance measurements

Measurements below used one RTX 5090 with 32 GiB VRAM, an i7-13700, and 128 GiB DDR5-5200. The desktop remained active. All local rows allocated 262,144-token capacity, used 3-bit KV, a 4-bit runtime head, and CPU embedding lookup. The INT8 prefill path was enabled; EXL3 used direct cached loads and NVFP4 used DP4A decode.

| Workload | NVFP4 | EXL3 2.5 bpw |
|---|---:|---:|
| Coding, 30 input + 256 output tokens, serial | 54.37 tok/s | 60.78 tok/s |
| Same coding request, prompt lookup | 55.97 tok/s | 58.94 tok/s |
| Repeated copying, 628 input + 512 output tokens, serial | 51.96 tok/s | 60.99 tok/s |
| Repeated copying with prompt lookup | 199.76 tok/s | 121.14 tok/s |
| Cold 16,384-token retrieval prompt, prefill | 1,931.84 tok/s | 920.93 tok/s |

The copy workload accepted 434 of 546 proposed tokens and matched serial output IDs. The coding lookup output differed from serial and accepted 12 of 85 proposals with NVFP4 and 14 of 88 with EXL3. Copy speed does not establish general coding throughput.

A separate NVFP4 test populated 262,016 tokens and retrieved the buried key correctly. Prefill took 1,579.32 seconds, or 165.90 tok/s. Its seven-token answer took 1.315 seconds, or 5.32 tok/s. This is the populated-context result; the short-prompt rows above measure a different workload. The EXL3 pack's local retrieval test populated 16,384 tokens and passed. A populated 256K test with EXL3 remains pending.

The 100 tok/s general generation target has not been reached. Shorter-prompt prefill exceeds 2,000 tok/s with NVFP4; that rate does not apply to a populated 256K context. Raw receipts are under `validation/nanbeige/`.

## Quality and validation

The quality fixture is a 4,096-token prefix of WikiText-2 raw validation, scoring its final 128 tokens. This short sample is a precision check, not a task-completion benchmark.

| Configuration | Perplexity |
|---|---:|
| Source checkpoint, ExLlamaV3 reference forward | 10.2676 |
| EXL3 2.5 bpw, same reference forward | 11.8915 |
| EXL3, DSX with INT8 prefill, 3-bit KV and 4-bit head | 12.7581 |
| NVFP4, DSX with INT8 prefill, DP4A decode, 3-bit KV and 4-bit head | 10.3605 |

The two reference rows ran on a 96 GB RTX PRO 6000 Blackwell. Their mean KL was 0.17443 and next-token agreement was 85.9375%. The DSX rows additionally include runtime precision choices and a batched prefix followed by tokenwise teacher forcing. They do not isolate weight quantization alone. Coding, reasoning, and agent task-completion quality remain unmeasured for this quant.

Independent checks compare packed EXL3 decoding with a CPU implementation, compare fused GEMV with BF16 dequantization and rotations, and compare attention with an f64 causal-softmax reference. The 65-row attention check covers partial tiles, multiple tiles, and several GQA heads. Template history, reasoning stream boundaries, and typed XML tool arguments have passing tests. The fractional quantizer patch produced identical indices and values to the original implementation on a 17-tile check.

The earlier 8K graph and fixed-shape equivalence receipts remain as historical checks. They do not establish bitwise equivalence for the newer prefill, quantized attention, and speculative paths.

## Implementation

- `nvidia/src/nanbeige.rs`: weight loading, native FP4 plans, EXL3 execution, KV allocation, graph capture, prefix reuse, and sampler integration.
- `nvidia/src/kernels/nanbeige.cuh`: quantization, normalization, RoPE, attention, head kernels, and NVFP4 GEMV experiments.
- `nvidia/src/kernels/nanbeige-exl3.cu`: standalone EXL3 kernels, with MIT-licensed upstream headers and provenance under `vendor/exllamav3/`.
- `nvidia/src/bin/nanbeige-server.rs`: chat template, OpenAI protocol, tools, reasoning, web UI, and cache-aware timing metadata.

## Final tuning on October 25

The 4,096-token batch option processed a cold 3,968-token WikiText prefix in 1.8862 seconds (2,103.75 tok/s) and the 16,384-token retrieval prompt in 11.8008 seconds (1,388.38 tok/s), returning the correct key. The corresponding 128-token perplexity check was 12.7036. These are short-sample measurements; the lower perplexity does not establish a quality improvement. Larger batches use canonical 3-bit KV for keys that have left the BF16 ring.

The deployed EXL3 configuration uses `DSX_NB_BATCH=4096`, `DSX_NB_ATTN8=2`, `DSX_NB_PART_SIZE=16384`, `DSX_NB_EXL3_NARROW=1`, `DSX_NB_EXL3_EXPERIMENT=pf4`, and `--ctx 262144 --kv-bits 3 --host-embed --head-bits 4 --lookup 3`. At batch 2,048, the same decode policy measured 63.13 tok/s serial coding and 65.11 with lookup; repeated copying measured 151.97 with lookup. The 100 tok/s general-generation target remains unmet.

## Pi file paths

The live Pi smoke completed bash, write, and read calls in the existing tmux session. Its first relative-path read failed, then an absolute-path read verified the written file. The launcher now appends an instruction to use absolute file paths for read, write, and edit tools. This avoids the observed relative-path ambiguity; it is not a change to the Pi tool implementation.

## Decode prefetch experiment

`DSX_NB_EXL3_EXPERIMENT=pf4` prefetches four packed trellis rows ahead in the 2.5 bpw single-token kernel. On the same 30-token coding prompt with 256 output tokens and 262144-token allocated capacity, serial decode increased from 62.05 to 65.91 TPS. Prompt lookup with maximum width 3 increased from 64.51 to 68.61 TPS, accepting 16 of 36 proposed tokens in 12 attempts. This is one coding prompt; it does not establish general 100 TPS throughput.

The same 3968-token WikiText prefill followed by 128 scored targets measured perplexity 12.70363 before prefetching and 12.70233 afterward. Mean KL from baseline to the prefetch configuration was 0.0018954, with 96.875% top-1 agreement. Kernel occupancy changes the activation quantization slices, so unchanged weight files do not imply identical logits. The small perplexity difference does not demonstrate a quality improvement.

The optional `DSX_NB_EXL3_CARVEOUT=0..100` experiment changes the single-token shared-memory/L1 preference. A value of 50 produced 68.55 TPS with lookup, effectively the same as the default preference, so it is not enabled in the service. A warp-register load experiment passed independent decoder checks but was slower in the projection benchmark and was removed. Receipts for both experiments remain in `validation/nanbeige/continue-*`.

The follow-up suite used 512 output tokens each for Rust code (49 input tokens), SQL (56), and a probability explanation (64). With prefetching and lookup width 3, rates were 72.65, 66.98, and 70.25 TPS respectively, or 69.88 TPS over the combined decode time. Accepted/drafted tokens were 79/141, 34/81, and 49/116. Final populated contexts were 561, 568, and 576 tokens, separate from allocated 262144-token capacity. Outputs were capped, and task success was not graded.

`DSX_NB_SPEC_GRAPHS=1` optionally captures verification batches of at most nine rows. Capture keys distinguish row count, attention split/GQA geometry, and whether all logits are collected. The same three prompts produced identical output token IDs with graphs on and off. Measured graph rates were 72.63, 68.53, and 69.89 TPS; there was no consistent improvement, so the live service leaves verification graphs off. Single-token CUDA graphs remain enabled. Warm/cold prefix branching and exact retry checks also passed.

The final prefetch configuration measured 2045.60 TPS for a cold 3968-token prefix. Its full-model quality receipt, benchmark inputs, graph comparison, and service configuration are saved under `validation/nanbeige/continue-*`. Earlier `continue-suite` runs overlapped CPU compilation and are excluded from the aggregate result; the `continue-graphs` comparisons started after compilation completed.

## EXL3 packaging audit

The published pack follows the upstream EXL3 tensor layout and two-shard convention. Root `config.json` and the detailed `quantization_config.json` both retain the canonical fractional `bits: 2.5`. A previous display workaround had renamed the root field to `bits_per_weight`; that deviation was removed. The Hub's integer-only warning and packed-element parameter count require EXL3 support in its metadata parser. Weight shards and checksums are unchanged.

The [parameter audit](validation/nanbeige/exl3-logical-parameter-audit.json) reconstructs logical matrix sizes from EXL3 tile shapes and agrees exactly with both source checkpoints: 33,118,977,024 parameters for Nanbeige, and 32,762,123,264 for Turboderp's Qwen3-32B EXL3 example. Its stored tensor elements total about 5.95B despite the 32.76B logical parameter count. This is why multiplying or renaming stored tensor dimensions to change the Hub badge would misrepresent the file format.

## Long-context decode investigation

A passive ten-second sample of the active Doom run measured approximately 23.52 output TPS while it streamed reasoning. The request began with 2,566 prompt tokens; its final generated count and populated context were not yet available. This live sample is separate from the controlled 69.88 TPS short-context suite.

The [attention partition sweep](validation/nanbeige/attention-partitions.md) measures the CUDA GQA kernel and merge on the rented RTX PRO 6000 Blackwell without interrupting the local RTX 5090 run. Larger partitions did not consistently improve long-context attention. The production partition remains 512 tokens; no end-to-end throughput gain is claimed. The new `--attention-bench` option supports further isolated measurements.

The [Doom comparison setup](validation/nanbeige/doom-comparison/README.md) records matched task prompts for local EXL3 Nanbeige, original 16-bit Nanbeige on a rented reference backend, and the existing DeepSeek DSX runner on two RTX 3090 GPUs. Its manifests identify the hardware, precision, and reasoning-control differences that limit attribution to weight quantization.


The final cancelled Doom request generated 43,931 tokens at an average 29.10 TPS, reaching 46,497 prompt-plus-generated tokens. It accepted 4,230 of 13,581 lookup drafts and produced no game file. This long reasoning run is distinct from the short-context suite. The user stopped the comparison and optimization; the rented reference instance has been released. Changes remain on `nanbeige`, with no merge to `main`.

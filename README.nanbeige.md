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
DSX_NB_ATTN8=2 DSX_NB_PART_SIZE=16384 DSX_NB_EXL3_NARROW=1 \
  ./target/release/dsx-nanbeige-server --model models/nanbeige-exl3 \
  --ctx 262144 --kv-bits 3 --host-embed --head-bits 4 --lookup 8 \
  --host 127.0.0.1 --port 8095
```

For NVFP4, provide that checkpoint directory instead and omit `DSX_NB_EXL3_NARROW`. `DSX_NB_DP4A=1` selects the experimental integer-dot-product NVFP4 decode kernel. Its first-layer validation was bit-identical to native cuBLASLt on the tested input; full-model sampling can differ because reductions and verification shapes differ.

Thinking defaults to off. Requests can enable it through `chat_template_kwargs.enable_thinking`. The original checkpoint's Jinja template renders history and XML tool calls. The API supports `/health`, `/props`, `/v1/models`, and `/v1/chat/completions`, including streaming, reasoning separation, tool results, and measured cached-token counts.

## Context and cache

`--ctx 262144` reserves capacity for 256K tokens. The 3-bit KV codec uses signed 128-wide Hadamard rotations, eight Lloyd-Max centroids, and a fitted BF16 norm, using 50 bytes per head per token. The canonical KV allocation occupies 10.15625 GiB. A BF16 recent window preserves the latest 2,048 keys and values. Its 2,592 slots also cover a 512-token prefill batch and 32-token cutoff alignment. Snapshot copies use the exact GPU payload size, excluding host allocation padding. This codec does not implement the complete QJL TurboQuant algorithm.

`--host-embed` gathers exact embedding rows from the CPU-mapped table. `--head-bits 4`, `6`, or `8` quantizes the output head at load time; zero preserves the 16-bit head. These are runtime choices and do not change the published EXL3 files.

Prefix reuse checks the longest common token prefix. Short rewinds reuse GPU cache; longer rewinds restore a pinned-RAM snapshot of the BF16 recent window. Exact prompt retries restore saved logits as well as KV state. The cache audit verified matching warm, cold, and retried outputs after a 600-token generation and rollback. API usage reports cached tokens separately, so cache hits do not inflate prefill throughput.

Prompt lookup proposes up to eight tokens from repeated text and verifies them with the target model. The coding workload had low acceptance and little benefit; repeated copying had high acceptance. Changing verification shapes can change floating-point rounding and free-running output, even with greedy sampling.

## Measured performance

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

The deployed EXL3 configuration uses `DSX_NB_BATCH=4096`, `DSX_NB_ATTN8=2`, `DSX_NB_PART_SIZE=16384`, `DSX_NB_EXL3_NARROW=1`, and `--ctx 262144 --kv-bits 3 --host-embed --head-bits 4 --lookup 3`. At batch 2,048, the same decode policy measured 63.13 tok/s serial coding and 65.11 with lookup; repeated copying measured 151.97 with lookup. The 100 tok/s general-generation target remains unmet.

## Pi file paths

The live Pi smoke completed bash, write, and read calls in the existing tmux session. Its first relative-path read failed, then an absolute-path read verified the written file. The launcher now appends an instruction to use absolute file paths for read, write, and edit tools. This avoids the observed relative-path ambiguity; it is not a change to the Pi tool implementation.

## Decode prefetch experiment

`DSX_NB_EXL3_EXPERIMENT=pf4` prefetches four packed trellis rows ahead in the 2.5 bpw single-token kernel. On the same 30-token coding prompt with 256 output tokens and 262144-token allocated capacity, serial decode increased from 62.05 to 65.91 TPS. Prompt lookup with maximum width 3 increased from 64.51 to 68.61 TPS, accepting 16 of 36 proposed tokens in 12 attempts. This is one coding prompt; it does not establish general 100 TPS throughput.

The same 3968-token WikiText prefill followed by 128 scored targets measured perplexity 12.70363 before prefetching and 12.70233 afterward. Mean KL from baseline to the prefetch configuration was 0.0018954, with 96.875% top-1 agreement. Kernel occupancy changes the activation quantization slices, so unchanged weight files do not imply identical logits. The small perplexity difference does not demonstrate a quality improvement.

The optional `DSX_NB_EXL3_CARVEOUT=0..100` experiment changes the single-token shared-memory/L1 preference. A value of 50 produced 68.55 TPS with lookup, effectively the same as the default preference, so it is not enabled in the service. A warp-register load experiment passed independent decoder checks but was slower in the projection benchmark and was removed. Receipts for both experiments remain in `validation/nanbeige/continue-*`.

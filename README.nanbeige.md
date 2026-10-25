# DSX Nanbeige NVFP4 fork

This branch adds a Nanbeige4.3-33B engine to DSX. It runs directly in Rust and CUDA, using native SM120 NVFP4 multiplication through CUDA's cuBLASLt library. It does not launch vLLM, PyTorch, or llama.cpp.

The original DSX checkout is `/home/lxe/dsx-correctness`, pinned at `53819a1781527e9e4f2c09ded29ba666dc9fc84f`. This fork retains DSX's GPU context, memory and launch boundary, safetensors reader, token sampler, FIFO engine worker, cancellation handling, and embedded web UI. Nanbeige uses ordinary dense Llama attention and feed-forward layers, so its model execution and kernels are separate from DeepSeek's MoE, MLA, Engram and expert streaming code.

## Checkpoint

`primitive-ai/Nanbeige4.3-33B-NVFP4` revision `4171a7d6eabfffc37b9d89629fb75d28fd5c2c5d`. All downloaded files were checked against the pinned inventory and the published LFS SHA256 hashes. The pack is 22,070,410,112 bytes across four safetensors files. It contains 364 NVFP4 decoder matrices; embeddings, output head and normalization weights remain BF16.

The checkpoint's global scales are inverse scales. Each projection retains its own weight and activation global scales. Weights use packed E2M1 values and E4M3 scales per 16 values. The loader swizzles scale storage for native cuBLASLt FP4 multiplication without requantizing weights. Activation quantization happens on the GPU.

## Run

```sh
CUDA_ARCHS=120 cargo build -p dsx --features cuda --release --bin dsx-nanbeige --bin dsx-nanbeige-server
./target/release/dsx-nanbeige --inspect
./target/release/dsx-nanbeige --validate
./target/release/dsx-nanbeige-server --ctx 8192 --port 8095
```

The API exposes `/health`, `/v1/models`, and `/v1/chat/completions`, including streaming, reasoning separation, XML tool calls, tool result replay and prefix reuse. Chat rendering uses the checkpoint's original Jinja template with Python-compatible methods and JSON formatting. The server uses DSX's single owning engine thread for the CUDA context.

Thinking defaults to off. A request can enable it with `chat_template_kwargs.enable_thinking`; reasoning effort levels do not impose a token budget in this model. The earlier Doom attempt was stopped at the user's request after extended reasoning without implementation.

## Validation and limitations

Receipts are stored in `/home/lxe/nanbeige-runner/receipts`. The independent CPU check computes E2M1/E4M3 decoding and activation quantization, then compares selected projection rows with native SM120 output. It also checks identical batch rows. Separate tests cover reasoning delimiters, template history, and typed XML tool arguments.

The initial native path measured about 45 output tokens/s on a 256-token greedy Python coding response at short context. A custom SIMT decode experiment measured about 39 and remains opt-in (`--simt-decode`). These measurements do not establish an improvement over the prior llama.cpp runner. CUDA graph and ordinary execution produced the same 256 output token IDs in the initial check.

This first implementation uses BF16 KV storage and an 8K service context. It does not yet implement the 96K quantized KV cache of the previous llama.cpp setup. Memory checks reject configurations that do not fit; no CPU expert offload applies to this dense model. The checkpoint publisher has not provided completed BF16 quality-drift measurements. Local arithmetic, code and tool smoke checks establish execution behavior, not retained benchmark quality.

A fixed 32-token multiplication shape keeps prefill and decode numerics consistent. A 25-token prompt and a 250-token prompt spanning multiple chunks produced bit-identical final logits for batched and sequential execution across all 166,912 vocabulary entries. Before fixing the shape, different cuBLASLt algorithm selections caused differences to grow through the network.

The live API measured 43.86 output tokens/s on a 256-token greedy coding response at 8K capacity. Pi completed real bash, write and read calls with thinking off. The generated prime-checking function passed 10,021 integer cases against an independent reference. Selected receipts are copied into `validation/nanbeige/`. The active endpoint is http://100.76.24.95:8095 and the user service is `dsx-nanbeige.service`. `/home/lxe/nanbeige-runner/run-pi.sh` starts Pi against the custom server.

The prototype prefill attention kernel uses partitioned online softmax rather than a tuned tensor-core FlashAttention kernel. Prefill performance and long-context attention remain optimization work. GPU counter profiling was attempted; this host currently denies access to NVIDIA performance counters. No host-wide profiling configuration was changed.

## Source files

- `nvidia/src/nanbeige.rs`: model loader, NVFP4 plans, dense forward pass, prefix reuse, graph capture and DSX Engine adapter.
- `nvidia/src/kernels/nanbeige.cuh`: activation quantization, scale swizzle, normalization, RoPE, causal attention, residuals, SwiGLU and experimental GEMV.
- `nvidia/src/bin/nanbeige-server.rs`: Nanbeige prompt and OpenAI protocol adapter.
- `nvidia/templates/nanbeige.jinja`: upstream chat template.

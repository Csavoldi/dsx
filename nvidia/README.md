# dsx: the NVIDIA runner

The current default model is T16 v1. [Current state](../docs/current-state.md) records integrated kernels,
configuration switches, benchmark provenance and the open static-residency decode check. Without `--features cuda`,
GPU subcommands such as `gen`, `eval` and `cmp` are absent; the CPU reference and GGUF tools remain available.

DeepSeek-V4.1-Flash(-1017) inference for one machine: 2x RTX 3090, i7-12700K, 128 GB DDR4-3200 (or one RTX 5090).
Rust + CUDA (cudarc driver API, one nvcc fatbin for sm_86 and sm_120). Config, readers, block formats, the CPU
reference and the server are in [`../core`](../core) (re-exported as `dsx::model`, `dsx::server`, ...). Plan and
numbers: ../DESIGN.md, ../FINDINGS.md, ../bench/RESULTS.md.

## Build

```bash
cargo build --release -p dsx --features cuda    # CUDA_ARCHS=86 for RTX 30xx (default 86,120); binaries in ../target/release
cargo test --release                            # CPU tests (core: formats, numerics, server; CPU expert kernels)
cargo test --release -p dsx --features cuda     # + GPU kernel tests (gemv vs CPU decoders, top-k, handshake)
```

## Commands

| | |
|---|---|
| `dsx ref --model DIR --ids ...` | CPU reference forward, token by token, exact model.py semantics (FP8/FP4 checkpoint dir, or a GGUF + `--engram DIR`) |
| `dsx cmp --model M --ids ...` | GPU decode next to the CPU reference, per-token logit diff; `--no-ref -n N` decodes greedily; `--main-gb/--aux-gb/--arena-gb` place experts |
| `dsx gen --model GGUF --engram DIR --ids @file -n N` | chunked prefill, then decode; `--check` reruns the prompt token by token and compares; `--bench-verify T` / `--check-verify T` time / check T-token verify passes |
| `dsx eval --model GGUF --engram DIR --eval DIR` | KL / top-1 / perplexity on Lucebox's eval set (`--decode`: token by token; `--from N`: positions N+) |
| `dsx spec --model DIR --ids ...` | DSpark (the checkpoint's draft stack) acceptance on the CPU reference |
| `dsx merge FIRST_SHARD OUTPUT` | join split GGUFs and verify tensor bytes |
| `dsx split INPUT PREFIX --shards N` | split a GGUF and verify tensor bytes |
| `dsx score --model GGUF --tokenizer FILE --request JSON --continuation TEXT` | use `dsx score --help` for request/continuation teacher-forced scoring arguments |

Debug: `DSX_DUMP=dir` (per-layer residual streams, ref vs GPU), `DSX_ROUTES=1` (expert picks), `DSX_PROF=1` (time per section),
`DSX_SYNC_MOE=1` (old host-synchronous expert path), `DSX_MOE_DEVICE=1` (device-driven expert kernels in the sync path),
`DSX_OLD_MOE=1` (VRAM experts keep their GGUF blocks and run moe_gu / moe_down instead of the repacked int8-mma kernels).
Placement experiments: `DSX_GPUS=0,2`, `DSX_AUX_ADAPTIVE=1/0`, `DSX_AUX_STREAM=1/0` (other GPUs' share of prefill's
expert stream), `DSX_STREAM_MAIN_W=x` (the main GPU's weight in that split, default 2), `DSX_ADAPT=N`, `DSX_ADAPT_AUX=N`
(in-flight swaps on other GPUs), `DSX_DECAY=x` (usage decay per token, default 0.97), `DSX_ROUTE_BIAS=x` (selection
bonus for VRAM experts; changes the model's routing). Profiling: `DSX_EVPROF=1` (GPU event times per decode stage).
Sensitivity: `DSX_NOISE=A-B:REL` (Gaussian noise on layers A..B's routed expert outputs in prefill).

## Layout

| file | |
|---|---|
| `../core/src/` | `gguf.rs`, `safetensors.rs` (mmap readers), `formats.rs`, `numerics.rs` (block decoders: Q8_0, Q2_0, PTQ1_0, Q3_K, Q6_K, FP8 32x32, FP4; fake quant; RoPE + YaRN), `model.rs` (config, weight handles, the CPU reference forward), `dspark.rs` (draft stack reference), `server/` |
| `server/real.rs` | the GPU engine behind the server (prefix reuse, snapshots) |
| `cpu/` | AVX2 expert kernels (with AVX-VNNI or AVX-512 VNNI where the CPU has it), pinned thread pool, per-layer CPU MoE |
| `gpu.rs` | CUDA boundary: contexts, buffers, hugepage host memory, launches, cuBLAS, stream flag ops |
| `kernels/` | `dsx.cu` (dense path, attention, compressor, indexer, router, prefill helpers), `moe.cuh`, `expert_fast.cuh` (repacked VRAM experts, int8 mma, T tokens), `topk.cuh` |
| `gpumodel.rs` | GPU decode: GPU roles by measured link, dense half on the main GPU, experts on every GPU and the CPU, sync-free per-layer handoff |
| `gpumodel/prefill.rs` | prefill: encoder for every token, decoder dependency cone, grouped expert GEMMs |
| `experts.rs` | VRAM expert stores (residency table) and the hugepage RAM arena |
| `handoff.rs` | mapped-memory exchange + CPU expert worker thread |
| `topk.rs` | GPU radix top-k / candidate blocks for the sparse indexer |

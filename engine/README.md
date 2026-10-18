# ds41

DeepSeek-V4.1-Flash(-1017) inference for one machine: 2x RTX 3090, i7-12700K, 128 GB DDR4-3200 (`192.168.0.102`).
Rust + CUDA (cudarc driver API, one nvcc fatbin for sm_86 and sm_120). Plan and numbers: ../DESIGN.md, ../FINDINGS.md,
../bench/RESULTS.md.

## Build

```bash
cargo build --release --features cuda          # CUDA_ARCHS=86 on .102 (default 86,120)
cargo test --release                            # CPU tests (formats, numerics, CPU expert kernels)
cargo test --release --features cuda            # + GPU kernel tests (gemv vs CPU decoders, top-k, handshake)
```

## Commands

| | |
|---|---|
| `ds41 ref --model DIR -ids ...` | CPU reference forward, token by token, exact model.py semantics (FP8/FP4 checkpoint dir, or a GGUF + `--engram DIR`) |
| `ds41 cmp --model M --ids ...` | GPU decode next to the CPU reference, per-token logit diff; `--no-ref -n N` decodes greedily; `--gpu0-gb/--gpu1-gb/--arena-gb` place experts |
| `ds41 gen --model GGUF --engram DIR --ids @file -n N` | chunked prefill, then decode; `--check` reruns the prompt token by token and compares |
| `ds41 spec --model DIR --ids ...` | DSpark (the checkpoint's draft stack) acceptance on the CPU reference |

Debug: `DS41_DUMP=dir` (per-layer residual streams, ref vs GPU), `DS41_ROUTES=1` (expert picks), `DS41_PROF=1` (time per section),
`DS41_SYNC_MOE=1` (old host-synchronous expert path), `DS41_MOE_DEVICE=1` (device-driven expert kernels in the sync path),
`DS41_OLD_MOE=1` (VRAM experts keep their GGUF blocks and run moe_gu / moe_down instead of the repacked int8-mma kernels).

## Layout

| file | |
|---|---|
| `gguf.rs`, `safetensors.rs` | mmap readers (GGUF splits, HF shards) |
| `formats.rs`, `numerics.rs` | block decoders (Q8_0, Q2_0, PTQ1_0, Q3_K, Q6_K, FP8 32x32, FP4), FP8/FP4 fake quant, RoPE + YaRN |
| `model.rs` | config, weight handles (`Mat`), the CPU reference forward |
| `dspark.rs` | DSpark draft stack (reference) |
| `cpu/` | AVX2 + AVX-VNNI expert kernels, pinned thread pool, per-layer CPU MoE |
| `gpu.rs` | CUDA boundary: contexts, buffers, hugepage host memory, launches, cuBLAS, stream flag ops |
| `kernels/` | `ds41.cu` (dense path, attention, compressor, indexer, router, prefill helpers), `moe.cuh`, `expert_fast.cuh` (repacked VRAM experts, int8 mma, T tokens), `topk.cuh` |
| `gpumodel.rs` | GPU decode: dense half on GPU0, experts on GPU0 / GPU1 / CPU, sync-free per-layer handoff |
| `gpumodel/prefill.rs` | prefill: encoder for every token, decoder dependency cone, grouped expert GEMMs |
| `experts.rs` | VRAM expert stores (residency table) and the hugepage RAM arena |
| `handoff.rs` | mapped-memory exchange + CPU expert worker thread |
| `topk.rs` | GPU radix top-k / candidate blocks for the sparse indexer |

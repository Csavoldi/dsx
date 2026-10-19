# dsx-amd: the MI325X runner

DeepSeek-V4.1-Flash-1017 on one AMD Instinct MI325X (gfx942, 256 GiB HBM3E): the whole model except the Engram
tables in HBM, one HIP graph per decode step, MFMA prompt kernels, DSpark speculative decoding. Rust + HIP; the
config, readers, CPU reference and server come from [`../core`](../core). Measurements: [../FINDINGS-AMD.md](../FINDINGS-AMD.md).

## Build

```bash
cargo build --release -p dsx-amd    # from the repo root; ROCm 7 (hipcc at /opt/rocm/bin, or HIPCC=...)
```

Without ROCm the host code still compiles (`cargo check -p dsx-amd`), with no kernels embedded.

## Commands

| | |
|---|---|
| `dsx-amd serve --model DIR` | the API server and web UI; `DIR` is the official checkpoint directory (config, tokenizer, Engram shards, `dsx-engram.json`) |
| `dsx-amd export --model DIR --rank RANK --out FILE` | load the checkpoint with the MXFP4 + 3-bit placement and write it as one GGUF (split with `dsx split`) |
| `dsx-amd eval --model DIR --eval EVALDIR` | KL / top-1 / perplexity on Lucebox's 1017 eval set |
| `dsx-amd check --model DIR --ids ...` | GPU decode (or `DSXA_PREFILL=n` the prompt path) against the CPU reference, per layer with `DSX_DUMP` |
| `dsx-amd bench`, `selftest` | decode speed at a context length; kernel tests against CPU implementations |
| `dsx-amd dev --model DIR` | load once, then run commands written to the `--cmd` file (default `/root/dev.cmd`) (`bench`, `prof`, `pbench`, `vtest`, `spec`, `eval`, `reload KERNELS.hsaco`, ...) |

| Environment | |
|---|---|
| `DSXA_WEIGHTS=FILE.gguf` | load a dsx-amd export (the first shard) instead of re-quantizing the checkpoint |
| `DSXA_DSPARK=3` (or `2`) | load the DSpark draft layers (official shards 44-46), experts in a 3-bit (2-bit) codebook; greedy requests use them |
| `DSXA_SPEC_MIN=2.0` | accepted drafts per pass below which speculation pauses for 15 plain tokens |
| `DSXA_VGRAPH=0` | verify / draft passes without HIP graphs |
| `DSXA_HEAD_FP8=1`, `DSXA_V2=1`, `DSXA_MM1=1`, `DSXA_ATTN_T=1` | alternative kernels (FP8 head, global-activation GEMVs, the first prompt GEMM, per-head-group prompt attention) |

## Layout

| | |
|---|---|
| `src/decode.rs` | weights to HBM (placement by free memory), the decode step recorded as one HIP graph |
| `src/prefill.rs` | the prompt path (2048-token chunks), also run for verify passes and DSpark draft blocks |
| `src/verify.rs`, `kernels/verify.h` | multi-token GEMVs for verify passes (positions as MFMA columns, device-side expert grouping) |
| `src/dspark.rs` | the DSpark draft stack, window updates, the speculative loop |
| `src/load.rs`, `src/ck.rs`, `src/export.rs` | checkpoint tensors to device tile layouts; checkpoint or export as one source; the GGUF export |
| `src/serve.rs` | the server's engine (prefix reuse, greedy through DSpark when loaded) |
| `kernels/` | `dsxa.hip` (decode kernels), `gemv.h` (weight tile formats, MFMA GEMV waves), `prefill.h` (prompt kernels), `verify.h` |
| `box/`, `tools/`, `lab/` | setup scripts for a rented box, codebook fitting and eval conversion, kernel experiments |
| `results/` | raw logs behind the numbers |

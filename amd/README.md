# dsx-amd: the MI325X runner

DeepSeek-V4.1-Flash-1017 on one AMD Instinct MI325X (gfx942, 256 GiB HBM3E): the whole model except the Engram
tables in HBM, one HIP graph per decode step, MFMA prompt kernels, DSpark speculative decoding. Rust + HIP; the
config, readers, CPU reference and server come from [`../core`](../core). Measurements: [../FINDINGS-AMD.md](../FINDINGS-AMD.md).

## Run

| 1x MI325X | |
|---|---:|
| Decode at 4K / 128K / 256K context | 114 / 112 / 110 tok/s |
| With DSpark drafts (greedy) | 1.4-1.7x on code, 1.2-1.4x on math, ~1.0x on prose |
| Prefill, 4K / 16K / 128K-token prompt | 1,840 / 1,640 / 1,160 tok/s |
| Quality vs FP8 (Lucebox eval) | KL 0.049, top-1 90.6% |
| Cost at $3.80/h, one stream | $9.3 per million generated tokens, $0.64 per million prompt tokens |

From the repo root:

```bash
hf download lxe/DeepSeek-V4.1-Flash-1017-dsx-mxfp4 --local-dir dsx-mxfp4
hf download deepseek-ai/DeepSeek-V4.1-Flash-1017 --local-dir v41-1017 \
  --include config.json tokenizer.json model.safetensors.index.json model-00047-of-00048.safetensors model-00048-of-00048.safetensors
cp dsx-mxfp4/dsx-engram.json v41-1017/

cargo build --release -p dsx-amd      # ROCm 7, gfx942
DSXA_WEIGHTS=dsx-mxfp4/DeepSeek-V4.1-Flash-1017-dsx-mxfp4-00001-of-00006.gguf \
  ./target/release/dsx-amd serve --model v41-1017 --ctx 262144 --host 0.0.0.0 --port 8090
```

`--model` is the official checkpoint directory (config, tokenizer, Engram shards, plus `dsx-engram.json` from the
quant repo). `DSXA_DSPARK=3` adds the checkpoint's DSpark draft layers for greedy requests; they are read from the
official `mtp.*` tensors (add `model-00044-of-00048.safetensors` .. `model-00046-of-00048.safetensors` to the
download) and take 6.5 GB of HBM, so 1,613 cold experts go to 3-bit
(KL 0.056). Measurements and design notes: [../FINDINGS-AMD.md](../FINDINGS-AMD.md).

The server is the same as the NVIDIA runner's (web UI at `/`, OpenAI-compatible `/v1`, Anthropic `/v1/messages`).

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

---
license: mit
base_model:
- deepseek-ai/DeepSeek-V4.1-Flash-1017
base_model_relation: quantized
pipeline_tag: text-generation
library_name: dsx
tags:
- deepseek
- deepseek-v4.1
- moe
- quantized
- mxfp4
- 3-bit
- mixed-precision
- amd
- mi325x
- rocm
---

# DeepSeek-V4.1-Flash-1017 · dsx MXFP4 + 3-bit

DeepSeek-V4.1-Flash-1017 (552B MoE + 196B Engram) in 270 GB, built to run entirely from the HBM of one AMD Instinct
MI325X with exact routing. Most routed experts keep the checkpoint's own MXFP4 weights; the least-used ones get a fitted
3-bit codebook. Runs on [**dsx**](https://github.com/lxe/dsx)'s AMD runner (`amd/`, Rust + HIP).

| | |
|---|---|
| Size | 270 GB (6 GGUF shards) + the official Engram shards (203 GB) |
| Format | dense weights FP8 as in the checkpoint; 57% of routed experts MXFP4 (unchanged), 43% 3-bit codebook + e4m3 scale per 32 |
| Routing | exact: no expert pruning, no router bias |
| Quality vs FP8 | KL 0.049 · top-1 90.6% · perplexity 8.71 (FP8: 8.56) |
| Speed, 1x MI325X | 108-111 tok/s decode at 4K-256K context · 1,150-1,660 tok/s prefill |
| Runtime | dsx `amd/` (OpenAI / Anthropic API, web UI) |

## Quality

Measured on [Lucebox's 1017 eval set](https://huggingface.co/Lucebox/DeepSeek-V4.1-Flash-1017-ROCMFP-GGUF):
8 × 1024 tokens against the FP8 model's top-256 log-probs, teacher-forced through the decoder.

| | Size | KL ↓ | Top-1 ↑ | Perplexity ↓ |
|---|---:|---:|---:|---:|
| FP8 / FP4 original | 299 GB | 0 | 100% | 8.56 |
| **dsx MXFP4 + 3-bit** | **270 GB** | **0.049** | **90.6%** | **8.71** |
| Lucebox ROCMFP23S | 179 GiB | 0.194 | 81.3% | 9.41 |
| Lucebox ROCMFP 2.5 bpw | 158 GiB | 0.253 | 78.7% | 9.78 |
| dsx 2-bit | 148 GB | 0.395 | 74.1% | 10.93 |

Sizes exclude the Engram tables (and, for the original, the vision and draft layers). Measured on these shards;
repeated runs give KL 0.047-0.051, because float atomics in the runner sum in varying order and that flips
near-tied expert choices.

Greedy reasoning (temperature 0, thinking on, reasoning effort low), two competition-math prompts: 6,624 and 13,482
generated tokens with boxed answers. The hosted model needed 5.6K and 21K on the same prompts.

## Speed

One MI325X (256 GiB HBM3E), greedy, exact routing.

| Context | Decode |
|---|---:|
| 4K | 110.7 tok/s |
| 128K | 109.0 tok/s (108.3 after a real 128K prompt) |
| 256K | 107.9 tok/s (107.4 after a real 256K prompt) |

| Prompt | Prefill |
|---|---:|
| 4K tokens | 1,658 tok/s |
| 16K tokens | 1,154 tok/s |

Long prompts slow down with length because the attention indexer scores every earlier compressed position.

## Run

```bash
# the quant: GGUF shards, loaded as one model
hf download lxe/DeepSeek-V4.1-Flash-1017-dsx-mxfp4 --local-dir dsx-mxfp4

# config, tokenizer and Engram tables from the official checkpoint
hf download deepseek-ai/DeepSeek-V4.1-Flash-1017 --local-dir v41-1017 \
  --include config.json tokenizer.json model.safetensors.index.json model-00047-of-00048.safetensors model-00048-of-00048.safetensors
cp dsx-mxfp4/dsx-engram.json v41-1017/

# ROCm 7 (gfx942), Rust. The AMD runner (dsx `amd/`) is being merged into the public repo; until it lands, these
# build steps need that branch.
git clone https://github.com/lxe/dsx && cd dsx/amd && cargo build --release

DSXA_WEIGHTS=../../dsx-mxfp4/DeepSeek-V4.1-Flash-1017-dsx-mxfp4-00001-of-00006.gguf \
  ./target/release/dsx-amd serve --model ../../v41-1017 --ctx 262144 --host 0.0.0.0 --port 8090
```

Open `http://localhost:8090/` for the web UI, or point any OpenAI-compatible client at `http://localhost:8090/v1`.

## Hardware

One GPU with 256 GiB of memory (AMD Instinct MI325X, gfx942); the weights take 271.6 GB of HBM with context up to
256K tokens. The Engram tables (203 GB) are read from disk through the page cache, 48 rows per token; ~160 GB of RAM.

## How it was made

- **Dense weights** (attention, shared experts, routers, Engram projections): the checkpoint's FP8 with its E8M0
  32×32 block scales, unchanged. Embeddings and the output head: bf16, unchanged.
- **Routed experts:** the 8,804 most used experts (by routing frequency measured on agentic text) keep the
  checkpoint's MXFP4 weights. The other 6,556 are re-quantized from those FP4 values to a 3-bit codebook of 8 levels
  (fitted by Lloyd iterations on a sample of experts, stored as FP8 values) with one e4m3 scale per 32 weights and a
  multiplier per matrix, round-to-nearest with a per-group scale search.
- The split is chosen so the whole model fits in 256 GiB of HBM next to the caches for 256K tokens of context.

## Files

| File | |
|---|---|
| `DeepSeek-V4.1-Flash-1017-dsx-mxfp4-0000k-of-00006.gguf` | the model in GGUF shards; the runner loads them from the first |
| `dsx-engram.json` | Engram hashing constants (copy next to the official checkpoint files) |
| `SHA256SUMS` | checksums of the shards |
| `expert-rank.json` | expert hotness used to choose which experts stay MXFP4 |
| `config.json` | build metadata |
| `results/` | raw quality, speed and reasoning-probe logs |

## Credits

[DeepSeek](https://huggingface.co/deepseek-ai) for the model (MIT) ·
[Lucebox](https://huggingface.co/Lucebox/DeepSeek-V4.1-Flash-1017-ROCMFP-GGUF) for the eval set ·
[llama.cpp](https://github.com/ggml-org/llama.cpp) (Georgi Gerganov and the ggml authors, MIT) for the web UI.

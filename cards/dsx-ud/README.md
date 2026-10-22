---
license: mit
base_model:
- deepseek-ai/DeepSeek-V4.1-Flash-1017
- unsloth/DeepSeek-V4.1-Flash-1017-GGUF
base_model_relation: quantized
pipeline_tag: text-generation
library_name: dsx
tags:
- deepseek
- deepseek-v4.1
- moe
- quantized
- iq1_m
- consumer-gpu
- rtx-3090
---

# DeepSeek-V4.1-Flash-1017 · dsx + Unsloth UD-IQ1_M experts

DeepSeek-V4.1-Flash-1017 (552B MoE + 196B Engram) in one 303 GB download for two RTX 3090s and 128 GB of RAM. The
routed experts are [Unsloth's UD-IQ1_M](https://huggingface.co/unsloth/DeepSeek-V4.1-Flash-1017-GGUF) tensors,
unchanged. The dense weights (Q8_0), router and vision tower come from
[dsx's 2-bit build](https://huggingface.co/lxe/DeepSeek-V4.1-Flash-1017-dsx-2bit), and the Engram table is the official
FP8 table at Q6_K, inside the GGUF together with the tokenizer. Runs on [**dsx**](https://github.com/lxe/dsx), a Rust +
CUDA engine for this model.

**In short:** this build takes only the routed experts from Unsloth's UD-IQ1_M, byte for byte, and replaces everything
else: the attention, shared-expert and LM-head weights come from dsx at Q8_0 (Unsloth uses Q6_K / Q5_K), and Engram is
DeepSeek's FP8 table at Q6_K instead of IQ4_NL, so the same experts score slightly better (KL 0.269 vs 0.275).
dsx differs from llama.cpp in how it runs a model this size on a desktop: it keeps the experts each conversation uses
most in VRAM and swaps them as it changes, computes the rest on the CPU straight from RAM in parallel with the GPUs, has
the CPU and GPUs signal each other through mapped-memory flags instead of host synchronization within a token, and uses
decode kernels made for this model's attention and expert shapes (optionally letting the GPU run ahead of the CPU's
lowest-ranked experts). Same experts, same machine: 28.5 tok/s in dsx, 12.3 in llama.cpp.

| | |
|---|---|
| Size | 303 GB in 5 GGUF shards: the model (142 GB in 4) + the Engram table at Q6_K (161 GB); tokenizer included |
| Experts | Unsloth UD-IQ1_M: IQ1_S, IQ2_XXS, IQ2_XS and IQ2_S per tensor (~1.95 bpw) |
| Quality vs FP8 | KL 0.269 · top-1 78.9% · perplexity 10.13 (FP8: 8.56) |
| With `--defer-rank 3` | KL 0.284 · top-1 78.2% · perplexity 10.25 |
| Speed, 2x RTX 3090 | 27-29.5 tok/s decode, 32-36 with `--defer-rank 3` · ~370 tok/s prefill (3K prompt) |
| Runtime | [dsx](https://github.com/lxe/dsx) from 2026-10-21 or later |

## What differs from Unsloth's GGUF and from dsx 2-bit

| | **This build** | Unsloth UD-IQ1_M | dsx 2-bit v4b |
|---|---|---|---|
| Routed experts (~96% of weights) | **Unsloth's, byte for byte** | IQ1_S / IQ2_XXS / IQ2_XS / IQ2_S per tensor, ~1.95 bpw | dsx's GPTQ: ternary / Q2_0 / Q3_K per expert, 2.03 bpw, Hadamard-rotated |
| Attention (q, kv, output) | Q8_0 | Q6_K | Q8_0 |
| Shared expert | Q8_0 | Q5_K / Q8_0 | Q8_0 |
| LM head | Q8_0 | Q6_K | Q8_0 |
| Token embeddings | F16 | Q8_0 | F16 |
| Indexer, compressor | F16 | BF16 | F16 |
| Router, norms | F32 | F32 | F32 |
| Hyper-connection mixing | F16 | F32 | F16 |
| Engram table (196B params) | Q6_K from the official FP8, inside the GGUF (161 GB) | IQ4_NL, inside the GGUF (110.6 GB) | official FP8, read from DeepSeek's shards |
| Vision tower | included (bf16) | not included | included |
| Download | 303 GB | 250 GB | 148 GB + 203 GB |
| Engine | dsx | llama.cpp | dsx |
| KL / top-1 vs FP8 | **0.269 / 78.9%** | 0.275 / 78.8% | 0.373 / 74.4% |
| Decode, 2x RTX 3090, 3K prompt | **27-29.5 tok/s** (32-36 with `--defer-rank 3`) | 12.3 tok/s | 29.5 tok/s |

The only shared part is the routed experts. Everything else, and the engine, comes from dsx.

## Quality

Measured on [Lucebox's 1017 eval set](https://huggingface.co/Lucebox/DeepSeek-V4.1-Flash-1017-ROCMFP-GGUF):
8 × 1024 tokens against the FP8 model's top-256 log-probs.

| | Size without Engram | KL ↓ | Top-1 ↑ | Perplexity ↓ |
|---|---:|---:|---:|---:|
| **This build, dsx** | **142 GB** | **0.269** | **78.9%** | **10.13** |
| This build, dsx, `--defer-rank 3` | 142 GB | 0.284 | 78.2% | 10.25 |
| Unsloth UD-IQ1_M, llama.cpp | ~140 GB | 0.275 | 78.8% | 10.11 |
| dsx 2-bit v4b | 148 GB | 0.373 | 74.4% | 10.68 |

The experts are the same bytes in both engines. dsx scores slightly better than llama.cpp because it uses Q8_0 dense
weights and the Engram table at Q6_K, where Unsloth's GGUF has lower-bit dense weights and Engram in IQ4_NL.

The Engram table format, everything else the same (this build's experts and dense weights in dsx):

| Engram table | Size | KL ↓ | Top-1 ↑ | Perplexity ↓ |
|---|---:|---:|---:|---:|
| **Q6_K (this build)** | **161 GB** | **0.269** | **78.9%** | **10.13** |
| official FP8 | 203 GB | 0.270 | 79.0% | 10.13 |
| IQ4_NL (Unsloth's table) | 110.6 GB | 0.277 | 78.3% | 10.18 |
`--defer-rank 3` adds the CPU's lower-ranked experts one layer late (after KTransformers), for +14% decode speed.

## Speed

Greedy decoding, exact routing, 300 generated tokens, 2x RTX 3090 (230 W) + 128 GB DDR4-3200.

| | Decode, 3K-token prompt | Decode, 60-token prompt |
|---|---:|---:|
| This build | 27.0-29.5 tok/s | 27.0 tok/s |
| This build, `--defer-rank 3` | 33.4-36.0 tok/s | 32.0 tok/s |
| dsx 2-bit v4b | 29.5 tok/s | |
| Unsloth UD-IQ1_M, llama.cpp | 12.3 tok/s | |

## Run

```bash
hf download lxe/DeepSeek-V4.1-Flash-1017-dsx-ud --local-dir dsx-ud

git clone https://github.com/lxe/dsx && cd dsx
CUDA_ARCHS=86 cargo build --release -p dsx --features cuda      # 86 = RTX 30xx, 120 = RTX 50xx

./target/release/dsx-server --model ../dsx-ud/DeepSeek-V4.1-Flash-1017-dsx-ud-00001-of-00005.gguf \
  --host 0.0.0.0 --port 8090 --ctx 131072 --arena-gb 110 --rank ../dsx-ud/expert-rank.json --defer-rank 3
```

The Engram table and the tokenizer are read from the GGUF; no `--engram` or `--tokenizer` needed.

Leave out `--defer-rank 3` for the best quality. Open `http://localhost:8090/` for the web UI, or point any
OpenAI-compatible client at `http://localhost:8090/v1`.

## Hardware sizing

VRAM + RAM should hold ~135 GB of experts: 2x 24 GB GPUs and 128 GB of RAM with `--arena-gb 110`. The Engram table
stays on disk (NVMe) and is read a few rows per token. Also needed: CUDA 13, Rust, an AVX2 CPU, ~305 GB of disk.

## How it was made

`quant/hybrid_gguf.py` in [dsx](https://github.com/lxe/dsx) writes one GGUF with every tensor and key of dsx's v4b
build except its routed experts, plus the `blk.N.ffn_{gate,up,down}_exps.weight` tensors of Unsloth's UD-IQ1_M
(revision `60ad3651`). `quant/engram/engram_gguf.py` turns the official FP8 Engram tables into one Q6_K tensor
(`per_layer_token_embd.weight`, llama.cpp's layout), quantized by ggml's own Q6_K code from the bf16-rounded FP8 values.
`dsx split --shards 5 --add engram-q6k.gguf --tokenizer-json tokenizer.json` writes the model in 4 shards, the Engram
table as the 5th, and the official `tokenizer.json` as the `dsx.tokenizer_json` key. The experts are unrotated, so the
build has no `dsx.expert_rot` key. dsx runs the four IQ formats with its own CPU kernels and GPU tiles.

## Files

| File | |
|---|---|
| `DeepSeek-V4.1-Flash-1017-dsx-ud-0000N-of-00005.gguf` | the model in shards 1-4, the Engram table (Q6_K) in shard 5; dsx loads them from the first |
| `SHA256SUMS` | checksums of the shards |
| `expert-rank.json` | expert hotness, used to place experts in VRAM |
| `vision.safetensors` | vision tower, aligner and image embeddings from the official checkpoint (bf16) |
| `config.json` | build metadata |

## Credits

[Unsloth](https://huggingface.co/unsloth) for the UD-IQ1_M experts (MIT) ·
[DeepSeek](https://huggingface.co/deepseek-ai) for the model (MIT) ·
[llama.cpp / ggml](https://github.com/ggml-org/llama.cpp) for the IQ formats and codebook tables (MIT) ·
[Xolotl / Strata](https://github.com/Indras-Mirror/Xolotl) for the V4.1 GGUF format (MIT) ·
[Lucebox](https://huggingface.co/Lucebox/DeepSeek-V4.1-Flash-1017-ROCMFP-GGUF) for the eval set ·
KTransformers ([Chen et al., SOSP '25](https://madsys.cs.tsinghua.edu.cn/publication/ktransformers-unleashing-the-full-potential-of-cpu/gpu-hybrid-inference-for-moe-models/SOSP25-chen.pdf)) for expert deferral.

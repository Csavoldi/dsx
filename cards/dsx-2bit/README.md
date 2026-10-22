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
- 2-bit
- mixed-precision
- gptq
- consumer-gpu
- rtx-3090
- rtx-4090
- rtx-5090
---

# DeepSeek-V4.1-Flash-1017 · dsx 2-bit

DeepSeek-V4.1-Flash-1017 (552B MoE + 196B Engram) in 148 GB, built to run on two RTX 3090s and 128 GB of RAM with
exact routing. Runs on [**dsx**](https://github.com/lxe/dsx), a Rust + CUDA engine made for this model. For the best quality on the same machine, use
[dsx-ud](https://huggingface.co/lxe/DeepSeek-V4.1-Flash-1017-dsx-ud) (KL 0.269, one 305 GB download).

## Which build

| Build | Made for | Download | KL / top-1 vs FP8 | Decode |
|---|---|---|---:|---:|
| [**dsx-t16**](https://huggingface.co/lxe/DeepSeek-V4.1-Flash-1017-dsx-t16) (recommended) | 2x 24 GB NVIDIA GPU + 128 GB RAM | 302 GB, Engram and tokenizer inside | 0.194 / 81.4% | 34.9 tok/s (36.8 with `--defer-rank 3`) |
| [dsx-ud](https://huggingface.co/lxe/DeepSeek-V4.1-Flash-1017-dsx-ud) | 2x 24 GB NVIDIA GPU + 128 GB RAM | 305 GB, Engram and tokenizer inside | 0.272 / 79.0% | 27-29.5 tok/s (32-36 with `--defer-rank 3`) |
| [dsx-2bit](https://huggingface.co/lxe/DeepSeek-V4.1-Flash-1017-dsx-2bit) | 2x 24 GB NVIDIA GPU + 128 GB RAM | 148 GB + the official Engram shards (203 GB) | 0.373 / 74.4% | ~29 tok/s (35.7 with `--defer-rank 3`) |
| [dsx-mxfp4](https://huggingface.co/lxe/DeepSeek-V4.1-Flash-1017-dsx-mxfp4) | 1x AMD Instinct MI325X (256 GiB) | 270 GB + the official Engram shards (203 GB) | 0.049 / 90.6% | 110-114 tok/s |

All four run on [dsx](https://github.com/lxe/dsx) and were measured on the same eval
([Lucebox 1017](https://huggingface.co/Lucebox/DeepSeek-V4.1-Flash-1017-ROCMFP-GGUF), 8 x 1,024 tokens against the FP8
model's top-256 log-probs).

## At a glance

| | |
|---|---|
| Size | 148 GB (4 GGUF shards) + the official Engram shards (203 GB) |
| Format | per-expert mixed precision: ternary, Q2_0 or Q3_K for each expert, Q8_0 dense |
| Routing | exact: no expert pruning, no router bias |
| Quality vs FP8 | KL 0.373 · top-1 74.4% · perplexity 10.68 (FP8: 8.56) |
| Speed, 2x RTX 3090 | ~29 tok/s decode · 300-470 tok/s prefill |
| Hardware | 2x 24 GB NVIDIA GPU (RTX 3090 / 4090 / 5090), 128 GB RAM, AVX2 CPU, ~350 GB NVMe ([details](#hardware)) |
| Runtime | [dsx](https://github.com/lxe/dsx) (OpenAI / Anthropic API, image input, web UI) |

## Quality

![Quality against the FP8 original](assets/quality.png)

Measured on [Lucebox's 1017 eval set](https://huggingface.co/Lucebox/DeepSeek-V4.1-Flash-1017-ROCMFP-GGUF):
8 × 1024 tokens against the FP8 model's top-256 log-probs.

| | Experts | KL ↓ | Top-1 ↑ | Perplexity ↓ |
|---|---|---:|---:|---:|
| FP8 / FP4 original | FP4 | 0 | 100% | 8.56 |
| Unsloth UD-Q2_K_XL | ~224 GB without Engram | 0.114 | 85.7% | 9.04 |
| Lucebox ROCMFP | 2.5 bpw | 0.253 | 78.7% | 9.78 |
| Unsloth UD-IQ1_S | ~122 GB without Engram | 0.361 | 75.1% | 10.76 |
| **dsx 2-bit** | **2.03 bpw, 148 GB** | **0.373** | **74.4%** | **10.68** |

Unsloth's GGUFs were measured in llama.cpp with the same token ids and metric
([tool](https://github.com/lxe/dsx/tree/main/amd/tools/lucebox-eval)). They also carry the Engram table, quantized to
IQ4_NL (110.6 GB); dsx reads it unquantized from the official shards.

## Reasoning

Calibrated on the full model's own reasoning traces as well as agent text. Reasoning tokens per run on two
Omni-MATH problems, temperature 1.0, reasoning effort low, all answers correct:

| | Problem A | Problem B |
|---|---|---|
| FP8 model, hosted (8 runs) | 5.7K-17.4K (median 9.4K) | 7.5K-13.9K (median 10.7K) |
| dsx 2-bit | 7.8K, 12.8K, 14.9K | 14.4K, 16.7K, 18.2K |
| Unsloth UD-IQ1_S, llama.cpp | 10.6K, 13.1K, 19.9K | 21.2K, 29.8K, 33.9K |
| Unsloth UD-IQ1_M, llama.cpp | 11.0K, 13.3K | 20.3K, 20.3K |

On harder problems it reasons longer than the FP8 model, up to about 2×. Sample at temperature 1.0 for reasoning.

## Speed

![dsx speed on 2x RTX 3090 and RTX 5090](assets/speed.png)

Greedy decoding, exact routing, 400 generated tokens. The RTX 3090 numbers come from build v4, which has the same
expert layout as this build (v4b); the RTX 5090 numbers come from the older build v2.

| | 2x RTX 3090, 128 GB DDR4 | RTX 5090, 128 GB DDR5 |
|---|---:|---:|
| Decode, prose | 28.9 tok/s | 12.0 tok/s |
| Decode, 3K-token agent prompt | 28.7 tok/s | 15.2 tok/s |
| Prefill, 3K / 8K tokens | 301 / 470 tok/s | 93 / 172 tok/s |

On the same 2x RTX 3090 machine, Unsloth's UD-IQ1_M in llama.cpp (PR #30817) runs at 12.0 tok/s decode and 75-109 tok/s
prefill on the 3K-token agent prompt.

Decode speed comes down to how many experts fit in VRAM + RAM. Two 24 GB GPUs + 128 GB hold all of them; one 32 GB GPU
+ 128 GB leaves ~40 GB of experts to be read from disk.

## Size

![Download size of DeepSeek-V4.1-Flash-1017 builds](assets/size.png)

![Size against perplexity](assets/size-vs-quality.png)

## Run

**This build needs dsx from 2026-10-19 or later.** Older builds do not know the rotation and produce garbage with it: pull and rebuild dsx first.

```bash
# the quant: 4 GGUF shards, loaded as one model
hf download lxe/DeepSeek-V4.1-Flash-1017-dsx-2bit --local-dir dsx-2bit

# Engram tables + tokenizer from the official checkpoint
hf download deepseek-ai/DeepSeek-V4.1-Flash-1017 --local-dir v41-1017 \
  --include model.safetensors.index.json model-00047-of-00048.safetensors model-00048-of-00048.safetensors tokenizer.json

# the engine (dsx's NVIDIA runner)
git clone https://github.com/lxe/dsx && cd dsx
CUDA_ARCHS=86 cargo build --release -p dsx --features cuda      # 86 = RTX 30xx, 120 = RTX 50xx

./target/release/dsx-server --model ../dsx-2bit/DeepSeek-V4.1-Flash-1017-dsx-2bit-00001-of-00004.gguf \
  --engram ../v41-1017 --tokenizer ../v41-1017/tokenizer.json --host 0.0.0.0 --port 8090 \
  --ctx 131072 --arena-gb 110 --rank ../dsx-2bit/expert-rank.json
```

Open `http://localhost:8090/` for the web UI, or point any OpenAI-compatible client at `http://localhost:8090/v1`:

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8090/v1", api_key="none")
r = client.chat.completions.create(model="deepseek-v4.1-flash", reasoning_effort="none",
                                   messages=[{"role": "user", "content": "Merge overlapping intervals in Python."}])
print(r.choices[0].message.content)
```

Thinking is on by default; `"reasoning_effort": "none"` turns it off. Images go in as data URLs (`image_url`), or as
base64 `image` blocks on the Anthropic endpoint; the web UI takes uploads.

## Hardware

| | |
|---|---|
| GPU | NVIDIA, 24 GB of VRAM or more; two 24 GB cards measured |
| RAM | 128 GB |
| CPU | x86 with AVX2 (AVX-VNNI or AVX-512 VNNI preferred) |
| Disk | ~350 GB (148 GB + the official Engram shards, 203 GB), NVMe |
| Software | CUDA 13, Rust |

| GPU | `CUDA_ARCHS` | Status |
|---|---|---|
| RTX 3090 | 86 | runs (the numbers on this card) |
| RTX 4090 | 89 | runs |
| RTX 5090 | 120 | runs |
| A100 | 80 | builds, not run yet |
| AMD | | not supported by this build; see [dsx-mxfp4](https://huggingface.co/lxe/DeepSeek-V4.1-Flash-1017-dsx-mxfp4) for the MI325X |

VRAM + RAM should hold ~140 GB of experts:

| Machine | Flags |
|---|---|
| 2x 24 GB GPU, 128 GB RAM | `--arena-gb 110` (GPUs are used automatically) |
| 1x 32 GB GPU, 128 GB RAM | `--main-gb=-2.0 --arena-gb 80`, env `DSX_ADAPT=0` |

## How it was made

- **Experts:** each of the 15,360 routed experts gets ternary (1.75 bpw), Q2_0 (2.25) or Q3_K (3.44), picked by its
  measured output error. One bit budget is shared across layers 2-36, so sensitive layers get up to 2.75 bpw and
  forgiving ones ~1.76. Average: 2.03 bpw.
- **Method:** GPTQ per expert, with Hessians from a layer-by-layer calibration pass through DeepSeek's reference
  `model.py` (GSQ pipeline from [Xolotl](https://github.com/Indras-Mirror/Xolotl)).
- **Rotation:** every routed expert is quantized in a Hadamard-rotated basis (128-wide blocks along each input
  dimension), which spreads outlier weights before rounding; dsx rotates the expert inputs to match at run time
  (GGUF key `dsx.expert_rot`). Same size as without it, KL 0.397 → 0.382.
- **Scale refit:** after GPTQ picks each expert's codes, its group scales are refit by least squares against the full
  calibration Hessian, GPTQ runs again from the refit scales, and a final refit follows. KL 0.382 → 0.373.
- **Calibration text:** 160 × 2048 tokens: coding-agent sessions, antirez's agent and code prompts, and 53 sequences
  of the full model's own reasoning traces on Omni-MATH and SuperGPQA questions (none from the evaluated rows).
- **Dense weights:** Q8_0. **Engram:** unquantized, read from the official shards.

Pipeline: [github.com/lxe/dsx/tree/main/quant](https://github.com/lxe/dsx/tree/main/quant).

## Files

| File | |
|---|---|
| `DeepSeek-V4.1-Flash-1017-dsx-2bit-0000N-of-00004.gguf` | the model in 4 GGUF shards; dsx loads them from the first |
| `SHA256SUMS` | checksums of the shards |
| `expert-rank.json` | expert hotness, used to place experts in VRAM |
| `vision.safetensors` | the vision tower, aligner and image embeddings from the official checkpoint (bf16, 0.97 GB); dsx loads it from beside the shards (`--no-vision` for text only) |
| `config.json` | build metadata: base model, quant tiers, files |
| `results/` | raw quality and speed logs |
| `assets/` | charts and the script that draws them |

This is build v4b. Earlier builds: revision [`v4`](https://huggingface.co/lxe/DeepSeek-V4.1-Flash-1017-dsx-2bit/tree/v4) (rotation without the refit, KL 0.382), [`v3a`](https://huggingface.co/lxe/DeepSeek-V4.1-Flash-1017-dsx-2bit/tree/v3a) (reasoning-trace calibration without rotation, KL 0.397), [`v2`](https://huggingface.co/lxe/DeepSeek-V4.1-Flash-1017-dsx-2bit/tree/v2) (agent-text calibration, KL 0.395) and [`v1`](https://huggingface.co/lxe/DeepSeek-V4.1-Flash-1017-dsx-2bit/tree/v1) (KL 0.462).

## Credits

[DeepSeek](https://huggingface.co/deepseek-ai) for the model (MIT) ·
[Xolotl / Strata](https://github.com/Indras-Mirror/Xolotl) for the GSQ tools and V4.1 GGUF format (MIT) ·
DeepSeek for the reference `model.py` the calibration runs · GPTQ ([Frantar et al.](https://arxiv.org/abs/2210.17323)) ·
Hadamard rotation as in QuIP# ([Tseng et al.](https://arxiv.org/abs/2402.04396)) and QuaRot ([Ashkboos et al.](https://arxiv.org/abs/2404.00456)) ·
[Lucebox](https://huggingface.co/Lucebox/DeepSeek-V4.1-Flash-1017-ROCMFP-GGUF) for the eval set ·
antirez for the calibration prompts ·
[llama.cpp](https://github.com/ggml-org/llama.cpp) (Georgi Gerganov and the ggml authors, MIT) for the web UI.

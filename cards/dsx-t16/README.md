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
- trellis
- exl3
- consumer-gpu
- rtx-3090
- rtx-4090
- rtx-5090
---

# DeepSeek-V4.1-Flash-1017 · dsx T16

DeepSeek-V4.1-Flash-1017 (552B MoE + 196B Engram) in one 252 GB download for two RTX 3090s and 128 GB of RAM, with the
routed experts in **T16**, dsx's own trellis format: exllamav3's bitshift trellis with a codebook made for desktop
CPUs, quantized from DeepSeek's original weights. Runs on [**dsx**](https://github.com/lxe/dsx), a Rust + CUDA engine
for this model.

**In short:** at the same size as dsx-ud, T16's experts give 28% lower KL against the FP8 model (0.194 vs 0.272), about
the quality of Project Maya's 178 GB Maya-S in 141 GB, and they decode faster on both the CPU and the GPU.

## Which build

| Build | Made for | Download | KL / top-1 vs FP8 | Decode |
|---|---|---|---:|---:|
| [**dsx-t16**](https://huggingface.co/lxe/DeepSeek-V4.1-Flash-1017-dsx-t16) (this, v1, recommended) | 2x 24 GB NVIDIA GPU + 128 GB RAM | 252 GB, Engram and tokenizer inside | 0.196 / 80.9% | 34.9 tok/s (36.8 with `--defer-rank 3`) |
| [dsx-ud](https://huggingface.co/lxe/DeepSeek-V4.1-Flash-1017-dsx-ud) | 2x 24 GB NVIDIA GPU + 128 GB RAM | 305 GB, Engram and tokenizer inside | 0.272 / 79.0% | 27-29.5 tok/s (32-36 with `--defer-rank 3`) |
| [dsx-2bit](https://huggingface.co/lxe/DeepSeek-V4.1-Flash-1017-dsx-2bit) | 2x 24 GB NVIDIA GPU + 128 GB RAM | 148 GB + the official Engram shards (203 GB) | 0.373 / 74.4% | ~29 tok/s (35.7 with `--defer-rank 3`) |
| [dsx-mxfp4](https://huggingface.co/lxe/DeepSeek-V4.1-Flash-1017-dsx-mxfp4) | 1x AMD Instinct MI325X (256 GiB) | 270 GB + the official Engram shards (203 GB) | 0.049 / 90.6% | 110-114 tok/s |

All run on [dsx](https://github.com/lxe/dsx) and were measured on the same eval
([Lucebox 1017](https://huggingface.co/Lucebox/DeepSeek-V4.1-Flash-1017-ROCMFP-GGUF), 8 x 1,024 tokens against the FP8
model's top-256 log-probs).

## At a glance

| | |
|---|---|
| Size | 252 GB in one GGUF file: the model (141 GB) and the Engram table at IQ4_NL (111 GB); tokenizer included. With a Q6_K Engram table (302 GB, slightly better): branch [`q6k-engram`](https://huggingface.co/lxe/DeepSeek-V4.1-Flash-1017-dsx-t16/tree/q6k-engram) |
| Experts | T16 trellis: 1.5 / 1.5 / 2 bits (gate / up / down) in layers 0-19, 2 / 2 / 2.5 in layers 20-39; ~1.9 bpw |
| Quality vs FP8 | KL 0.196 · top-1 80.9% · perplexity 9.54 (FP8: 8.56); with the Q6_K Engram table: 0.194 · 81.4% · 9.60 |
| Speed, 2x RTX 3090 | 34.9 tok/s decode, 36.8 with `--defer-rank 3` · ~530 tok/s prefill (3K prompt) |
| Hardware | 2x 24 GB NVIDIA GPU (RTX 3090 / 4090 / 5090), 128 GB RAM, AVX2 CPU (AVX-VNNI for the fast CPU kernel), ~305 GB NVMe ([details](#hardware)) |
| Runtime | [dsx](https://github.com/lxe/dsx) from 2026-10-22 or later (T16 support) |

## Quality

Measured on [Lucebox's 1017 eval set](https://huggingface.co/Lucebox/DeepSeek-V4.1-Flash-1017-ROCMFP-GGUF):
8 x 1024 tokens against the FP8 model's top-256 log-probs. T16 and dsx-ud were measured in the same session and build.

| | Size without Engram | KL ↓ | Top-1 ↑ | Perplexity ↓ |
|---|---:|---:|---:|---:|
| **This build (T16 v1, IQ4_NL Engram), dsx** | **141 GB** | **0.196** | **80.9%** | **9.54** |
| This build with the Q6_K Engram table (branch `q6k-engram`) | 141 GB | 0.194 | 81.4% | 9.60 |
| dsx-ud (Unsloth UD-IQ1_M's experts), dsx | 142 GB | 0.272 | 79.0% | 10.17 |
| Project Maya Maya-S, llama.cpp | 178 GB | 0.195 | 82.0% | 9.49 |
| Unsloth UD-IQ1_M as shipped, llama.cpp | ~140 GB | 0.275 | 78.8% | 10.11 |

Through the decode path (one token at a time, as in generation; 2 x 1,023 tokens of the same set):

| | KL ↓ | Top-1 ↑ | Perplexity ↓ |
|---|---:|---:|---:|
| **This build** | **0.205** | **82.0%** | **8.65** |
| dsx-ud | 0.303 | 80.4% | 9.03 |

## How T16 works

The experts are quantized with exllamav3's machinery (a QTIP-style bitshift trellis found by a Viterbi search, 128-wide
Hadamard rotations, LDLQ error feedback with each expert's input Hessian), with five changes:

1. **The T16 codebook.** A weight's value comes from its 16-bit window of the bit stream: multiply by 0x3A2F (16 bits)
   and add the product's two bytes. That is one `vpmullw` and one `vpdpbusd` per pair of weights on an AVX-VNNI CPU, where
   exllamav3's codebook needs a 32-bit multiply per weight. ~4% more error than exllamav3's on random data.
2. **Column-major positions inside each 16 x 16 tile**, so one file decodes fast on both the CPU and the GPU.
3. **A damped Hessian** (sigma_reg 2). With the full Hessian the experts fit the calibration text but lose 1.6-2.4x of
   their accuracy on new text; damping keeps the fit general.
4. **An output-gain correction.** A trellis quantizer that minimizes squared error reconstructs weights about 13%
   too small at 1.5 bits (7% at 2 bits): shrinking cuts the noise more than it adds bias. Per matrix that is optimal, but
   the bias is the same in every expert and layer, so it compounds through the network. Each output's stored scale is
   multiplied back (x1.13 for 1.5-bit gate/up, x1.04 for 2-bit gate/up); on a half-T16 model that took KL from 0.31 to
   0.25. It costs nothing at run time. The two values were chosen on the eval set above.
5. **The bit layout** above, sized to match Unsloth's UD-IQ1_M.

Calibration: 635K tokens (coding-agent sessions, agent and code prompts, the model's own reasoning traces, Wikipedia in
five languages, code, chat). A v2 on a broader 1.2M-token set is being quantized now.

## Speed

2x RTX 3090 (250 / 280 W) + i7-12700K + 128 GB DDR4, dsx-ud measured in the same session:

| | This build | dsx-ud |
|---|---:|---:|
| Decode, the same 2,046 tokens (teacher-forced eval) | **215 s** | 253 s |
| Decode, 3K-token prompt + 300 tokens | **34.9 tok/s** | 27.9-28.0 tok/s |
| Decode with `--defer-rank 3` | **36.8 tok/s** | 36.1-36.2 tok/s |
| Prefill, 3K-token prompt | **529 tok/s** (dsx 2026-10-22; 361-367 before) | 393-409 tok/s (earlier dsx) |

The free-running rows generate different text per build, which moves the share of experts the CPU computes; the
teacher-forced row (identical tokens) is the fair comparison: 15% faster. Prefill was ~9% slower at first; dsx since
expands T16 weights 2.5-3x faster (899e6e4) and keeps the RAM arena in hugepages, which copy to the GPU at 24.6 GB/s
instead of 13.5 (cc88c1a): 3K-token prefill 388 → 529 tok/s on the same build otherwise. dsx-ud was not re-measured.

Per layer, T16's experts compute 20-34% faster than Unsloth's codebook mix on the i7-12700K (14 threads: 1.21 / 1.46 ms
per 6-expert call vs 1.83) and 22% faster on an RTX 3090 (125 vs 161 us per layer's 6 experts).

## Run

```bash
hf download lxe/DeepSeek-V4.1-Flash-1017-dsx-t16 --local-dir dsx-t16

git clone https://github.com/lxe/dsx && cd dsx
CUDA_ARCHS=86 cargo build --release -p dsx --features cuda      # 86 = RTX 30xx, 120 = RTX 50xx

./target/release/dsx-server --model ../dsx-t16/DeepSeek-V4.1-Flash-1017-dsx-t16.gguf \
  --host 0.0.0.0 --port 8090 --ctx 131072 --arena-gb 110 --rank ../dsx-t16/expert-rank.json --defer-rank 3
```

**The Q6_K Engram version** (302 GB, KL 0.194 / top-1 81.4% instead of 0.196 / 80.9%) is on the branch `q6k-engram`:

```bash
hf download lxe/DeepSeek-V4.1-Flash-1017-dsx-t16 --revision q6k-engram --local-dir dsx-t16-q6k
# then --model dsx-t16-q6k/DeepSeek-V4.1-Flash-1017-dsx-t16-00001-of-00004.gguf (4 shards; dsx loads the rest)
```

The Engram table and the tokenizer are read from the GGUF. Leave out `--defer-rank 3` for the best quality. Open
`http://localhost:8090/` for the web UI, or point any OpenAI-compatible client at `http://localhost:8090/v1`.

## Hardware

| | |
|---|---|
| GPU | NVIDIA, 24 GB of VRAM or more; two 24 GB cards measured |
| RAM | 128 GB |
| CPU | x86 with AVX2; the fast T16 kernel needs AVX-VNNI or AVX-512 VNNI (others use a scalar decode) |
| Disk | ~255 GB, NVMe |
| Software | CUDA 13, Rust |

| GPU | `CUDA_ARCHS` | Status |
|---|---|---|
| RTX 3090 | 86 | runs (the numbers on this card) |
| RTX 4090 | 89 | runs |
| RTX 5090 | 120 | runs |
| A100 | 80 | builds, not run yet |

## Files

| File | |
|---|---|
| `DeepSeek-V4.1-Flash-1017-dsx-t16.gguf` | the model (T16 experts, Q8_0 dense) and the Engram table (IQ4_NL, from Unsloth's UD-IQ1_M GGUF), one file |
| `SHA256SUMS` | checksum of the GGUF |
| `expert-rank.json` | expert hotness, used to place experts in VRAM |
| `vision.safetensors` | vision tower, aligner and image embeddings from the official checkpoint (bf16) |
| `config.json` | build metadata |

## Credits

[DeepSeek](https://huggingface.co/deepseek-ai) for the model (MIT) · [exllamav3](https://github.com/turboderp-org/exllamav3)
(turboderp, MIT) for the trellis quantizer that T16 patches · the T16 format: [docs/t16.md](https://github.com/lxe/dsx/blob/main/docs/t16.md) · QTIP ([Tseng et al.](https://arxiv.org/abs/2406.11235))
for bitshift trellises · [llama.cpp / ggml](https://github.com/ggml-org/llama.cpp) for the Q6_K and IQ4_NL formats (MIT) · [Unsloth](https://huggingface.co/unsloth/DeepSeek-V4.1-Flash-1017-GGUF) for the IQ4_NL Engram table ·
[Xolotl / Strata](https://github.com/Indras-Mirror/Xolotl) for the V4.1 GGUF format (MIT) ·
[Lucebox](https://huggingface.co/Lucebox/DeepSeek-V4.1-Flash-1017-ROCMFP-GGUF) for the eval set.

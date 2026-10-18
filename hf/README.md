---
license: mit
base_model:
- deepseek-ai/DeepSeek-V4.1-Flash-1017
base_model_relation: quantized
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
---

# DeepSeek-V4.1-Flash-1017 · ds41 2-bit (148 GB)

**DeepSeek's 552B-parameter MoE (+196B Engram) on a desktop: 2x RTX 3090, 128 GB DDR4, exact routing.
31.5 tok/s decode, 340 tok/s prefill.**

A 2.03 bits/weight quant of [DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017)
calibrated on agentic coding text, and the engine it was built for: **[github.com/lxe/ds41](https://github.com/lxe/ds41)**
(Rust + CUDA, OpenAI / Anthropic compatible server).

> **Research release, v1.** Measured on one machine. This is **not** a llama.cpp / Ollama / LM Studio GGUF: llama.cpp
> has no V4.1 architecture and none of the per-expert quant types used here. Run it with
> [ds41](https://github.com/lxe/ds41). It uses the same format as [Xolotl](https://github.com/Indras-Mirror/Xolotl)'s V4.1
> GSQ release, so Xolotl should load it too (not tested). **v2 is in progress**: better quality at the same speed, faster
> serving ([discussion](https://huggingface.co/lxe/DeepSeek-V4.1-Flash-1017-ds41-2bit/discussions)).

## Results

Machine: i7-12700K, 128 GB DDR4-3200 (40 GB/s measured), RTX 3090 on PCIe 4.0 x16 + RTX 3090 on PCIe 3.0 x4, NVMe.
Placement: the hottest 3,052 of 15,360 experts in VRAM (34 GB across both GPUs), the other 12,258 in a 104 GB RAM arena,
nothing read from disk per token. Exact routing (no routing bias), greedy, thinking off, unless noted.

### Speed

| | |
|---|---:|
| Decode, 3000-token agent prompt (`ds41 gen`) | **31.5 tok/s** (31.7 ms/token) |
| Decode through the API server (temperature sampling, SSE streaming) | 19-23 tok/s (server overhead, fixed in v2) |
| Prefill, 3000 tokens in one 4096 chunk | 340 tok/s |
| Prefill of a 2,000-token agent system prompt (server) | 7.8 s, then reused across turns |
| Verify pass, 6 tokens (for speculative decoding) | 157 ms (26 ms/token if all accepted) |

Where a decode step's 32 ms go: dense half on GPU0 ~14 ms (projections, hyper-connections, Engram), waiting for the
CPU's experts ~12 ms (14 threads, AVX-VNNI, from RAM), shared expert + router + GPU experts ~5 ms. GPU1's experts run in
parallel and add ~0.

Engine history, same quant and prompt:

| Change | Decode |
|---|---:|
| First full run | 12.5 tok/s |
| One CPU per spinning thread (the CPU expert worker shared a core with the GPU launch thread) | 20.3 tok/s |
| Tiled split-K decode attention (276 -> 21 us per layer) | **31.5 tok/s** |

### Quality

[Lucebox's 1017 eval](https://huggingface.co/Lucebox/DeepSeek-V4.1-Flash-1017-ROCMFP-GGUF): 8 sequences x 1024 tokens,
KL divergence against the FP8 model's top-256 log-probs, top-1 agreement, perplexity.

| | bits/weight (experts) | KL | top-1 | perplexity |
|---|---:|---:|---:|---:|
| FP8 / FP4 original (teacher) | FP4 | 0 | 100% | 8.56 |
| Lucebox ROCMFP fp2s | 2.5 | 0.253 | 78.7% | 9.78 |
| **This quant** | **2.03** | **0.462** | **71.5%** | **11.66** |
| This quant, routing bias 0.05 | 2.03 | 0.464 | 71.4% | 11.71 |
| This quant, routing bias 0.1 | 2.03 | 0.478 | 70.6% | 11.97 |
| This quant, routing bias 0.2 | 2.03 | 0.526 | 69.5% | 12.53 |

Decode path (the same sequences fed token by token, 2 x 1024 tokens): KL 0.461, top-1 73.0%, perplexity 10.09.

Routing bias (nudging the router toward experts already in VRAM) is off by default: at 0.2 it bought ~4 tok/s and cost
2 points of top-1. The gap to the 2.5 bpw quant is the price of fitting in 128 GB RAM + 48 GB VRAM without touching
the disk; v2 moves the coldest experts (the coldest 10% carry ~1.3% of the traffic) to NVMe to afford ~2.3-2.5 bpw.

### Expert usage on agentic text

Tokens routed per expert in the calibration pass (layers 2-6): the hottest 10% of experts take 32-41% of routed tokens,
the hottest 25% take 55-61%, the hottest 50% ~80%, the coldest 10% 1.0-1.7%. `expert-rank.json` is that ranking
(`[layer, expert, tokens]`, hottest first); the engine fills VRAM in this order.

## The quant

| | |
|---|---|
| Routed experts | per-expert tiers: PTQ1_0 (ternary, 1.75 bpw), Q2_0 (2.25), Q3_K (3.44); 2.0 bpw budget per layer, layers 0, 1, 37-39 at 2.25; average 2.03 |
| Method | GPTQ per expert with Hessians from a layer-streaming calibration forward of DeepSeek's `model.py` on the FP8/FP4 checkpoint; tiers allocated by measured held-out output error per byte (GSQ pipeline from Xolotl) |
| Calibration text | 128 x 2048 tokens: the author's coding-agent sessions + antirez's agent/code prompts |
| Dense weights | Q8_0 (attention, shared expert, routers, hyper-connections, head) |
| Engram | not in this file: read from the official checkpoint's last two shards at run time |
| Size | 148 GB (4 parts) |

Pipeline and patches: [github.com/lxe/ds41/tree/main/quant](https://github.com/lxe/ds41/tree/main/quant).

## Files

| File | |
|---|---|
| `DeepSeek-V4.1-Flash-1017-ds41-2bit.gguf.part-0N-of-04` | the GGUF as plain byte ranges |
| `SHA256SUMS` | checksums of every part and of the joined file |
| `reassemble.sh` | checks the parts, joins them, verifies the result |
| `expert-rank.json` | expert hotness ranking for placement |
| `results/` | raw speed and quality outputs |

## Run it

You need ~140 GB of RAM + VRAM for the experts (less works, with the rest read from NVMe), an NVIDIA GPU with 24 GB+,
CUDA 13, Rust, and ~350 GB of disk.

```bash
pip install -U huggingface_hub
hf download lxe/DeepSeek-V4.1-Flash-1017-ds41-2bit --local-dir ds41-2bit
bash ds41-2bit/reassemble.sh ds41-2bit                    # add --delete-parts if short on disk

# Engram tables + tokenizer: only these files from the official checkpoint (203 GB)
hf download deepseek-ai/DeepSeek-V4.1-Flash-1017 --local-dir v41-1017 \
  --include model.safetensors.index.json model-00047-of-00048.safetensors model-00048-of-00048.safetensors tokenizer.json

git clone https://github.com/lxe/ds41 && cd ds41/engine
CUDA_ARCHS=86 cargo build --release --features cuda       # 86 = RTX 30xx, 120 = RTX 50xx

./target/release/ds41-server --model ../../ds41-2bit/DeepSeek-V4.1-Flash-1017-ds41-2bit.gguf \
  --engram ../../v41-1017 --tokenizer ../../v41-1017/tokenizer.json --host 0.0.0.0 --port 8090 \
  --ctx 131072 --gpu0-gb=-1.5 --gpu1-gb=-1.0 --arena-gb 104 --rank ../../ds41-2bit/expert-rank.json
```

Placement: `--gpu0-gb` / `--gpu1-gb` = GB of experts per GPU (negative: all free VRAM minus that), `--gpu1-gb 0` for one
GPU, `--arena-gb` = RAM arena. Then any OpenAI-compatible client:

```bash
curl http://localhost:8090/v1/chat/completions -H 'content-type: application/json' -d '{
  "model": "deepseek-v4.1-flash", "stream": true, "reasoning_effort": "none",
  "messages": [{"role": "user", "content": "Write a Python function that merges overlapping intervals."}] }'
```

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8090/v1", api_key="none")
r = client.chat.completions.create(model="deepseek-v4.1-flash", reasoning_effort="none",
                                   messages=[{"role": "user", "content": "Explain MoE routing in two sentences."}])
print(r.choices[0].message.content)
```

The server also speaks Anthropic `/v1/messages` and OpenAI `/v1/responses`, with tool calls and streaming. Thinking is on
by default; `"reasoning_effort": "none"` or `"thinking": {"type": "disabled"}` turns it off.

## Credits

[DeepSeek](https://huggingface.co/deepseek-ai) for the model (MIT). [Xolotl / Strata](https://github.com/Indras-Mirror/Xolotl)
(Niko1221 and contributors, MIT) for the GSQ quantization tools and the V4.1 GGUF format.
[Lucebox](https://huggingface.co/Lucebox/DeepSeek-V4.1-Flash-1017-ROCMFP-GGUF) for the 1017 eval set. antirez for the
agentic prompts.

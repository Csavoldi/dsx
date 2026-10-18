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

# DeepSeek-V4.1-Flash-1017 · ds41 2-bit (148 GB) · v2

**DeepSeek's 552B-parameter MoE (+196B Engram) on a desktop: 2x RTX 3090, 128 GB DDR4, exact routing.
30-37 tok/s decode, 330-510 tok/s prefill. v2: KL 0.395 / top-1 74.1% against the FP8 model at the same size as v1.**

A 2.03 bits/weight quant of [DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017)
calibrated on agentic coding text, and the engine it was built for: **[github.com/lxe/ds41](https://github.com/lxe/ds41)**
(Rust + CUDA, OpenAI / Anthropic compatible server).

> **v2 (2026-10-18)** replaces v1 at the same size: the bit budget is now shared across layers (fragile layers get
> up to 2.75 bpw, forgiving decoder layers ~1.76), KL 0.462 -> 0.395, top-1 71.5% -> 74.1%. v1 stays available at
> revision [`v1`](https://huggingface.co/lxe/DeepSeek-V4.1-Flash-1017-ds41-2bit/tree/v1).
>
> **Research release.** Measured on one machine. This is **not** a llama.cpp / Ollama / LM Studio GGUF: llama.cpp
> has no V4.1 architecture and none of the per-expert quant types used here. Run it with
> [ds41](https://github.com/lxe/ds41). It uses the same format as [Xolotl](https://github.com/Indras-Mirror/Xolotl)'s V4.1
> GSQ release, so Xolotl should load it too (not tested). **v2 is in progress**: better quality at the same speed, faster
> serving ([discussion](https://huggingface.co/lxe/DeepSeek-V4.1-Flash-1017-ds41-2bit/discussions)).

## Results

### Quality

[Lucebox's 1017 eval](https://huggingface.co/Lucebox/DeepSeek-V4.1-Flash-1017-ROCMFP-GGUF): 8 sequences x 1024 tokens,
KL divergence against the FP8 model's top-256 log-probs, top-1 agreement, perplexity. Lower KL / perplexity is better.

| | size | experts, bits/weight | KL | top-1 | perplexity |
|---|---:|---|---:|---:|---:|
| FP8 / FP4 original (teacher) | 510 GB | FP4 | 0 | 100% | 8.56 |
| Lucebox ROCMFP fp2s | | 2.5 | 0.253 | 78.7% | 9.78 |
| **This quant, v2** | **148 GB** | **2.03, shared across layers** | **0.395** | **74.1%** | **10.93** |
| This quant, v1 | 148 GB | 2.03, 2.0 per layer | 0.462 | 71.5% | 11.66 |
| v2 alternative: per-layer, no ternary cap | 148 GB | 2.03, 2.0 per layer | 0.463 | 71.8% | 11.93 |

Routing bias (nudging the router toward experts already in VRAM) is off: on v1, at 0.2 it bought ~4 tok/s and cost 2
points of top-1. Every number here is exact routing.

### Speed, 2x RTX 3090 (the target machine)

i7-12700K, 128 GB DDR4-3200 (40 GB/s), RTX 3090 on PCIe 4.0 x16 + RTX 3090 on PCIe 3.0 x4. Placement: the hottest
~3,300 experts in VRAM (34 GB across both GPUs; GPU0's set adapts to the conversation), the rest in a 110 GB RAM arena,
the coldest ~530 read from the file when routed. Greedy, thinking off, 400 decoded tokens.

| | v2 | v1 |
|---|---:|---:|
| Decode, short prose prompt | **37.2 tok/s** | 29.9 tok/s |
| Decode, 3000-token agent prompt | **30.0 tok/s** | 31.2 tok/s |
| Prefill, 3000 tokens | 334 tok/s | 340 tok/s |
| Prefill, 8000 tokens | **508 tok/s** | 478 tok/s |

Decode speed depends on the text: the experts the CPU has to compute from RAM are the slow part, and how many a token
needs depends on what it is about. v2's forgiving decoder layers are mostly ternary, so their experts are smaller and
~200 more fit in VRAM (11% of prose picks fall to the CPU on v2). Prefill streams the RAM experts to GPU0 over PCIe
for every 4K chunk; above a few hundred tokens nearly every expert is used, so long prompts amortize it best.

Engine history (v1, 3000-token agent prompt): 12.5 tok/s first run -> 20.3 (one CPU per spinning thread) -> 31.5
(tiled split-K decode attention); adaptive expert residency then took short prose from 26 to 30.

### Speed, RTX 5090 + 128 GB DDR5 (a desktop)

i7-13700, 128 GB DDR5-5200 (72 GB/s), one RTX 5090 32 GB, the author's workstation with the desktop's heavy apps
closed (~90 GB of RAM free). 128 GB is not enough for this quant: 17 GB of experts fit in VRAM and 80 GB in the RAM
arena, so ~5,000 experts (~40 GB) are read from the NVMe drive through the page cache when routed.

| | static placement, 80 GB arena | adaptive placement, 75 GB arena |
|---|---:|---:|
| Decode, short prose prompt | **12.0 tok/s** | 7.6 tok/s |
| Decode, 3000-token agent prompt | **15.2 tok/s** | 7.8 tok/s |
| Prefill, 3000 tokens | 93 tok/s | 68 tok/s |
| Prefill, 8000 tokens | 172 tok/s | 114 tok/s |

Here the disk, not the GPU, sets the speed: half of the routed experts live outside RAM + VRAM. Use static placement
on such a machine (`DS41_ADAPT=0`): the adaptive cache keeps RAM copies of GPU0's experts, which a RAM-starved box
cannot spare. With ~160 GB of RAM (or a second GPU) the whole quant fits and the faster RAM and GPU should beat the
3090 box; ds41 does not yet batch its disk reads the way an SSD-first engine does.

### Expert usage

Tokens routed per expert in the calibration pass (layers 2-6): the hottest 10% of experts take 32-41% of routed tokens,
the hottest 25% take 55-61%, the hottest 50% ~80%, the coldest 10% 1.0-1.7%. `expert-rank.json` is that ranking
(`[layer, expert, tokens]`, hottest first); the engine fills VRAM in this order, then adapts GPU0's set to the
conversation (up to 8 expert swaps per token).

## The quant

| | |
|---|---|
| Routed experts | per-expert tiers: PTQ1_0 (ternary, 1.75 bpw), Q2_0 (2.25), Q3_K (3.44). v2: one budget for layers 2-36 (2.0 bpw on average; fragile layers 3-13, 17, 32 get 2.2-2.75, most decoder layers ~1.76), layers 0, 1, 37-39 at 2.25; overall 2.03 |
| Method | GPTQ per expert with Hessians from a layer-streaming calibration forward of DeepSeek's `model.py` on the FP8/FP4 checkpoint; tiers allocated by measured held-out output error per byte (GSQ pipeline from Xolotl, native CUDA GPTQ from ds41) |
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
GPU, `--arena-gb` = RAM arena. Sizing: VRAM + arena should hold ~140 GB of experts. 2x 24 GB GPUs + 128 GB RAM:
`--arena-gb 110` (GPU0's experts adapt to the conversation, the default). One 32 GB GPU + 128 GB RAM: `--arena-gb 80`
and `DS41_ADAPT=0` (the rest is read from disk; see the 5090 numbers). Then any OpenAI-compatible client:

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

Open `http://localhost:8090/` for a chat UI (llama.cpp's web UI, built in). The server also speaks Anthropic
`/v1/messages` and OpenAI `/v1/responses`, with tool calls and streaming. Thinking is on
by default; `"reasoning_effort": "none"` or `"thinking": {"type": "disabled"}` turns it off.

## Credits

[DeepSeek](https://huggingface.co/deepseek-ai) for the model (MIT). [Xolotl / Strata](https://github.com/Indras-Mirror/Xolotl)
(Niko1221 and contributors, MIT) for the GSQ quantization tools and the V4.1 GGUF format.
[Lucebox](https://huggingface.co/Lucebox/DeepSeek-V4.1-Flash-1017-ROCMFP-GGUF) for the 1017 eval set. antirez for the
agentic prompts.

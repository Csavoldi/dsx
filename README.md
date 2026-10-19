# dsx

Run **DeepSeek-V4.1-Flash-1017** (552B MoE + 196B Engram) at full routing on a desktop: two RTX 3090s, 128 GB of RAM,
an NVMe drive. A Rust + CUDA engine built for this one model, with its own 2-bit quant
([lxe/DeepSeek-V4.1-Flash-1017-dsx-2bit](https://huggingface.co/lxe/DeepSeek-V4.1-Flash-1017-dsx-2bit)).

- 30-37 tok/s decode, 334-508 tok/s prefill on 2x RTX 3090
- Exact routing: no expert pruning, no router bias
- OpenAI and Anthropic compatible server with tool calls, streaming, image input and a built-in web UI
- Quality vs the FP8 model: KL 0.395, top-1 74.1%

![dsx speed on 2x RTX 3090 and RTX 5090](cards/dsx-2bit/assets/speed.png)

![Quality against the FP8 original](cards/dsx-2bit/assets/quality.png)

| | 2x RTX 3090, 128 GB DDR4 | RTX 5090, 128 GB DDR5 |
|---|---:|---:|
| Decode, prose | 37.2 tok/s | 12.0 tok/s |
| Decode, 3K-token agent prompt | 30.0 tok/s | 15.2 tok/s |
| Prefill, 3K / 8K tokens | 334 / 508 tok/s | 93 / 172 tok/s |

On the single-GPU box, ~40 GB of experts live on disk, which sets its speed.

## Quick start

```bash
# weights: the quant, plus the Engram tables and tokenizer from the official checkpoint
hf download lxe/DeepSeek-V4.1-Flash-1017-dsx-2bit --local-dir dsx-2bit
hf download deepseek-ai/DeepSeek-V4.1-Flash-1017 --local-dir v41-1017 \
  --include model.safetensors.index.json model-00047-of-00048.safetensors model-00048-of-00048.safetensors tokenizer.json

# build (CUDA_ARCHS: 86 = RTX 30xx, 89 = RTX 40xx, 120 = RTX 50xx, 80 = A100) and serve
CUDA_ARCHS=86 cargo build --release -p dsx --features cuda
./target/release/dsx-server --model dsx-2bit/DeepSeek-V4.1-Flash-1017-dsx-2bit-00001-of-00004.gguf \
  --engram v41-1017 --tokenizer v41-1017/tokenizer.json --host 0.0.0.0 --port 8090 \
  --ctx 131072 --gpu0-gb=-1.5 --gpu1-gb=-1.0 --arena-gb 110 --rank dsx-2bit/expert-rank.json \
  --vision dsx-2bit/vision.safetensors
```

Open `http://localhost:8090/` for the web UI, or use any OpenAI client with base URL `http://localhost:8090/v1`:

```bash
curl http://localhost:8090/v1/chat/completions -H 'content-type: application/json' -d '{
  "model": "deepseek-v4.1-flash", "stream": true, "reasoning_effort": "none",
  "messages": [{"role": "user", "content": "Write a haiku about MoE routing."}] }'
```

Thinking is on by default; `"reasoning_effort": "none"` or `"thinking": {"type": "disabled"}` turns it off.

## Requirements

| | |
|---|---|
| GPU | NVIDIA, 24 GB+, CUDA 13. Run on RTX 3090 (sm_86), RTX 4090 (sm_89) and RTX 5090 (sm_120); A100 (sm_80) builds with `CUDA_ARCHS=80` but is untested |
| Memory | ~140 GB of VRAM + RAM for the experts |
| CPU | x86 with AVX2 (AVX-VNNI or AVX-512 VNNI preferred) |
| Disk | ~350 GB: the quant (148 GB) + the Engram shards (203 GB) |

## Placement

Experts are spread across VRAM, a RAM arena and the file, hottest first.

| Flag | |
|---|---|
| `--gpu0-gb`, `--gpu1-gb` | GB of experts per GPU; negative = all free VRAM minus that much; `--gpu1-gb 0` for one GPU |
| `--arena-gb` | RAM for experts the CPU computes |
| `--rank` | expert hotness order (`expert-rank.json` from the quant repo) |
| `--vision` | the vision tower (`vision.safetensors` from the quant repo): image input; leave it out for text only |
| `DSX_ADAPT=N` | GPU0 expert swaps per token to follow the conversation (default 8; 0 = static) |

| Machine | Flags |
|---|---|
| 2x 24 GB GPU, 128 GB RAM | `--gpu0-gb=-1.5 --gpu1-gb=-1.0 --arena-gb 110` |
| 1x 32 GB GPU, 128 GB RAM | `--gpu0-gb=-2.0 --gpu1-gb 0 --arena-gb 80`, `DSX_ADAPT=0` |

## How it works

| Part | Where it runs |
|---|---|
| Attention, shared expert, routers (Q8_0) | GPU0 |
| Hottest routed experts (~3,300 of 15,360) | VRAM on both GPUs; GPU0's set adapts to the conversation |
| Remaining experts | 14 CPU threads, straight from a hugepage RAM arena |
| Engram n-gram rows | read from the official shards by a prefetch thread |

- Within a token, GPU0 posts each layer's expert picks to shared memory and the CPU and GPU1 answer through flags,
  so the whole token runs without waiting on the host.
- Prefill runs the first half of the layers on every token and the second half only on the tokens it can still affect.
- The server reuses the cached prefix across turns.

## Clients

Any OpenAI-compatible client works. For the [pi](https://github.com/badlogic/pi-mono) coding agent, add to
`~/.pi/agent/models.json`:

```json
{ "providers": { "dsx": {
  "baseUrl": "http://localhost:8090/v1", "api": "openai-completions", "apiKey": "none", "authHeader": false,
  "compat": { "supportsDeveloperRole": false, "supportsReasoningEffort": true, "maxTokensField": "max_tokens" },
  "models": [{ "id": "deepseek-v4.1-flash", "reasoning": true, "input": ["text"], "contextWindow": 131072,
               "maxTokens": 8192, "thinkingLevelMap": { "off": "none", "high": "high" } }] } } }
```

```bash
pi --provider dsx --model deepseek-v4.1-flash:off
```

## Tools

`target/release/dsx` (NVIDIA runner):

| Command | |
|---|---|
| `gen` | prefill + greedy decode, with timings |
| `eval` | KL / top-1 / perplexity against teacher log-probs |
| `cmp`, `ref` | GPU against the CPU reference implementation |

## On one AMD MI325X

[`amd/`](amd/) is a second runner (Rust + HIP) for one AMD Instinct MI325X, which holds the whole model in its 256 GiB
of HBM. Its build keeps most experts in the checkpoint's own MXFP4
([lxe/DeepSeek-V4.1-Flash-1017-dsx-mxfp4](https://huggingface.co/lxe/DeepSeek-V4.1-Flash-1017-dsx-mxfp4), 270 GB):
110-114 tok/s decode at up to 256K context, KL 0.049 vs FP8. Same server as above. Setup and numbers:
[amd/README.md](amd/README.md), [FINDINGS-AMD.md](FINDINGS-AMD.md).

## Repo

| Path | |
|---|---|
| [`core/`](core/) | shared by both runners: config, GGUF / safetensors readers, block formats, the CPU reference forward and DSpark reference, the server and web UI |
| [`nvidia/`](nvidia/) | the desktop runner: CUDA kernels, expert placement over VRAM / RAM / disk, CPU expert kernels ([nvidia/README.md](nvidia/README.md)) |
| [`amd/`](amd/) | the MI325X runner: HIP kernels, the MXFP4 + 3-bit build and its export, DSpark on the GPU |
| [`quant/`](quant/) | the 2-bit quantization pipeline, with native CUDA GPTQ |
| [`cards/`](cards/) | the Hugging Face model cards ([dsx-2bit](cards/dsx-2bit/README.md), [dsx-mxfp4](cards/dsx-mxfp4/README.md)) |
| [`FINDINGS.md`](FINDINGS.md), [`FINDINGS-AMD.md`](FINDINGS-AMD.md) | research notes and measurements |

One Cargo workspace: `cargo build --release` builds `core` and the NVIDIA runner (CPU-only without
`--features cuda`); `-p dsx-amd` builds the AMD runner (needs ROCm). Binaries land in `target/release/`.

The 2-bit quant uses dsx's per-expert formats (ternary, Q2_0, Q3_K per expert) and loads in the NVIDIA runner; the
MXFP4 + 3-bit build loads in the AMD runner.

## Credits

[DeepSeek](https://huggingface.co/deepseek-ai) for the model (MIT) ·
[Xolotl / Strata](https://github.com/Indras-Mirror/Xolotl) for the GSQ tools and V4.1 GGUF format (MIT) ·
[Lucebox](https://huggingface.co/Lucebox/DeepSeek-V4.1-Flash-1017-ROCMFP-GGUF) for the eval set ·
antirez for the calibration prompts · the web UI is [llama.cpp](https://github.com/ggml-org/llama.cpp)'s, by Georgi
Gerganov and the ggml authors (MIT), vendored in [`core/webui`](core/webui) with its license.

MIT license.

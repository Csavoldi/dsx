# dsx

Run **DeepSeek-V4.1-Flash-1017** (552B MoE + 196B Engram) at full routing on a desktop: two RTX 3090s, 128 GB of RAM,
an NVMe drive. A Rust + CUDA engine built for this one model, with its own 2-bit quant
([lxe/DeepSeek-V4.1-Flash-1017-dsx-2bit](https://huggingface.co/lxe/DeepSeek-V4.1-Flash-1017-dsx-2bit)).

- ~29 tok/s decode, 300-470 tok/s prefill on 2x RTX 3090
- Exact routing: no expert pruning, no router bias
- OpenAI and Anthropic compatible server with tool calls, streaming, image input and a built-in web UI
- Quality vs the FP8 model (Lucebox eval): KL 0.373, top-1 74.4%

| | 2x RTX 3090, 128 GB DDR4 | RTX 5090, 128 GB DDR5 |
|---|---:|---:|
| Decode, prose | 28.9 tok/s | 12.0 tok/s |
| Decode, 3K-token agent prompt | 28.7 tok/s | 15.2 tok/s |
| Prefill, 3K / 8K tokens | 301 / 470 tok/s | 93 / 172 tok/s |

2x RTX 3090: build v4 (same layout as the current v4b); RTX 5090: build v2. On the single-GPU box, ~40 GB of experts
live on disk, which sets its speed.

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
  --ctx 131072 --arena-gb 110 --rank dsx-2bit/expert-rank.json
```

Open `http://localhost:8090/` for the web UI, or use any OpenAI client with base URL `http://localhost:8090/v1`:

```bash
curl http://localhost:8090/v1/chat/completions -H 'content-type: application/json' -d '{
  "model": "deepseek-v4.1-flash", "stream": true, "reasoning_effort": "none",
  "messages": [{"role": "user", "content": "Write a haiku about MoE routing."}] }'
```

Thinking is on by default; `"reasoning_effort": "none"` or `"thinking": {"type": "disabled"}` turns it off.
On a machine others can reach, add `--api-key` (see [Parameters](#parameters)).

## Requirements

| | |
|---|---|
| GPU | NVIDIA, 24 GB+, CUDA 13. Run on RTX 3090 (sm_86), RTX 4090 (sm_89) and RTX 5090 (sm_120); A100 (sm_80) builds with `CUDA_ARCHS=80` but is untested |
| Memory | ~140 GB of VRAM + RAM for the experts |
| CPU | x86 with AVX2 (AVX-VNNI or AVX-512 VNNI preferred) |
| Disk | ~350 GB: the quant (148 GB) + the Engram shards (203 GB) |

## Parameters

`dsx-server` options; every one has a default except `--model`, `--engram` and `--tokenizer`.

**Model files**

| | |
|---|---|
| `--model` | the quant's first GGUF shard (the others load from beside it) |
| `--engram` | directory with the official checkpoint's Engram shards (`model-00047/00048-of-00048.safetensors`) |
| `--tokenizer` | the official `tokenizer.json` |
| `--rank` | expert hotness order (`expert-rank.json` in the quant repo); without it, experts are placed layer-major round robin |
| `--vision` / `--no-vision` | image input: default `vision.safetensors` next to the model when present (~0.5 GB of the main GPU); `--no-vision` for text only |

**Server**

| | |
|---|---|
| `--host`, `--port` | listen address (default `127.0.0.1:8090`) |
| `--api-key` | require this key on every API request (`Authorization: Bearer <key>` or `x-api-key: <key>`; also `DSX_API_KEY`). Health checks and the web UI's files stay open. One shared key, no TLS: put a TLS proxy in front of a public address |
| `--model-name` | model id reported by `/v1/models` (default `deepseek-v4.1-flash`) |
| `--ctx` | context window, prompt + generated tokens (default 32768; 131072 costs ~0.5 GB more of the main GPU) |
| `--max-tokens` | generation cap when a request sets none (default 8192) |

**Sampling defaults** (a request's own fields override them)

| | |
|---|---|
| `--temperature`, `--top-p` | default 1.0 and 1.0, the model's recommended sampling |
| `--hesitation-penalty` | logit penalty on hesitation tokens ("Wait", "But", ...) while reasoning; 0 = off (default) |

**Memory and placement**

| | |
|---|---|
| `--gpu0-gb` | GB of experts on the main GPU; negative = its free VRAM minus that much (default -1.5) |
| `--gpu1-gb` (`--aux-gb`) | GB of experts on each other GPU; negative = free VRAM minus that much (default -1.0); 0 = main GPU only |
| `--arena-gb` | RAM for the experts the CPU computes and copies of the cached ones (default 0: set it, ~110 on 128 GB) |
| `--chunk` | prefill tokens per pass (default 4096); its buffers take main-GPU VRAM: 2048 fits ~150 more experts there, 1024 ~220, for slower prefill |
| `DSX_GPUS=0,2` | use only these GPUs |
| `DSX_ADAPT=N` | expert swaps per token per adaptive GPU (default 8; 0 = static placement) |
| `DSX_AUX_ADAPTIVE=1` / `0` | force the other GPUs adaptive / fixed regardless of their link |

| Machine | Flags |
|---|---|
| 2x 24 GB GPU, 128 GB RAM | `--arena-gb 110` |
| 1x 32 GB GPU, 128 GB RAM | `--gpu0-gb=-2.0 --arena-gb 80`, `DSX_ADAPT=0` (the cache's RAM copies cost space this machine does not have) |

`--mock` (with `--mock-echo`, `--mock-delay-ms`) runs a scripted engine without the model, for client testing.

## Placement

Every routed expert (40 layers x 384) lives in exactly one place that computes it during decode:

| Where | On 2x RTX 3090, 128 GB | Computes |
|---|---|---|
| Main GPU (best host link) | ~1,050 experts, 10.5 GB, next to the dense weights | itself, an adaptive cache |
| Other GPUs | ~2,190 experts, 23 GB on a 24 GB card | themselves, on the activations shipped to them |
| RAM arena (hugepages) | ~12,600 experts, 110 GB | 14 CPU threads |
| The GGUF file | the ~550 coldest | the CPU, through the page cache |

About 74% of decode's expert picks are served from VRAM on that machine; the CPU's share sets decode speed.

**GPU roles.** At startup dsx measures each visible GPU's memory and host-to-device bandwidth. The best-linked one
(within 15%, then the most memory) becomes the main GPU: dense weights, KV cache, prefill buffers, vision tower and
its own expert cache. Every other GPU is an expert store. A store whose link is at least half as fast as the main
GPU's swaps experts like the main GPU's cache; a slower one (a PCIe x4 slot) holds a fixed set, because copying
experts over it would compete with the activations it needs every layer.

**Which experts go where.** `--rank` orders experts by how often the router picked them on calibration text
(`expert-rank.json` in the quant repo). Fixed stores take the hottest first, spread evenly across them; adaptive
stores take the next; the arena takes the rest in the same order until `--arena-gb` is full; whatever is left stays
in the file.

**Following the conversation.** After each token, every adaptive store commits finished copies and starts up to
`DSX_ADAPT` new ones (default 8): the experts with the highest decayed usage (x0.97 per token, seeded from the
ranking) replace clearly colder residents of the same quant types, copied from the arena on a side stream. Experts an
adaptive store may evict keep a copy in the arena, which is why the arena is sized for them too.

**Prefill** streams the main GPU's missing experts from the arena over PCIe while the previous expert computes; other
GPUs run their own experts on the chunk's activations in parallel.

Flags: [Parameters](#parameters) (memory and placement).

## How it works

| Part | Where it runs |
|---|---|
| Attention, indexer, shared expert (Q8_0), routers (bf16) | the main GPU |
| Routed experts | wherever they live (above): main GPU, other GPUs, CPU |
| Engram n-gram rows | read from the official shards; during decode by prefetch threads while layer 0 runs |

- Per layer, the main GPU computes attention and the router, writes the 6 picks and the expert input to mapped host
  memory and raises a flag. The CPU worker and every other GPU (its stream waits on the flag) compute the picks they
  hold and write partial sums into their own slots; the main GPU runs its own experts and the shared expert meanwhile,
  waits on all flags and adds everything. A token runs without host synchronization.
- Prefill runs the first half of the layers on every prompt token and the second half only on the tokens later
  positions can still attend to, in chunks of `--chunk` tokens.
- The server keeps the cached prefix across turns and re-prefills only what changed.

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

With `--api-key`: drop `"authHeader": false`, set `"apiKey"` to the key and point `baseUrl` at the server's URL.

## Tools

`target/release/dsx` (NVIDIA runner):

| Command | |
|---|---|
| `gen` | prefill + greedy decode, with timings; `--bench-verify T` times T-token verify passes, `--check-verify T` checks them against decode |
| `eval` | KL / top-1 / perplexity against teacher log-probs (`--decode` token by token, `--from N` for positions N+) |
| `cmp`, `ref` | GPU against the CPU reference implementation |

## Repo

| Path | |
|---|---|
| [`core/`](core/) | config, GGUF / safetensors readers, block formats, the CPU reference forward, the server and web UI |
| [`nvidia/`](nvidia/) | the desktop runner: CUDA kernels, expert placement over VRAM / RAM / disk, CPU expert kernels ([nvidia/README.md](nvidia/README.md)) |
| [`amd/`](amd/) | a separate runner for one AMD MI325X with a near-lossless MXFP4 build: [amd/README.md](amd/README.md) |
| [`quant/`](quant/) | the 2-bit quantization pipeline, with native CUDA GPTQ |
| [`cards/`](cards/) | the Hugging Face model cards |
| [`FINDINGS.md`](FINDINGS.md), [`DESIGN.md`](DESIGN.md) | measurements and design notes |

`cargo build --release` builds `core` and the NVIDIA runner (CPU-only without `--features cuda`); binaries land in
`target/release/`.

## Credits

[DeepSeek](https://huggingface.co/deepseek-ai) for the model (MIT) ·
[Xolotl / Strata](https://github.com/Indras-Mirror/Xolotl) for the GSQ tools and V4.1 GGUF format (MIT; derived files in `quant/v41` carry its license) ·
DeepSeek for the reference `model.py` the calibration runs · GPTQ ([Frantar et al.](https://arxiv.org/abs/2210.17323)) ·
[Lucebox](https://huggingface.co/Lucebox/DeepSeek-V4.1-Flash-1017-ROCMFP-GGUF) for the eval set ·
antirez for the calibration prompts · the web UI is [llama.cpp](https://github.com/ggml-org/llama.cpp)'s, by Georgi
Gerganov and the ggml authors (MIT), vendored in [`core/webui`](core/webui) with its license.

MIT license.

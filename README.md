# ds41

A Rust + CUDA inference engine for one model and one class of machine:
**[DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017)** (552B MoE + 196B Engram)
on a desktop with **2x RTX 3090, 128 GB DDR4 and an NVMe drive**. It ships with its own 2-bit quant:
**[lxe/DeepSeek-V4.1-Flash-1017-ds41-2bit](https://huggingface.co/lxe/DeepSeek-V4.1-Flash-1017-ds41-2bit)**.

Measured on that box (i7-12700K, 128 GB DDR4-3200, RTX 3090 on PCIe 4.0 x16 + RTX 3090 on PCIe 3.0 x4), exact routing:

| | |
|---|---:|
| Decode, 3000-token agent prompt (engine benchmark) | **31.5 tok/s** |
| Decode on general prose (more experts outside the agentic hot set) | 22-24 tok/s |
| Prefill, 3000 tokens | 340 tok/s |
| Quality vs the FP8 model (8 x 1024 tokens): KL / top-1 / perplexity | 0.462 / 71.5% / 11.66 (FP8: 0 / 100% / 8.56) |

Details: [FINDINGS.md](FINDINGS.md) (research and every measurement), [DESIGN.md](DESIGN.md), [bench/RESULTS.md](bench/RESULTS.md).

## How it runs a 750B-parameter model on a desktop

- **Dense half on GPU0** (attention, shared expert, routers, Q8_0): ~14 ms per token.
- **Routed experts in three tiers**: the hottest ~3,000 of 15,360 in VRAM on both GPUs (ranked by how often an agentic
  calibration set routes to them: the hottest 25% take ~58% of the traffic), the rest in a 104 GB hugepage RAM arena
  computed in place by 14 CPU threads (AVX2 / AVX-VNNI int8 kernels). Nothing is read from disk per token.
- **No host synchronization per layer**: GPU0 publishes each layer's picks to mapped host memory, the CPU worker and GPU1
  pick them up by spinning on flags, GPU0's stream waits on theirs (~6 us round trip).
- **Engram** n-gram rows (two 98 GB tables) are read from the official checkpoint by a prefetch thread while layer 0 runs.
- **Prefill** runs the encoder half on every token and the decoder half only on the tokens later layers can still see.
- An **OpenAI / Anthropic compatible server** (chat completions, responses, messages, tool calls, streaming) with prefix
  reuse across turns, and llama.cpp's web UI built in.

## Requirements

- An NVIDIA GPU with >= 24 GB (built for sm_86 and sm_120; set `CUDA_ARCHS`), CUDA 13, Rust stable.
- About 140 GB of RAM + VRAM for the experts to stay off the disk (the engine runs with less: experts that do not fit
  are read from the GGUF through the page cache, at NVMe speed).
- An x86 CPU with AVX2 (AVX-VNNI preferred). The thread layout assumes an i7-12700K (CPUs 0-15 P-cores); see
  `src/cpu/pool.rs`.
- ~350 GB of disk: the GGUF (148 GB) and the two official shards that hold the Engram tables (203 GB).

## Get the weights

```bash
pip install -U huggingface_hub
# the quant (4 parts + checksums), then join and verify
hf download lxe/DeepSeek-V4.1-Flash-1017-ds41-2bit --local-dir ds41-2bit
bash ds41-2bit/reassemble.sh ds41-2bit          # --delete-parts if short on disk
# Engram tables + tokenizer from the official 1017 checkpoint (only these files)
hf download deepseek-ai/DeepSeek-V4.1-Flash-1017 --local-dir v41-1017 \
  --include model.safetensors.index.json model-00047-of-00048.safetensors model-00048-of-00048.safetensors tokenizer.json
```

## Build and run

```bash
cd engine
CUDA_ARCHS=86 cargo build --release --features cuda     # 86 = RTX 30xx, 120 = RTX 50xx

# OpenAI / Anthropic compatible server
./target/release/ds41-server --model ../ds41-2bit/DeepSeek-V4.1-Flash-1017-ds41-2bit.gguf \
  --engram ../v41-1017 --tokenizer ../v41-1017/tokenizer.json --host 0.0.0.0 --port 8090 \
  --ctx 131072 --gpu0-gb=-1.5 --gpu1-gb=-1.0 --arena-gb 104 --rank ../ds41-2bit/expert-rank.json

curl http://localhost:8090/v1/chat/completions -H 'content-type: application/json' -d '{
  "model": "deepseek-v4.1-flash", "messages": [{"role": "user", "content": "Write a haiku about MoE routing."}],
  "reasoning_effort": "none", "stream": true }'
```

Placement flags: `--gpu0-gb` / `--gpu1-gb` are GB of experts per GPU (negative = all free VRAM minus that much),
`--arena-gb` the RAM arena, `--rank` the hotness order. With one GPU pass `--gpu1-gb 0`. Thinking is on by default;
`"reasoning_effort": "none"` (OpenAI) or `"thinking": {"type": "disabled"}` turns it off per request.

Open `http://HOST:8090/` for a chat UI: llama.cpp's web UI, embedded in the server (`engine/webui`, rebuilt by
`webui/build.sh`). Any OpenAI-compatible client works (base URL `http://HOST:8090/v1`, any API key). For example the
[pi](https://github.com/badlogic/pi-mono) coding agent, `~/.pi/agent/models.json`:

```json
{ "providers": { "ds41": {
  "baseUrl": "http://HOST:8090/v1", "api": "openai-completions", "apiKey": "none", "authHeader": false,
  "compat": { "supportsDeveloperRole": false, "supportsReasoningEffort": true, "maxTokensField": "max_tokens" },
  "models": [{ "id": "deepseek-v4.1-flash", "reasoning": true, "input": ["text"], "contextWindow": 131072,
               "maxTokens": 8192, "thinkingLevelMap": { "off": "none", "high": "high" } }] } } }
```

```bash
pi --provider ds41 --model deepseek-v4.1-flash:off
```

Command-line tools (`target/release/ds41`): `gen` (prefill + greedy decode, timings, `--bench-verify`), `eval`
(KL / top-1 / perplexity against teacher log-probs), `cmp` / `ref` (GPU against the CPU reference). See
[engine/README.md](engine/README.md).

## Quant

[quant/](quant/) has the pipeline (on top of Xolotl's V4.1 tools): layer-streaming calibration on agentic text,
per-expert GPTQ into 1.75 / 2.25 / 3.44-bit tiers at a 2.03 bits/weight budget, dense weights Q8_0.
The file is **not** a llama.cpp GGUF: llama.cpp has no V4.1 architecture and none of these expert types. It uses the
same format as [Xolotl](https://github.com/Indras-Mirror/Xolotl)'s V4.1 GSQ release, so Xolotl should load it too (untested).

## Status

Research code, tuned on one machine. Next (v2): close the quality gap (spill the coldest experts to NVMe to afford
~2.3-2.5 bits/weight), expert placement that adapts to the conversation (general prose runs at 22-24 tok/s because the VRAM hot set is
ranked on agentic text), faster prefill and dense projections, DSpark speculative decoding.

## Credits

DeepSeek for the model (MIT). [Xolotl / Strata](https://github.com/Indras-Mirror/Xolotl) (Niko1221 and contributors,
MIT) for the GSQ quant tools and the V4.1 GGUF format. [Lucebox](https://huggingface.co/Lucebox/DeepSeek-V4.1-Flash-1017-ROCMFP-GGUF) for the 1017 quality
eval set. antirez for the agentic calibration prompts.

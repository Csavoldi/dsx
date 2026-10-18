# dsx design

One model (DeepSeek-V4.1-Flash-1017), one quant format, one class of machine: two 24 GB GPUs, 128 GB of RAM, an NVMe
drive. The performance model behind these choices is in [FINDINGS.md](FINDINGS.md), section 3.

## Principles

1. **No experts on disk.** A read from NVMe costs ~7× the same read from RAM; VRAM + RAM hold the whole expert pool.
2. **The CPU computes RAM experts in place** during decode. GPUs take RAM experts only during prefill, where one copy
   serves a whole chunk.
3. **VRAM holds the hottest experts**, ranked by measured routing frequency; GPU0's share adapts to the conversation.
4. **GPU1 is a fixed expert store.** Its PCIe 3.0 x4 link carries only activations.
5. **Quant formats fit the kernels:** CPU unpacking keeps up with RAM bandwidth (AVX2 / AVX-VNNI), GPU experts run on
   int8 `mma.sync`.
6. **Exact routing.** A routing bias exists for experiments, off by default.

## Memory layout

| Device | Contents |
|---|---|
| GPU0 (24 GB, x16) | dense weights (Q8_0, ~8 GB), KV cache and indexer keys, adaptive expert cache (~11 GB), prefill buffers |
| GPU1 (24 GB, x4) | fixed set of hot experts (~23 GB) |
| RAM (~110 GB) | hugepage-backed, CUDA-registered arena: every expert not on GPU1, laid out per (layer, expert) |
| NVMe | the GGUF (loaded once; the coldest experts are read from it on demand) and the Engram tables |

## Decode

Per layer, with no host synchronization inside a token:

1. GPU0 runs attention and the router, then writes the 6 expert ids and the expert input into mapped host memory
   and raises a flag.
2. The CPU worker (a coordinator on its own core, 14 compute threads on the P-cores) sees the flag, computes the
   RAM-resident experts, writes the partial sum and raises its flag.
3. GPU1's stream waits on the same flag (`cuStreamWaitValue32`), runs its resident experts and writes its partial sum.
4. GPU0 runs its own resident experts and the shared expert meanwhile, waits on both flags, adds the three sums and
   continues.

A flag round trip costs ~6 µs. After each token, GPU0's expert cache commits finished copies and starts up to 8 new
ones: the hottest experts by decayed usage replace the coldest resident expert of the same quant type, copied from the
arena and repacked on a side stream.

## Prefill

- Layers 0-20 run on every prompt token; layers 21-39 run only on the tokens later positions can still attend to.
- Prompts are processed in chunks (4096 tokens by default). Each chunk streams the arena's experts to GPU0 through
  four staging buffers while the previous expert computes; GPU1 computes its own experts on the shipped activations.
- Attention uses a tensor-core kernel for long chunks, and Engram rows are deduplicated per chunk.

## Server

- OpenAI chat completions and responses, Anthropic messages, tool calls, streaming, through DeepSeek's
  `deepseek-recipe` prompt encoding.
- Prefix reuse: the KV of the last conversation stays on the GPU; a request continues it or rewinds to the snapshot
  at the end of the previous prompt.
- llama.cpp's web UI is embedded and served at `/`, with the llama.cpp endpoints it expects (`/props`, `/slots`) and
  a `timings` chunk at the end of streamed responses.

## Quantization

Per-expert tiers (ternary 1.75, Q2_0 2.25, Q3_K 3.44 bpw) allocated by measured error under one budget shared across
layers; GPTQ with Hessians from a layer-streaming calibration pass through DeepSeek's `model.py`. Native CUDA kernels
handle GPTQ and the scale search, batched over experts. Details: [quant/README.md](quant/README.md).

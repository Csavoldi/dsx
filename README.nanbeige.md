<h1 align="center">DSX · Nanbeige</h1>

<p align="center"><b>Nanbeige4.3-33B on one RTX 5090</b><br>
Rust and CUDA · EXL3 weights · browser chat · OpenAI-compatible API · Pi tool calls</p>

DSX runs **[Nanbeige4.3-33B](https://huggingface.co/Nanbeige/Nanbeige4.3-33B)** locally on a 32 GB RTX 5090.
This branch contains the Nanbeige runner, its server and measurements. The DeepSeek runner is on
[`main`](https://github.com/lxe/dsx/tree/main).

The current setup uses [Luni's 4.0 bpw EXL3 decoder](https://huggingface.co/Luni/Nanbeige-4.3-33B-EXL3/tree/4.00bpw),
a 4-bit output head and a 3-bit KV cache. It shares DSX's request worker, sampler and browser UI.

## Quick start

You need Rust, CUDA 13, a 32 GB RTX 5090 and about 35 GB of disk space for the two model packs.
Build from the `nanbeige` branch:

```bash
git switch nanbeige
CUDA_ARCHS=120 cargo build --release -p dsx --features cuda \
  --bin dsx-nanbeige --bin dsx-nanbeige-server
```

Download both packs. The Luni pack supplies the decoder weights; the base pack supplies embeddings,
norms and the uncompressed output head that DSX quantizes when it loads the model.

```bash
hf download lxe/Nanbeige4.3-33B-EXL3-2.5bpw --local-dir models/nanbeige-base
hf download Luni/Nanbeige-4.3-33B-EXL3 \
  --revision bc301055ecdfd5d992f2afaf5f6a405a7f235572 \
  --local-dir models/nanbeige-4bpw
```

Start the server:

```bash
DSX_NB_BATCH=4096 DSX_NB_ATTN8=2 DSX_NB_PART_SIZE=16384 \
DSX_NB_EXL3_NARROW=1 DSX_NB_EXL3_EXPERIMENT=pf4 DSX_NB_GQA_I8PV=1 \
DSX_NB_EXL3=models/nanbeige-4bpw \
  ./target/release/dsx-nanbeige-server --model models/nanbeige-base \
  --ctx 262144 --kv-bits 3 --host-embed --head-bits 4 --lookup 3 \
  --host 127.0.0.1 --port 8095
```

Open [localhost:8095](http://localhost:8095/) for browser chat. The API endpoint is
`http://localhost:8095/v1`, and the model ID is `nanbeige4.3-33b`.

## Coding agents and API

For Pi, add the provider from [models.json](validation/nanbeige/pi-models.json) to your Pi model configuration,
then run:

```bash
pi --provider dsx-nanbeige --model nanbeige4.3-33b --thinking off
```

The provider uses XML tool calls and preserves reasoning in conversation history. Thinking is off by default;
set `chat_template_kwargs.enable_thinking` to `true` to enable it. Nanbeige exposes an on/off switch for reasoning.

```bash
curl http://localhost:8095/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{
    "model": "nanbeige4.3-33b",
    "stream": true,
    "temperature": 1.0,
    "top_p": 0.95,
    "chat_template_kwargs": {"enable_thinking": false},
    "messages": [{"role": "user", "content": "Write a Python function that merges sorted iterators."}]
  }'
```

## Performance

On the local RTX 5090, short-context Pi code-generation requests ran at roughly **40–50 tokens/s** with the
4.0 bpw setup. A cold **3,910-token** API prompt took **2.42 seconds**, or **1,615 prefill tokens/s**.
Repeating that request reused all 3,910 prompt tokens and returned the same answer. These runs allocated
262,144-token capacity; throughput falls as the populated conversation grows.

The matched quant comparison used one 30-token coding prompt, 512 output tokens, the same runtime head and KV
precision, and prompt lookup width 3. Its precision check scored the final 128 tokens of a 4,096-token WikiText input.

| Decoder | Coding tokens/s | Perplexity | Mean KL against source logits |
|---|---:|---:|---:|
| Local EXL3 2.5 bpw | 64.08 | 12.7023 | 0.19287 |
| Luni EXL3 4.0 bpw | 52.95 | 11.1146 | 0.06081 |

Those measurements used the previous attention path. Enabling INT8 GQA attention with the 4.0 bpw decoder changed
perplexity to **11.1642**, a 0.45% increase on the same sample. The current default favors the 4.0 bpw pack's measured
precision; the 2.5 bpw pack remains available for faster short-context decoding.

The [quant receipts](validation/nanbeige/agent-retry-20261026/quant-comparison/summary.json) and
[attention measurements](validation/nanbeige/agent-retry-20261026/gqa-i8pv/report.md) record settings, numerical error
and context lengths. The attention scaling comparison uses the 2.5 bpw decoder. Prefill optimization remains open.

## Context and cache

`--ctx 262144` reserves 256K-token capacity. The 3-bit KV cache occupies about 10.16 GiB and keeps a BF16 window
for recent keys and values. `--host-embed` reads embedding rows from the CPU-mapped model file. `--head-bits 4`
quantizes the output head at startup; `6`, `8` and `0` select other head precisions, with zero retaining 16 bits.

Follow-up requests reuse the longest matching token prefix. The worker keeps a pinned-RAM snapshot of the recent
KV window so it can restore state after a longer rewind or an interrupted request. API usage reports reused tokens
separately from newly processed input. `--lookup 3` proposes repeated prompt text and verifies it with the model;
its benefit depends on how often the output repeats earlier text.

## Validation and task runs

The tests cover EXL3 decoding against an independent CPU implementation, projection kernels against BF16
references, attention against an f64 causal-softmax reference, chat history, Unicode streaming and typed tool calls.
The precision scores above include weight, head and KV effects. Task completion is evaluated separately.

In the [expanded Pi cube run](validation/nanbeige/expanded-cube-20261026/README.md), the model wrote and repaired a
rotating wireframe without code fixes from the parent agent. The cube rotates, though its framing remains left of
center. The recording preserves the initial blank frames and subsequent repairs. Earlier assisted runs and Doom
attempts remain in [the experiment records](validation/nanbeige/).

## Implementation

| File | Contents |
|---|---|
| [nanbeige.rs](nvidia/src/nanbeige.rs) | Model loading, EXL3 execution, KV cache and CUDA graphs |
| [nanbeige.cuh](nvidia/src/kernels/nanbeige.cuh) | Attention, normalization, RoPE and head kernels |
| [nanbeige-exl3.cu](nvidia/src/kernels/nanbeige-exl3.cu) | EXL3 projection kernels |
| [nanbeige-server.rs](nvidia/src/bin/nanbeige-server.rs) | Chat template, streaming API, tool calls and browser UI |

MIT license. EXL3 kernel attribution is in [vendor/exllamav3](vendor/exllamav3/).

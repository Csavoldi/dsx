# A Rust engine for DeepSeek-V4.1-Flash-1017 on .102

One model, one quant, one machine (2x RTX 3090, i7-12700K, 128 GB DDR4-3200). Plan only; nothing built yet.
The physics behind it is FINDINGS.md section 6. Working name: `ds41-102`.

## Goal

| Metric | Target | Ceiling (roofline.py) | Best today on similar hardware |
|---|---:|---:|---:|
| Decode, agent session | 25-40 tok/s | ~17-19 exact / ~25 biased per pass, x1.5-2 with DSpark | Xolotl 23-28 (4090, biased) |
| Prefill, new tokens | 1,000+ tok/s | ~2,000-3,700 | Xolotl 250 |
| Follow-up turn | only new tokens are read | | |

## Rules from the physics

1. **No experts on NVMe.** One NVMe read costs ~7x the same read from RAM.
2. **Experts average ~1.9 bpw** so the pool (~127 GB) fits in VRAM (~32 GB over both cards) plus a ~95 GB pinned RAM arena.
3. **The CPU computes RAM experts in place** during decode. The GPUs take RAM experts only in prefill, where one copy serves a whole chunk.
4. **GPU1 is a static expert store**, holding the experts most likely to be hit. Its PCIe 3.0 x4 link carries only activations.
5. **The quant format is picked for the kernels**: an AVX2/VNNI unpack that keeps up with ~40 GB/s on 8 P-cores, and int8 `mma.sync` on sm_86. Probably Q2_0-like 2-bit with ternary for cold experts, Q8_0/Q6_K dense, FP8 Engram rows left as they are.
6. **Session KV is never thrown away** (890 B/token). Each turn replays only the 128-token window and prefills the new tokens.
7. **Speculative decoding (DSpark) from day one**, verifying through the same tiers.
8. **Routing bias is optional, off by default**, and its quality cost is measured. Exact routing is the baseline.

## Layout

```
GPU0 (24 GB, x16)   dense half (Q8/Q6, ~8 GB), FP4 KV + indexer keys, DSpark drafts (~3 GB),
                    expert slots (~11 GB, LRU per layer), prefill staging buffers
GPU1 (24 GB, x4)    ~21 GB static hot experts + their kernels; gets the layer's hidden state, returns partial sums
CPU  (8P + 4E)      P-cores: RAM-expert matvecs (AVX2/VNNI), pinned per layer; E-cores: Engram prefetch, I/O, API
RAM  (~95 GB)       pinned arena: every expert not in VRAM, laid out per (layer, expert) contiguous
NVMe                weights file (load once) + Engram tables via mmap (48 rows/token, prefetched one layer ahead)
```

Per decode layer: GPU0 runs attention and the router, sends the top-6 ids; each tier starts its experts at once;
GPU0 sums the results. The router for layer L+1 can't be known early, but expert prefetch prediction (Xolotl gets
~62-70% recall) can warm GPU0's slots under attention.

## What to borrow

| From | What |
|---|---|
| **pocket-engine** (my Rust+CUDA engine; I assume this is what "Bullet Train kernels" meant, since Bullet Train itself is orchestration) | build.rs/CUDA setup, GGUF quant kernels (bit-exact vs gguf-py), register-decoded `mma.sync` qgemm, small-row qgemv, CUDA-graph decode, GPU sampler, MTP/ngram speculative decoding, prefix cache, OpenAI server. It targets Hopper+ today, so it needs an sm_86 build path (mma.sync is Ampere-native; drop the wgmma/tcgen05/TMA kernels) |
| Strata / Xolotl (MIT) | expert tiering and slot LRU, pinned-arena loading (hugepages: 1.2 GB/s pin vs 0.16), routing-profile seeding, Engram extraction, DSpark verify, the GSQ quant pipeline (`tools/v41`) as the starting point for our quantizer |
| DeepSeek | `inference/model.py` as the spec; [deepseek-recipe](https://github.com/deepseek-ai/deepseek-recipe) (Rust) for prompt encoding and tool-call parsing |
| ik_llama / llama.cpp | CPU i-quant/AVX2 kernels for reference, plus the V4.1 graph to compare outputs against |

## Phases (each gated on a check before the next)

0. **Measure the box (~1 h, no downloads; needs Strata idle, GPU parts need it stopped).**
   RAM read bandwidth; AVX2 matvec GB/s per candidate 2-bit/ternary format on the P-cores (decides rule 5); NVMe
   random 256 B and 10 MB reads; H2D bandwidth to each GPU; empty-kernel and graph-launch latency on a 3090.
1. **Reference and goldens.** Download the 1017 FP8 (510 GB on /home). A layer-streaming CPU reference in Python
   (dequant FP8/FP4, one layer in RAM at a time) produces per-layer hidden states and logits for ~20 prompts,
   including a 32K agent transcript. Slow but exact, and it needs no H100.
2. **Router statistics.** From phase 1 runs: expert hit frequencies per layer on agent/code traffic. This
   decides the GPU1 static set, GPU0 seed set and per-expert bit budget. Check the balanced-routing assumption here.
3. **Quantizer.** Calibrated per-expert formats at ~1.9 bpw average (more bits for hot and early-layer experts).
   Gate: top-1 agreement and KL vs FP8 on held-out agent text, compared with Xolotl's GSQ (0.674 vs 0.701).
4. **Engine, GPU0 only, all experts in VRAM-or-CPU.** Correct first: matches the goldens layer by layer.
5. **Tiers + GPU1**, then **DSpark**, then **chunked prefill with streamed experts**, then **session KV store + API**.
   A/B every step end to end; drop what doesn't measure faster.
6. **Agent benchmark**: replay real agent sessions (30-200K context) against Strata's server API and Xolotl, reporting turn latency, decode tok/s, prefill tok/s.

## Costs and asks before starting

- Disk on .102: 510 GB FP8 + ~130 GB quant + 203 GB Engram tables (extracted) ≈ 850 GB of the 2.1 TB free.
- Phases 0 (GPU part) and 4+ need Strata stopped on .102. Phases 1-3 are CPU and RAM heavy (Strata's 56 GB plus the reference may not fit together).
- Phases 1-3 could run on the workstation instead (125 GB RAM, RTX 5090, 1.1 TB free on /home), which leaves .102 alone until phase 4.

## Update 2026-10-17 afternoon: decisions after phase 0 and reading model.py

Measured (bench/RESULTS.md): RAM 39.5 GB/s; CPU 2-bit matvec 38.9 GB/s on 16 P-core threads (so the CPU is not the
limit); GPU0 H2D 24.7 GB/s only from hugepage memory (13 GB/s from plain pinned memory, IOMMU on); GPU1 3.3 GB/s.

What model.py shows that matters for speed:
- **Prefill only needs the encoder.** Decoder layers 21-39 take their long-range KV from layer 20's cache
  (`kv_source_layers` = 2, 8, 14, 20) and keep only a 128-token window themselves. So a long prompt runs
  layers 0-20 over every token and layers 21-39 over the last 128 only ("SWA bounded replay"). That halves both the
  expert bytes streamed per prefill chunk and the FLOPs. Prefill ceiling ~2x the earlier estimate.
- KV per token: layer 20 latent (ratio 1, FP4 512-d) 288 B + layers 2/8/14 (ratio 2) 432 B + index keys 170 B = 890 B.
- The indexer keys and queries are FP4 (E2M1). Times two, every E2M1 value is an integer, so the indexer can run on int8 tensor cores exactly.
- Attention is MQA (1 KV head, 512-d, 64 query heads) over window + top-512 compressed rows, with an attention sink.
- DSpark: 3 draft blocks (128 experts, top-3) reading the attention inputs of layers 37-39, which draft 5 tokens per
  step, plus a Markov head and a confidence head that decides how many drafts to verify.

Quant plan: run Xolotl's GSQ pipeline (`tools/v41`: layer-streaming calibration on DeepSeek's own model.py, per-expert
GPTQ, Lagrangian tier allocation over PTQ1_0 1.75 / Q2_0 2.25 / Q3_K 3.44 bpw) on the **1017** FP8 checkpoint, with a
budget of **~138 GB** of experts (VRAM ~35 GB with Q6_K dense and the Q2_K DSpark sidecar, plus a ~103 GB pinned arena).
That's ~2.0 bpw on average, about 40% ternary. The output is a GGUF that both Xolotl (as an A/B baseline) and our engine read.
Quality gate: GSQ's held-out head gate (top-1 / ppl vs the FP8 teacher) and an all-Q2_0 control.

Engine (`engine/`, Rust + CUDA, cudarc and an nvcc fatbin as in pocket-engine):
`gguf` loader, hugepage arena, `dense` (GPU0: norms, hc mixes + Sinkhorn, attention, compressor, indexer, router,
shared expert, engram, head), `experts` (GPU kernels for Q2_0/PTQ1_0/Q3_K, CPU AVX2-VNNI kernels for the same, a
per-layer scheduler across GPU0 slots / GPU1 static set / CPU), `dspark`, `prefill` (encoder-only, chunked, expert
streaming), `session` (KV store, window replay), `server` (OpenAI + Anthropic, deepseek-recipe encoding).
Correctness reference: per-layer hidden states from the GSQ calibration chain (exactly model.py) and from Xolotl.

## Decode orchestration (planned, after the correctness pass)

One token = one CUDA graph on GPU0, no host round trips inside it:

1. Per layer, GPU0 runs attention and the router, then writes the top-6 expert ids plus x (20 KB) into
   **mapped pinned host memory** with a release flag (a zero-copy store from the kernel).
2. **CPU workers** (16 threads on the P-cores) spin on that flag, compute the RAM-resident experts in place, and write
   the partial sum (20 KB) into mapped memory with a flag.
3. **GPU1** runs its own graph: a `cuStreamWaitValue32` node on the same flag, a zero-copy read of x, its static experts,
   then a write of the partial sum to mapped memory plus a flag.
4. GPU0 computes its VRAM-slot experts and the shared expert meanwhile, waits on the two flags (wait-value memop
   nodes in the graph), adds the three partial sums, and continues.

Each handshake is a ~1-5 us flag flip; nothing goes through cuMemcpy or a host sync. The expected per-token cost is
dense ~10-12 ms plus, per layer, max(CPU experts, GPU1 experts), plus 40 x ~10 us of handshakes.

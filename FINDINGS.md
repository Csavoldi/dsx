# DeepSeek-V4.1-Flash-1017 on the 2x3090 box (.102)

2026-10-17 (Pacific). Speed model: `roofline.py` (edit and rerun). Engine plan: `DESIGN.md`.

## Short answer

- It **is MoE**: 552B backbone, 16B active per token, 384 experts (6 used per token), plus 196B of Engram lookup tables.
- **Strata can't run it.** Strata is a Qwen3.8-Flash-Next-only engine. But a Strata fork, **Xolotl**
  ([Indras-Mirror/Strata-DSV4.1](https://github.com/Indras-Mirror/Strata-DSV4.1)), runs V4.1 on one RTX 4090 +
  90 GB RAM at **~26 tok/s decode, ~250 tok/s prefill** (its author's measurements).
- Catches with Xolotl: single GPU only, 3090 (sm_86) untested, its only quant is the **September V4.1, uncensored**
  (not 1017), and the 26 tok/s needs **biased routing** (it changes which experts the model picks; with exact routing it measured 3.5 tok/s).
- **No 1017 quant fits this box yet.** The 1017 GGUFs are 264 GB and up and need a llama.cpp fork.
- Physics of this box: the ceiling is **~17-19 tok/s with exact routing, ~25 biased**, x1.5-2 with speculative decoding.
  Prefill ceiling is ~1,500-3,700 tok/s; today's engines reach 250. Section 6.

## 1. The model

Post: [@deepseek_ai, Oct 17 1:35 AM](https://x.com/deepseek_ai/status/2111375640717033777).
Weights: [deepseek-ai/DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017).

| | |
|---|---|
| Type | MoE: 384 routed experts + 1 shared, 6 routed per token |
| Size | 552B backbone + 196B Engram tables + vision + 3 draft (DSpark/MTP) layers |
| Active | 16B per token decoding, 8B prefill |
| Layers | 40 (20 "encoder" + 20 "decoder"), hidden 5120, expert width 2304 |
| Attention | sparse (CSA2 + indexers), 128-token window, FP4 KV: **890 bytes/token** (128K = 117 MB) |
| Context / license | 1M / MIT |
| Files | 510 GB: dense FP8, experts packed FP4, Engram FP8. No Jinja chat template |

- **1017 vs V4.1 (Oct 16):** identical config and size, but every weight shard changed, Engram tables included.
  V4.1 quants run the same code but are not 1017.
- Not the same as **V4-Flash (0731)**: that one is ~284B with 256 experts, and plenty of engines already run it.

Where the bytes are:

| Part | Params | Where it goes |
|---|---:|---|
| Routed experts | 544B (15,360 x 35.4M) | quantized; split across VRAM / RAM / NVMe |
| Dense (attention, shared expert, routers, embed/head) | ~8B | GPU |
| Engram tables | 196B, 2 x 384M rows | NVMe, mmap: 48 random 256-byte rows per token |
| Draft layers | ~14B | optional, GPU |

## 2. The box (.102)

| | |
|---|---|
| Board / CPU | MSI PRO Z690-A DDR4, i7-12700K (8P+4E), AVX2 + AVX-VNNI, **no AVX-512** |
| RAM | 4 x 32 GB DDR4-3200, dual channel: **51 GB/s peak, ~40 real**. This is the bottleneck |
| GPU0 | 3090, CPU slot, PCIe 4.0 x16 (~25 GB/s) |
| GPU1 | 3090, **chipset slot, PCIe 3.0 x4 (~3.5 GB/s)**. Fine for holding experts, bad for streaming them |
| Disk | SN850X 4 TB (/home, 2.1 TB free). The second NVMe is the Windows (NTFS) disk |
| Running now | Strata (Qwen3.8-Flash-Next IQ3_XXS uncensored): both GPUs full, 56 GB RSS |

Anything that runs DeepSeek here needs Strata stopped while it runs: the RAM and VRAM don't fit both.

## 3. Strata and its forks

- [Niko1221/Strata](https://github.com/Niko1221/Strata) v0.1.44.1 in `~/ai/Strata`. Hand-written kernels for one
  model. The tricks: hot experts cached in VRAM, all experts pinned in RAM and computed by the CPU in place, n-gram table on SSD, MTP drafting.
- Its author on DeepSeek: "a new loader, packer, kernels and drafter. Not on the roadmap" ([#479](https://github.com/Niko1221/Strata/issues/479)).
- **Xolotl** ([Strata-DSV4.1](https://github.com/Indras-Mirror/Strata-DSV4.1)), V4.1:
  - Tiers: VRAM cache (~1,300-1,600 experts), pinned RAM arena (60 GiB), the rest read from NVMe. Its own GPTQ-style "GSQ" quant: ternary / 2-bit / 3-bit per expert, 152 GiB.
  - Measured on 4090 + 5700X + 90 GB DDR4: decode 25.5-26.3 tok/s (agent sessions 22.7-28.2), prefill 244-288 tok/s.
  - Quality: GSQ top-1 agreement 0.674 vs the FP8 model's 0.701. The routing bias raises code perplexity 2.38 to 2.87.
  - **One GPU only.** The author says sm_86 "should build", untested, and multi-GPU is not planned
    ([#1](https://github.com/Indras-Mirror/Strata-DSV4.1/issues/1)). A 1017 + multi-GPU request is open ([#2](https://github.com/Indras-Mirror/Strata-DSV4.1/issues/2)).
  - Quant: [IndrasMirror/DeepSeek-V4.1-Flash-GSQ-Uncensored](https://huggingface.co/IndrasMirror/DeepSeek-V4.1-Flash-GSQ-Uncensored),
    163 GB, plus the 203 GB Engram shards fetched separately. It is quantized from dealignai's abliterated V4.1.
- [PolyStrata](https://github.com/VecSzn/PolyStrata): **V4-Flash (0731) only.** UD-IQ2_XXS (91 GB): 48-56 tok/s on a 5090 + 120 GB.
  A 24 GB card + 128 GB box ran UD-IQ3_XXS at 27-31 tok/s.

## 4. Runtimes for V4.1 / 1017

| Runtime | Status |
|---|---|
| llama.cpp mainline | **No.** [PR #28696](https://github.com/ggml-org/llama.cpp/pull/28696) open since Sep 10. Forks run it (vcruz305, smalinin). Known bug: past ~1K tokens, layers that reuse indices attend densely (wrong but fluent output) |
| ik_llama.cpp | **Merged** ([#2455](https://github.com/ikawrakow/ik_llama.cpp/pull/2455), Sep 21) + DSpark, vision. Open: long-context bug [#2683](https://github.com/ikawrakow/ik_llama.cpp/issues/2683) |
| vLLM / SGLang | Day-0, but **no Ampere path** ([vLLM #57144](https://github.com/vllm-project/vllm/issues/57144)) |
| ktransformers | Not upstream; one community recipe on dual EPYC |
| ExLlamaV3 | PR #449 open. EXL3 quants are served through a vLLM plugin today |
| Xolotl | Yes, one GPU, see above |

## 5. Quants

| Quant | Size | Runtime | Ran? |
|---|---:|---|---|
| **1017** mrfakename GGUF Q2_K / Q3_K_M / Q4_K_M (mondk = byte-identical mirror) | 265 / 347 / 445 GB | vcruz305 llama.cpp fork | no reports |
| **1017** EXL3 2.9 / 3.0 / 3.5 bpw (LibertAIDAI, PixelML) | 211-258 GB + 203 GB Engram | vLLM + EXL3 plugin | perplexity only |
| **1017** Lucebox ROCMFP | 383-405 GB | own engine | quality only |
| **1017** MLX (oQ2e-oQ4e, Q9, LSQ3) | 177-452 GB | Mac | Q9: 15.9 tok/s on M3 Ultra |
| V4.1 GSQ-Uncensored (Xolotl) | 163 + 203 GB | Xolotl | **yes, 26.8 tok/s on a 4090 + 90 GB** |
| V4.1 vcruz305 GGUF Q2_K / Q3_K_M | 265 / 347 GB | ik_llama / fork | yes: ik_llama author, A6000 + 3090 + 251 GB, Q3_K_M ~21 tok/s |
| V4.1 EXL3 3.0 (0xSero offload) | ~200 GB host | own | yes: one 3090 + 503 GB 8-channel, 16-21 tok/s |

None of the big quant makers (unsloth, bartowski, ubergarm) has posted V4.1 yet. The 1017 GGUFs keep experts near
4 bits, and at 265 GB they would page from NVMe on a 128 GB box: expect a few tok/s.

## 6. First principles: what limits speed on this box

**Decode** (one token = one pass over 40 layers):

1. The dense half (attention, shared expert, routers) runs on GPU0 every token: **~20 ms**. Xolotl measured 19 ms on a 4090; a 3090 has about the same bandwidth. That alone caps decode at ~50 passes/s.
2. Routed experts: 6 per layer = 8.5B params per token = **~2.0-2.5 GB per token** at ~2 bits. Each byte comes from wherever that expert lives:

   | Tier | Speed | Who computes |
   |---|---:|---|
   | VRAM (936 GB/s) | free | GPU |
   | RAM (~40 GB/s) | ~50 ms per GB | CPU in place. Copying to the GPU is slower: 25 GB/s |
   | NVMe (~6 GB/s) | ~170 ms per GB | read, then CPU |

3. **DeepSeek balances its experts on purpose**, so with exact routing each tier serves traffic roughly in proportion to how much of the model it holds. Caching hot experts helps less than it did for Qwen. This fits Xolotl's numbers: 3.5 tok/s exact vs 26 biased on a box where half the experts live on NVMe.
4. So the rule is: **keep zero experts on NVMe, put as many as possible in VRAM, and the rest in RAM.**
   With Strata stopped: ~32 GB of experts in VRAM (both GPUs), ~95 GB pinned in RAM, so **~127 GB total → ~1.9 bits/weight on average.**

| Experts avg bpw | Pool | GPUs | Exact routing | Biased (60% VRAM hits) |
|---:|---:|---:|---:|---:|
| 2.4 (Xolotl GSQ) | 163 GB | 1 | 5.6 tok/s | 11 |
| 2.4 | 163 GB | 2 | 8.0 | 14 |
| 2.1 | 143 GB | 2 | 13.9 | 21 |
| **1.9** | **129 GB** | **2** | **17.0** | **25** |
| 1.75 (ternary) | 119 GB | 2 | 18.5 | 26 |

(Single-token passes from `roofline.py`, no speculative decoding. The model is on the pessimistic side for the 1-GPU
row: Xolotl gets 26 there with biased routing, using its NVMe penalty as well.)

5. **The second 3090 is worth ~2x** at the same quant: it holds 21 GB more experts. Its x4 link doesn't matter for decode, because only activations (~40 KB per layer) cross it.
6. **Speculative decoding** (DSpark: 3 draft layers, blocks of 5) verifies several tokens in one pass. Each pass reads more experts, but the dense 20 ms is shared. Expect ~1.5-2x on code and agent text, so **~25-40 tok/s is the realistic target**.
7. CPU risk: 8 P-cores must decode ~1.5 GB of 2-bit weights per token. With i-quant style lookup tables, AVX2 may run out of compute before RAM runs out of bandwidth. **Use quant formats the CPU unpacks cheaply** (Q2_0-style, ternary). Measure this first.

**Long agentic contexts:**

- KV is tiny (117 MB at 128K, 0.9 GB at 1M), so long context barely slows decode. Per-token attention reads a fixed top-512 rows plus the window. The indexer that picks them scales with context but is small.
- **Never re-read the prompt.** Each agent turn appends a 1-5K tool result to a 30-200K context. Keep every session's KV (a 200K session is 180 MB, so dozens fit in RAM, and they can spill to disk), and replay only the last 128 window tokens.
- A typical turn of 2K tokens in and 400 out: at 1,000 tok/s prefill and 25 tok/s decode that is 2 s + 16 s. **Decode dominates agent turns.**
- **Prefill ceiling:** a long prompt is processed in chunks. Each chunk must stream every non-resident expert over GPU0's PCIe once (~100 GB / 25 GB/s = 4 s), so 8K chunks give ~2,000 tok/s and 16K chunks ~3,700 with both GPUs computing. Xolotl's 250 and llama.cpp's ~50-100 are far below this, so engineering can still win a lot here.
- Long prompts lose some speed to the indexer (O(n²) in total), roughly +4 s of compute at 128K.

Cheap hardware levers (lxe's call, none done): DDR4-3600 XMP if the DIMMs allow it (+12% bandwidth). GPU1 can't get a better slot on this board.

## 7. Options

| Option | Model | Download | Expected | Effort / risk |
|---|---|---:|---|---|
| A. Xolotl + GSQ-Uncensored, GPU0 only | V4.1 (Sept), abliterated | 163 + 203 GB | ~20-26 tok/s biased, ~200-250 prefill | sm_86 untested; 1 GPU; not 1017 |
| B. ik_llama.cpp, both GPUs, own ~2-bit 1017 quant | 1017 | 508 GB source + ~140 GB output | ~8-15 tok/s, ~100-300 prefill | needs an imatrix run and a quantize pass; long-context bug open |
| C. V4-Flash-0731 UD-IQ2_XXS on PolyStrata | V4-Flash (older, 284B) | 91 GB | 27-50 tok/s, ~1,000 prefill | proven on 24 GB + 128 GB boxes; not V4.1 |
| D. Custom Rust engine for this box + 1017 | 1017 | 510 GB FP8 | target 25-40 tok/s, 1,000+ prefill | weeks; see DESIGN.md |

## 8. Recommendation

1. **Today, to try it:** Option A on GPU0 with Strata stopped. 366 GB on /home (2.1 TB free), plus 203 GB temporary
   while the Engram tables are extracted. It gives us real routing and tier numbers on this exact box, which the custom engine needs anyway.
2. **For 1017 and max speed:** Option D, a Rust engine built around the section 6 rules: both GPUs, no NVMe expert tier,
   CPU-friendly ~1.9 bpw experts, persistent session KV, DSpark. Start with the measurements in DESIGN.md phase 0 (an hour, read-only, no downloads).
3. Watch [Xolotl #2](https://github.com/Indras-Mirror/Strata-DSV4.1/issues/2) and ik_llama for a 1017 quant. Someone may publish a GSQ-1017.

Nothing has been downloaded or started on .102. Downloads need lxe's go-ahead.

## 9. Update, 2026-10-17 1:45 PM

- Fresh sweep: still no 1017 quant runnable on this box. [Baekpica 1017 VQ GGUF](https://huggingface.co/Baekpica/DeepSeek-V4.1-Flash-1017-VQ-GGUF)
  (~114 GB, ~1.6 bpw VQ experts) needs its own DGX Spark engine with the whole model resident. unsloth created an empty 1017 repo at 1:36 PM.
- 1017 = September layout with new weights; the Engram shards and vision changed too (September Engram loads silently and mismatches).
- Quality yardstick for our quant: Lucebox's 1017 eval set with teacher log-probs (`eval/` in
  [Lucebox ROCMFP repo](https://huggingface.co/Lucebox/DeepSeek-V4.1-Flash-1017-ROCMFP-GGUF)): 2.5 bpw fp2s experts = KL 0.253, top-1 78.7%; 3-bit down = KL 0.195.
- ik_llama #2683: V4.1 loses context past ~200K, blamed on f16 indexer math. Our indexer runs in f32.
- Xolotl: sm_86 build works on .102 (built 2026-10-17). Author: multi-GPU not planned; a 1017 GSQ is planned, no date.
- Downloads on .102 via aria2 at 1.9 Gbps (capped at 2 Gbps per lxe), 1017 first.

## 10. Progress, 2026-10-17 3 PM

- Downloaded and checksum-verified: 1017 FP8/FP4 checkpoint (510 GB), the V4.1 GSQ baseline is still downloading.
- **Engine (`engine/`, Rust + CUDA) runs 1017 end to end.** The CPU reference (exact model.py math on the original
  FP8/FP4 weights) answers "Paris" to "What is the capital of France?". The GPU path matches it to bf16 rounding; the
  only differences come from router near-ties (expert scores equal to 4 decimals, so rounding noise picks either).
- Built: GPU decode (dense half on GPU0), CPU expert kernels at RAM speed, GPU expert stores on GPU0 and GPU1,
  a sync-free per-layer handoff between them through mapped memory (5-8 us per hop), GPU top-k for the sparse indexer,
  prefill (encoder over every token, decoder over the exact dependency cone only, tensor-core GEMMs), session snapshots
  for prefix reuse, the DSpark draft stack (reference), and an optional routing bias (off by default).
- Quant: a round-to-nearest Q2_0 1017 export is converting (a stand-in for engine work). The calibrated quant
  (Xolotl's GSQ pipeline, agentic calibration from lxe's sessions + antirez's prompts) is calibrating on GPU1, ~8 min/layer.
- Expert pool budget from the measurements: GPU0 ~9-10 GB spare after the dense half and prefill buffers, GPU1 ~23 GB,
  RAM arena ~100 GB, so **~133-138 GB of experts (~2.0 bpw)** with nothing on NVMe.

## 11. Expert usage is skewed on agentic text (calibration, 2026-10-17 4:45 PM)

From the 1017 calibration pass (128 x 2048 tokens: lxe's agent sessions + antirez's agent/code prompts), tokens
routed per expert, layers 2-6:

| | share of routed tokens |
|---|---:|
| hottest 10% of experts | 32-41% |
| hottest 25% | 55-61% |
| hottest 50% | ~80% |
| coldest 10% | 1.0-1.7% |

The section 6 model assumed balanced routing (traffic share = capacity share). It isn't: the hottest 25% of experts
(~32 GB at ~2 bpw, what fits in both GPUs) serve ~58% of routed traffic. RAM then carries ~42%: ~0.84 GB/token
= ~21 ms + ~10 ms dense, so **~30 tok/s with exact routing**, before speculative decoding. Placement must rank
experts by measured hotness (GPU1 then GPU0 get the hottest).

## 12. DSpark (built-in speculative drafts), first measurement on 1017

CPU reference, 60-token agent prompt, 40 greedy tokens, 5-token drafts: accepted drafts per step histogram
[0:14, 1:10, 2:6, 3:3, 4:2, 5:0], **mean 1.11** (2.1 tokens per verify pass). Expert union per layer over windows of
consecutive tokens: 1 token 6.0, 2 tokens 9.8, 4 tokens 15.9, **6 tokens 20.9** (3.5x the bytes of one token).
So a verify pass would cost ~3.5x a decode step's expert traffic for ~2.1 tokens: a net loss on this box as measured.
Caveat: the prompt ended mid-text and the model emitted EOS first, so this understates a real assistant turn.
Re-measure on proper agent turns before deciding; parked for now.

## 13. Engine status and measurements, 2026-10-17 6:15 PM

Correctness (V4.1 GSQ quant as a stand-in, same architecture): GPU decode matches the CPU reference except at router
near-ties; prefill vs token-by-token decode gives identical continuations; the API server answers OpenAI and Anthropic
requests, including a structured tool call.

| Optimization (decode, dense half) | before | after |
|---|---:|---:|
| Dense Q8 GEMV (repacked int8 rows + scales, 16-byte loads), 3090 | 22.6 ms/token | 10.6 ms |
| hc mixes (24 blocks + warp Sinkhorn), 3090 | 6.1 ms | ~1.1 ms |
| Top-token selection (partial select, not a full 129K sort) | ~40 ms | ~0 |
| Engram rows (helper thread, plain threads instead of rayon next to the spinning pool), 5090 | 6.2 ms | 0.6 ms |
| Dense-only decode, 5090, end to end | 15 tok/s | **73 tok/s** |

VRAM-resident experts: repacked layouts + int8 mma.sync, 1.0-1.3 TB/s on the 5090 at T = 1 and T = 6 (a verify pass
costs the same as one token for GPU experts); were 100-170 GB/s on the 3090.

Prefill, 4096-token chunk without routed experts, 5090: 3.1 s (attention kernel 0.53, indexer 0.18, projections ~0.6,
Engram 0.67 warm, shared expert 0.4). Engram rows are deduplicated per chunk (3x fewer reads); a tensor-core prefill
attention kernel is in progress.

## 14. First full runs of our 1017 quant on .102, 2026-10-17 evening

Quant: GSQ pipeline on 1017, 2.03 bpw experts (edge layers 2.25), 148 GB file, finished 8:43 PM. Placement: 34 GB
of the hottest experts on the GPUs (by our calibration ranking), all other experts in a 104 GB RAM arena, nothing on NVMe.

**Speed, 3000-token agent prompt, exact routing:**

| Step | Decode | Notes |
|---|---:|---|
| First run | 12.5 tok/s (80 ms) | |
| One CPU per spinning thread | 20.3 tok/s (49 ms) | main thread, CPU-expert coordinator, Engram and CUDA helper threads were all on CPU 16; the coordinator saw routed picks up to 1.5 ms late |
| Tiled split-K decode attention | **31.5 tok/s (32 ms)** | 276 -> 21 us per layer |

Where the 32 ms go now: dense half on GPU0 ~14 ms (q projection 3.2, o projection 4.6, Engram 1.4, hc 2), CPU
experts 14 ms (the GPU waits 12 ms for them), shared expert + router + GPU experts ~5 ms. GPU1's experts run in
parallel and cost ~0.

Prefill: 340 tok/s on 3000 tokens (not tuned yet). Verify pass of 6 tokens: 157 ms (26 ms per token if all are
accepted), so DSpark breaks even only above ~1.2 accepted drafts per pass.

**Quality, Lucebox's 1017 eval (8 x 1024 tokens, teacher top-256):**

| | KL | top-1 | ppl |
|---|---:|---:|---:|
| Teacher (FP8 1017) | 0 | 100% | 8.56 |
| Lucebox 2.5 bpw | 0.253 | 78.7% | 9.78 |
| **Ours, 2.03 bpw, exact routing** | **0.462** | **71.5%** | **11.66** |
| Ours, routing bias 0.05 | 0.464 | 71.4% | 11.71 |
| Ours, routing bias 0.1 | 0.478 | 70.6% | 11.97 |
| Ours, routing bias 0.2 | 0.526 | 69.5% | 12.53 |

Routing bias buys little (measured before the attention fix: 20.3 tok/s exact, 21.5 at 0.05, 24.4 at 0.2) and costs
KL; leave it off. The quality gap to Lucebox's 2.5 bpw is large: 2.03 bpw is what fits in
RAM + VRAM without NVMe, so the next quality step is better bit allocation or spilling the coldest experts to NVMe
to afford ~2.3-2.5 bpw.

## 15. colibri (JustVugg/colibri), reference research, 2026-10-18

Pure-C MoE engines (Apache-2.0, ~43K stars) that keep the dense part in RAM and stream routed experts from disk,
with a GPU tier for hot experts. It has a DeepSeek V4.1 Flash engine (docs/deepseek-v41.md): 0.21-0.56 tok/s on
CPU servers, reading the official FP4 experts from NVMe (4.5 GB of experts per token at FP4). Different niche from
ds41 (low-RAM machines, unquantized weights); useful for us:

- **Index-key slot.** In the original V4.1 `model.py` a ratio-2 KV-source layer republished its index keys only
  when it completed a group, so on every other decode step layers 2/8/14 scored against layer 20's keys; colibri
  reproduces that by default. **The 1017 `model.py` republishes unconditionally** (`if self.owns_k:
  shared_attn.index_k = self.k_cache`), which is what ds41's reference and GPU paths do. Checked: no change needed.
- **DSpark measured on CPU:** a draft round costs ~1/5 of a main token (window-only attention, 128 experts, top-3);
  +17% with an acceptance guard (stop drafting below 60% acceptance); without the guard the gain mostly vanishes.
  Exact speculative decoding needs rollback of three modulo-addressed states a draft can clobber: the window ring,
  the compressor's half-filled group slots, and the published index-key slot. ds41's verify pass has no rollback yet;
  this is the list to implement.
- **Disk tier (for v2's NVMe spill):** resolve a step's experts to cache slots first (hits stamped, victims
  reserved), then read the misses together at queue depth ~8 (6.8 GB/s on two striped PM9A1; deeper steals cores);
  one `pread` per expert for all three matrices; router lookahead prefetch one layer ahead (71.6% predictable on
  GLM-5.2); O_DIRECT measured per drive; a mirror on a second NVMe gave +37.5%.
- **Learned placement:** a per-user hot set updated every turn (`.coli_usage`) on top of per-layer LRU; ds41's
  ranking is static from calibration. Worth doing with the RAM arena once part of the pool lives on NVMe.
- **Engram:** ~13 KB per token, at most 48 rows of 264 bytes per table; streaming it is the right call (as ds41 does).

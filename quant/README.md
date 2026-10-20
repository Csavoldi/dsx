# Quantization pipeline

How [DeepSeek-V4.1-Flash-1017-dsx-2bit](https://huggingface.co/lxe/DeepSeek-V4.1-Flash-1017-dsx-2bit) was made. Built on the V4.1 tools of
[Xolotl](https://github.com/Indras-Mirror/Xolotl) (MIT, Niko1221 and the Strata contributors), commit `7375e305`.

The published build is v4b. Each build changed one thing; its script runs it end to end on one machine (paths in the
scripts are the build machine's):

| Build | What changed | Script |
|---|---|---|
| v1 | per-layer tier allocation at 2.0 bpw | `calib_all.sh` (calibration), `quant_chain.sh` (GPTQ one layer behind it), `finish.sh` (GGUF) |
| v2 | one bit budget across layers 2-36 | `quant_chain_joint.sh`, `finish_v2.sh` |
| v3a | calibration text with the full model's reasoning traces (`traces/collect.py`), v2's tiers | `run_v3a.sh` (`quant_chain_v3.sh`) |
| v4 | experts quantized in a 128-block Hadamard-rotated basis (`v41_quant.py --rot`) | `run_v4.sh` |
| v4b | least-squares scale refit after GPTQ (`--refit`) | `run_v4b.sh` |

| File | |
|---|---|
| `xolotl-v41.patch` | our changes to Xolotl's `tools/v41`: load the 1017 checkpoint, preload the Engram rows a batch needs in one sequential sweep, a `--budget-bpw` allocator, GPU RTN option, corpus from local agent sessions |
| `v41/` | the calibration and quantization scripts (Xolotl-derived, with our changes: native GPTQ, `--rot`, `--refit`) |
| `native/` | CUDA kernels for GPTQ and the scale search (below) |
| `traces/collect.py` | the full model's reasoning traces for the calibration text (via OpenRouter) |
| `make_rank.py` | expert hotness from the calibration routing counts: `expert-rank.json`, used to put the hottest experts in VRAM |
| `experiments/quant_variants.py` | one-layer A/B of quantization variants on dumped experts (`V41_DUMP`) |
| `guard.sh`, `watchdog.sh` | every job in a memory-capped cgroup; a watchdog stops jobs on GPU Xid errors, low RAM or high temperatures |

Calibration text (v3a on): 160 x 2048 tokens of coding-agent sessions, antirez's agent and code prompts, and 53
sequences of the full model's reasoning traces on Omni-MATH and SuperGPQA questions (none from the evaluated rows).
v4b's GPTQ pass takes ~3 h on one RTX 3090, plus GGUF assembly and the eval.

```bash
git clone https://github.com/Indras-Mirror/Xolotl && cd Xolotl && git checkout 7375e305
git apply /path/to/dsx/quant/xolotl-v41.patch
# then copy quant/v41/*.py over tools/v41 and adapt the paths in run_v4b.sh
```

## Native quantizer (faster runs)

`native/qkern.cu` (+ `native/build.sh`) replaces the two hot loops with CUDA kernels, batched over experts:
GPTQ (a warp per row, the 128-column block in registers, the H^-1 block in shared memory; the trailing update one
batched GEMM) and the static-group scale search (one pass, every candidate per group in registers).
`v41/v41_native.py` (ours) wraps them; `v41/v41_quant.py` + `v41/v41_calib.py` (Xolotl's files with our changes) queue
experts (8 or 160K routed tokens per batch) and quantize each batch in one call. `V41_NATIVE=0` keeps the old path.

| per expert (RTX 5090) | before | native |
|---|---:|---:|
| GPTQ gate+up / down | 73 / 31 ms | 6.9 / 2.5 ms |
| scale search | 3-49 ms | 0.7-9.7 ms |

Integer codes match the old path to >= 99.999% (rounding ties under FMA contraction), same H-weighted error.

## Credits

- `v41/v41_alloc.py`, `v41_calib.py`, `v41_formats.py`, `v41_gptq.py`, `v41_gptq_graph.py`, `v41_quant.py` are derived from
  [Xolotl](https://github.com/Indras-Mirror/Xolotl)'s `tools/v41` (commit `7375e305`), MIT License, Copyright (c) 2026
  Niko1221 and the Strata contributors ([`v41/LICENSE-Xolotl`](v41/LICENSE-Xolotl)); our changes are marked in the
  files and in `xolotl-v41.patch`.
- GPTQ: Frantar, Ashkboos, Hoefler, Alistarh, *GPTQ: Accurate Post-Training Quantization for Generative Pre-trained
  Transformers* ([arXiv 2210.17323](https://arxiv.org/abs/2210.17323)).
- Hadamard rotation of expert inputs (`--rot`): incoherence processing as in QuIP# (Tseng et al.,
  [arXiv 2402.04396](https://arxiv.org/abs/2402.04396)) and QuaRot (Ashkboos et al., [arXiv 2404.00456](https://arxiv.org/abs/2404.00456)).
- The calibration forward runs DeepSeek's reference `inference/model.py` and kernels from the
  [DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) checkpoint (MIT).

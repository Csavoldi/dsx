# Quantization pipeline

How `DeepSeek-V4.1-Flash-1017-ds41-2bit.gguf` was made. Built on the V4.1 tools of
[Xolotl](https://github.com/Indras-Mirror/Xolotl) (MIT, Niko1221 and the Strata contributors), commit `7375e305`.

| File | |
|---|---|
| `xolotl-v41.patch` | our changes to Xolotl's `tools/v41`: load the 1017 checkpoint, preload the Engram rows a batch needs in one sequential sweep, a `--budget-bpw` allocator, GPU RTN option, corpus from local agent sessions |
| `calib_all.sh` | layer-streaming calibration on GPU1: DeepSeek's `model.py`, one layer at a time from the FP8/FP4 checkpoint (it does not fit in RAM), 128 x 2048 tokens |
| `quant_chain.sh` | one layer behind it on GPU0: tier allocation at 2.0 bits/weight (layers 0, 1, 37-39 at 2.25), then GPTQ into PTQ1_0 (1.75) / Q2_0 (2.25) / Q3_K (3.44) |
| `finish.sh` | assembles the GGUF (dense weights Q8_0 from the checkpoint, experts from the chain, Engram left in the checkpoint) and the expert ranking |
| `make_rank.py` | expert hotness from the calibration routing counts: `expert-rank.json`, used to put the hottest experts in VRAM |
| `guard.sh`, `watchdog.sh` | every job in a memory-capped cgroup; a watchdog stops jobs on GPU Xid errors, low RAM or high temperatures |

Calibration text: the author's own coding-agent sessions plus antirez's agent/code prompts (agentic, long-context).
About 5 hours on 2x RTX 3090 for the whole model.

```bash
git clone https://github.com/Indras-Mirror/Xolotl && cd Xolotl && git checkout 7375e305
git apply /path/to/ds41/quant/xolotl-v41.patch
# then, from tools/v41 with the paths in the scripts adjusted:
./calib_all.sh & ./quant_chain.sh & ./finish.sh
```

## Native quantizer (faster runs)

`native/qkern.cu` (+ `native/build.sh`) replaces the two hot loops with CUDA kernels, batched over experts:
GPTQ (a warp per row, the 128-column block in registers, the H^-1 block in shared memory; the trailing update one
batched GEMM) and the static-group scale search (one pass, every candidate per group in registers).
`v41/v41_native.py` wraps them; `v41/v41_quant.py` + `v41/v41_calib.py` (Xolotl's files with our changes) queue
experts (8 or 160K routed tokens per batch) and quantize each batch in one call. `V41_NATIVE=0` keeps the old path.

| per expert (RTX 5090) | before | native |
|---|---:|---:|
| GPTQ gate+up / down | 73 / 31 ms | 6.9 / 2.5 ms |
| scale search | 3-49 ms | 0.7-9.7 ms |

Integer codes match the old path to >= 99.999% (rounding ties under FMA contraction), same H-weighted error.

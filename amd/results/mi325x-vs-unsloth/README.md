# One MI325X: dsx MXFP4 + 3-bit vs Unsloth's GGUF quants (October 19, 2026)

Same box for every row: one AMD Instinct MI325X (DigitalOcean `gpu-mi325x1-256gb`, 256 GiB HBM3E, ROCm 7.14).

| | Unsloth UD-IQ1_S | Unsloth UD-Q2_K_XL | dsx MXFP4 + 3-bit |
|---|---:|---:|---:|
| Runtime | llama.cpp PR #30817 (b72050d), HIP gfx942 | same | dsx-amd |
| Total size | 233 GB | 335 GB | 270 GB + official Engram (203 GB) |
| Expert/dense weights (without Engram) | ~122 GB | ~224 GB | 270 GB |
| Engram table | 110.6 GB IQ4_NL (CPU, mmap) | 110.6 GB IQ4_NL (CPU, mmap) | official FP8 (NVMe + page cache) |
| Lucebox KL / top-1 / ppl (FP8: ppl 8.56) | 0.361 / 75.1% / 10.76 | 0.114 / 85.7% / 9.04 | 0.049 / 90.6% / 8.71 |
| Prefill, 3,000-token prompt | 554 tok/s | not measured | 1,488 tok/s (961 on a cold Engram cache) |
| Decode after it | 54 tok/s (56 after a short prompt) | not measured | 115.6 tok/s |

For reference, dsx's NVIDIA 2-bit build (148 GB, 2x RTX 3090): KL 0.395 / top-1 74.1% / ppl 10.93.

## Overthinking: S01 / S02, temperature 1.0, top_p 1.0, reasoning effort low

Pocket Model v26 rows (Omni-MATH): S01 answer 480/49, S02 answer 500. Completion tokens (reasoning + answer),
"Wait" count in the trace.

| | S01 runs | S02 runs |
|---|---|---|
| Unsloth UD-IQ1_S | 10,565 / 19,855 / 13,080 (median 13.1K); Wait 19 / 35 / 31; 3/3 correct | 33,922 / 29,765 / 21,215 (median 29.8K); Wait 60 / 57 / 33; 3/3 correct |
| dsx MXFP4 + 3-bit | 5,239 / 7,956 / 10,955 (median 8.0K); Wait 6 / 7 / 19; 3/3 correct | 17,866 / 16,374 / 22,216 (median 17.9K); Wait 27 / 21 / 29; 3/3 correct |
| Hosted full model (8 runs, from Pocket Model) | median 9.4K (5.7-17.4K) | median 10.7K (7.5-13.9K) |

## How

- Quality: `amd/tools/lucebox-eval/lucebox-eval.cpp` (built inside llama.cpp's examples) feeds Lucebox's exact token ids
  (8 x 1024) and computes the same per-position KL(teacher || model) over the FP8 teacher's top-256, top-1 agreement
  and perplexity as `dsx-amd eval`.
- Speed: llama-server `/completion` with the token ids of the same 3,000-token prompt, 256 forced tokens; dsx-amd
  `dev` with `pbench @prompt3000.ids`.
- Overthinking: `amd/tools/overthink_probe.py` against each server's OpenAI API. llama.cpp got
  `chat_template_kwargs: {"reasoning_effort": 50}` (Unsloth's "low"), dsx `reasoning_effort: "low"`. Not checked:
  that both templates render the same prompt text for "low".
- llama.cpp flags (Unsloth's guide): `-ngl 999 -fa on --load-mode mmap --lazy-mode on -ot per_layer_token_embd.weight=CPU`.
  Without `-ot` the Engram table still stays on the CPU; speed was the same.
- UD-Q2_K_XL: the eval took 2 hours and the server did not finish loading in 15 minutes, so its speed and S01/S02 runs
  were dropped.

Raw data: `transcripts/` (one JSON response per run plus the per-run JSONL), `logs/` (evals, speed, server logs).

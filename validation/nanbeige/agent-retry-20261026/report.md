# Nanbeige live throughput and cache observations

Observed on 2026-10-26 during the configuration-retry Doom run. This monitor did not restart services, launch benchmarks, or modify model code. A tokenizer streaming failure ended the first phase; the parent repaired the decoder and restarted the server for a thinking-off phase. The backend finish code recorded Cancelled, but the parent reproduced a Unicode bytefallback invalid-prefix error as the cause, rather than a user cancellation.

## Observed throughput

The first seven completed requests generated at 40.23 tokens/s in aggregate, weighted by their backend decode duration and excluding each request's first token. The six short tool turns were 40.71–49.42 tokens/s. Their prompts ranged from 2,290 to 5,533 tokens. The final reasoning request started with 5,628 prompt tokens, generated 12,245 tokens before a tokenizer streaming error aborted the request, and averaged 39.57 tokens/s over 309.404 seconds. Its nominal final populated context was 17,873 tokens; this is not evidence for full 262,144-token populated context.

The previous 69.88 tokens/s benchmark used only 49–64 prompt tokens and stopped at 561–576 populated tokens. Those measurements do not establish an unexplained regression in the current longer-context run. Workload, sampling, speculative acceptance and attention paths differ.

The final request accepted 1,586 of 3,345 proposed lookup tokens (47.41%). Across all seven requests, acceptance was 49.59%. These values alone do not prove speculation speeds up execution: verification overhead and wasted rows still require paired lookup-off measurements.

## GPU activity

Nine samples, two seconds apart, from 16:39:55 through 16:40:11 UTC preceded the streaming abort. Median GPU activity was 97% (94–99%), power 411.49 W, temperature 70 C, SM clock 2,850 MHz, memory clock 13,801 MHz. The GPU was busy. Median nvidia-smi memory activity was 22%; that percentage measures memory-controller activity time and must not be interpreted as achieved bandwidth or occupancy.

Existing isolated Nsight Compute evidence in `validation/nanbeige/continue2-pf4-ncu.log` reports 43.08% DRAM throughput, 33.27% compute throughput, 54.75% scheduler cycles with no eligible warp, and 46.8% long-scoreboard stall contribution for one pf4 projection. This supports memory-latency hiding as an optimization direction. It does not characterize every kernel or the current whole request.

## Prefix and KV cache

Prefix reuse is visible in backend receipts: 2,519/2,759; 2,968/3,086; 3,279/3,434; 3,960/4,220; and 4,878/5,533 prompt tokens were reused from GPU state. After interruption, 5,529/5,628 tokens were reused via a pinned-RAM recent-KV snapshot. Including the first cold request, 85.84% of prompt tokens were cached. Prefill of 99 new tokens in the final request took 330 ms; this includes cache recovery and synchronization, so its 299.3 tokens/s should not be compared directly to a cold 3,968-token batch benchmark.

Source stores all historical KV in 3-bit form, while a 2,592-slot BF16 ring protects approximately 2,048 recent positions plus prefill and speculation guards. Rewinds larger than 504 positions restore the recent ring from a pinned snapshot or fall back to replay. A saved 4,096-token prefix audit forces 600 generated tokens and verifies that cached/retried/cold continuations produce identical eight tokens, including snapshot restoration. That is meaningful local recovery evidence, not exhaustive verification of arbitrary 256K histories.

The snapshot is 552,075,264 bytes and is taken after prompt prefill. Its copy and synchronization add fixed per-turn latency. Avoid interpreting prompt throughput on 100-token suffixes as long-batch capacity. Keeping a stable system prompt and tool schema preserves the observed reuse.

## Optimization proposals

1. Test context-dependent GQA partition sizes first. At position 2,048, production switches from per-head parallel decode to a GQA tensor-core kernel with 512-token partitions and 48 KiB shared memory. The existing RTX 5090 synthetic sweep measured 0.1668 ms per attention layer at 4,096 tokens with partition 512 versus 0.048264 ms with partition 128. Across 52 layers, the isolated difference is about 6.16 ms per step. If transferable, this could improve current 40–45 tokens/s by roughly 20–35%; it is an estimate, not a result. Smaller partitions add merge work and launches/CTAs, and may lose at long contexts. Production pads split counts to powers of two whereas the sweep uses exact counts. Verify warmed production-padded timing, independent attention output, graph keys across partition changes, speculative rewind and full-model quality. Consider 128 for short context, 256 at intermediate contexts, and retain 512 where measured best. Reject if whole-model timing does not improve.

2. Profile attention and projections separately after the agent is idle. The 2,048-token attention-path transition and power-of-two split thresholds deserve measurements at 1,024/2,048/4,096/8,192/16,384 populated tokens. Keep fixed sampling and output counts, report acceptance, and compare lookup off/on. This resolves how much of the 70-to-40 comparison comes from attention rather than speculation or client behavior. GPU activity alone cannot resolve it.

3. Improve pf4 latency hiding only after establishing its whole-step share. Existing profiling suggests register/warp balance, independent trellis loads and prefetch scheduling. Changes can add instructions or reduce occupancy, so require projection correctness plus end-to-end timing and quality drift. A previous coalescing experiment was already rejected; do not assume contiguous accesses alone solve the measured scoreboard stalls.

4. Consider device-side sampling after measuring CPU gaps. Current engine downloads 166,912 FP32 logits (667,648 bytes) per ordinary step, and several rows during verification, then constructs the top-p distribution on CPU. Moving sampling and acceptance to GPU could remove transfers and synchronizations but adds distribution/RNG compatibility work. Preserve temperature/top-p semantics, statistical distribution, stop tokens and deterministic tests. No gain estimate is justified without gap timing.

5. Consider asynchronous/reduced recent-ring snapshots if short agent turns dominate. The current full-ring snapshot is robust and demonstrated. Any attempt to avoid or delay copying must preserve cancellation recovery and the 504-token rewind guard. The maximum improvement is its measured share of time-to-first-token, which is not separated in the current receipt.

Raw observations are stored beside this report in initial-request-summary.json, journal-before-restart.txt, gpu-samples.jsonl, and gpu-active-phase-summary.json. No optimization gain is booked from this monitoring report.

## Thinking-off follow-up

After the parent repaired the Unicode bytefallback streaming decoder and restarted the server, a fresh thinking-off Pi attempt completed 54 tool requests. The run repeatedly alternated `echo okN` and tmux preview probes rather than writing game code. Its backend weighted decode rate was 94.29 tokens/s, draft acceptance 89.70%, and prompt cache reuse 98.27%. Prompt length started at 1,254 tokens and nominal populated context reached 13,404 tokens. This demonstrates prefix retention across tool turns and fast repetition with high lookup acceptance; it does not establish representative coding throughput or task completion.

The exact live `/metrics/progress` endpoint was checked against completed backend receipts. For a 120-token response at 5,925 prompt tokens, it reported 98.00 tokens/s while the backend reported 97.81; the small difference follows their slightly different timing boundaries. During queued/prefill phases the endpoint retains the previous response's generated count and TPS, so display code must use `phase` and `active` to distinguish historical rates from current streaming.

The first-phase finish code `Cancelled` was a decoder error, not an intentional user cancellation. The parent reproduced the tokenizer invalid-prefix exception on a malformed Unicode bytefallback run and replaced streaming decode with buffering that matches full-reference decode. The overthinking and the streaming abort are separate observations.

## First useful code output

The next narrowed milestone prompt produced a real 7,800-byte `game/doom.py` file. Its first write request used 585 prompt tokens and generated 2,465 output tokens in 47.808 seconds: **51.54 output tokens/s**, with cold prefill 1,529.4 tokens/s (382 ms). Nominal final context was 3,050 tokens. This response crosses the 2,048-token attention-path transition, unlike the earlier short-context 69.88 tokens/s benchmark. The monitor did not run the game or claim visual correctness. Subsequent short tool requests reused 3,050/3,117 and 3,247/3,296 prompt tokens, confirming continuation cache reuse after the write.

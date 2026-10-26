# Experimental GQA INT8 QK/PV validation

`DSX_NB_GQA_I8PV=1` selects an optional 24 KiB shared-memory GQA decode kernel, while the default BF16 QK/high-low PV path remains unchanged. Four query positions map to up to 28 rows across seven heads sharing each KV head. Production partitions remain 512 tokens; algorithm-specific graph keys prevent cache collisions.

CUDA/Rust release CLI compilation passed. Ten independent f64 attention audits passed: aligned position 4,096 with 65 rows and unaligned position 4,097 with 512 rows, plus 1/4/8/9 rows at both positions. Both exact split counts and production power-of-two padding were checked. Worst sampled relative RMS error was 1.8081% for INT8 versus 0.1694% for default GQA; maximum absolute error was 0.015861 versus 0.001951. These numbers establish additional attention approximation error, not losslessness or model-quality preservation.

## Isolated production-padded timing

All rows below include split merge, 512-token partitions, eight warmup calls and seven samples of four launches. Values are medians in milliseconds for one query row.

| Populated synthetic position | Default GQA | INT8 GQA | Isolated speedup |
| --- | ---: | ---: | ---: |
| 4,096 | 0.16734 | 0.09622 | 1.74x |
| 32,768 | 0.34708 | 0.20107 | 1.73x |
| 65,536 | 0.62255 | 0.31002 | 2.01x |
| 131,072 | 1.74741 | 0.75234 | 2.32x |
| 262,135 | 3.36527 | 1.24938 | 2.69x |

Raw receipts contain row counts 1/4/8 and partitions 128/256/512/1024/2048. The benchmark selector also supports `DSX_NB_ATTN_PADDED=1`; default benchmark split geometry remains exact ceil division for compatibility with earlier receipts. The independent validator accepts `DSX_NB_ATTN_AUDIT_ROWS` for focused production row counts.

## Limits and next validation

The benchmark uses synthetic random packed KV, constant query vectors, constant BF16 recent KV and no loaded model weights. GPU clocks were not locked and old/new runs were sequential in that order. The independent f64 accuracy checks cover positions around 4K, not a full real 256K model history. Full-model logits/PPL, task completion, speculative acceptance and graph replay must be compared before deployment. Actual 32K/64K populated-context decode is the next decisive timing test. Isolated attention speedups do not equal full-model output TPS gains.

The existing server was confirmed idle before stopping. Only the CLI was compiled; the original server binary was preserved and the baseline service was restarted and verified healthy afterward. No experimental server deployment or source commit was performed. Source edits are confined to `nvidia/src/nanbeige.rs` and `nvidia/src/kernels/nanbeige.cuh`.

## Matched full-model follow-up

The same pinned CLI binary (`f1f44d9ac59bb722b8a5ede370d1dbcbec97e3f317698f8db99d40a9b7aa48b4`) ran both flag settings with unchanged EXL3 2.5 bpw weights, runtime head 4, Q3 KV, host embedding, batch 4,096, pf4 and allocated capacity 262,144. Greedy decode disabled prompt lookup to isolate the attention change. Each run generated 128 tokens and saved all output IDs.

| Input tokens | Final populated context | Default output TPS | INT8 output TPS | Gain |
| --- | ---: | ---: | ---: | ---: |
| 4,096 | 4,224 | 36.84 | 46.82 | 27.1% |
| 32,768 | 32,896 | 24.96 | 35.67 | 42.9% |
| 65,536 | 65,664 | 14.85 | 27.08 | 82.3% |

These are WikiText continuation workloads. The 64K prompt repeats the 32K token sequence twice and is explicitly synthetic. They establish full-model timing at actual populated contexts but do not establish agent task completion or 100 TPS. Greedy sequences diverged at output positions 23, 12 and 13 respectively, so later token content differs between configurations.

Matched teacher-forced scoring of 128 targets after 3,968 prefix tokens measured PPL 12.70232835→12.73208187 (+0.2342%), mean KL(default||INT8) 0.00208196, maximum KL 0.02317962 and top1 agreement 98.4375%. This quality check is short-context; long-context retrieval and task quality remain unmeasured.

Cold prefill throughput was 2,052→1,937 tokens/s at 4K, 905→916 at 32K, and 552→533 at 64K. The experimental flag changes decode attention only, so no prefill improvement is claimed. Sequential single runs with unlocked clocks are not confidence intervals.

The runner script and exact input hashes are in `fullmodel-validation.py` and `fullmodel/manifest.json`; small receipts include `fullmodel/runs.json`, `fullmodel/summary.json` and per-run JSON containing output IDs. Full-logit files and pinned binaries are local-only large artifacts and should not be committed. The unchanged baseline service was restored and its progress endpoint verified responsive in `finally`; no experimental server was deployed.

A quality-preserving adaptive partition alternative is still worth testing before choosing INT8 deployment. Existing isolated data favors 128-token partitions at short contexts, 256 at intermediate contexts and 512 at long contexts. It needs partition-specific graph keys, default-versus-adaptive full-model comparison and numerical/caching transition checks.

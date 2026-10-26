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

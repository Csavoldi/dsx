# T16 speculative decoding measurements

`spec-bench` measures the cost of target verification before a learned drafter is integrated. It runs scalar decode and forced-accept verification on the same token IDs with fixed expert residency, alternates arm order, and checks rollback using different rejected suffixes at every retained-prefix length.

```bash
CUDA_ARCHS=86 cargo build -p dsx --release --features cuda --bin spec-bench
DSX_ADAPT=0 DSX_CPU_THREADS=14 DSX_DEFER=3 \
DSX_T16_WARP_HAD=1 DSX_T16_SPECIALIZED_DQ=1 \
DSX_PREFILL_REUSE_BUFFERS=1 DSX_T16_FUSED_POST_GU=1 \
  target/release/spec-bench --model /path/to/t16.gguf \
  --rank /path/to/rank.json --ids /path/to/prompt.ids \
  --continuation /path/to/continuation.ids
```

Token files contain decimal IDs separated by whitespace or commas. Without `--continuation`, the harness generates a greedy reference once and replays it in every arm. Width 0 denotes scalar `step`; widths 1–8 denote rewindable block verification. Context and placement must match the serving profile being evaluated. The default allocation is 256K context, 8K chunks, 110 GB arena, and 1.5/1 GB main/aux GPU headroom. The harness refuses adaptive placement so each timing arm sees the same expert stores. Oracle results exclude drafting and assume every proposed token is accepted; they are optimistic ceilings.

`--lookup` additionally generates real output using context-copy proposals and the existing deterministic-proposal speculative sampler. It defaults to temperature 1 and top-p 1, saves a fresh seed, and reports proposal count, accepted-prefix tokens, total elapsed time and output IDs. Rejected suffixes must pass rollback checks first. If verification differs from scalar logits, generation requires the explicit diagnostic option `--allow-path-differences`; that option does not make the paths equivalent or establish unchanged quality. A sequence generated with lookup should not be compared token-for-token to a scalar sampled sequence: speculative acceptance consumes random draws differently.

## Compatibility with DCOCO

[DCOCO](https://arxiv.org/html/2610.12570v1) improves acceptance of a parallel neural drafter using a trained candidate-pair corrector. Prompt lookup is a cheap proposer for measuring the verification loop; it does not implement DCOCO. The [released checkpoints](https://github.com/two-delta/DCOCO) target Qwen3-4B and Qwen3-8B. DeepSeek needs its own compatible drafter, target features and trained correction weights.

The current desktop verifier consumes at most eight tokens. Single-token expert deferral changes the scalar target's behavior, while multi-token verification uses immediate expert results. Even with deferral disabled, the verifier's arithmetic path must be checked against scalar decode. Correct speculative sampling preserves its verifier distribution only when the logits and accepted state represent that target consistently.

A drafter must fit alongside GPU1's expert store. Include displaced experts, extra CPU work, synchronization, rejection and commit costs in a final speed comparison. The 8K prefill rate does not measure small-block verification cost.

## Desktop measurement receipt

The [October 25 screen](../bench/results/2026-10-25-speculation/README.md) measured 40.70 TPS scalar deferral-3 decode versus 51.53 TPS with perfect eight-token verification before drafting. Actual context-copy speculation on a second, boundary-crossing fixture regressed from 35.93 TPS scalar to 26.92 TPS at width 8. Rollback suffix comparisons passed; scalar-target equivalence did not. DCOCO is not enabled, and no production throughput gain is claimed.

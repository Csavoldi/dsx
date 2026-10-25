# T16 optimization decisions

This ledger records the October 24, 2026 experiments on 192.168.0.102. T16 weights
remain fixed. Changes stay opt-in until task completion and quality support
adoption; a faster forced continuation alone does not establish agent performance.

| Mechanism | Measured evidence | Decision |
|---|---|---|
| Paired gate/up launches | [Matched oracle screen](paired-screen.json): 44.01665→44.82439 tokens/s, +1.835%; twelve kernel cases exact | Keep opt-in; task results do not isolate this gain |
| GPU categorical sampler | [Matched oracle screen](gpu-sampler-screen.json): 44.54684→45.54070 tokens/s, +2.231%; saved actual distributions and sampled draws agree within recorded limits | Keep opt-in; use failure-conditioned replay for regression claims |
| Stable adaptive-cache ties | [Stable replay](stable-failure-replay.json) removes unordered equal-score eviction choices; repeated and candidate arrays agree on 16,677,120 F32 values | Keep deterministic experimental path; reproduce residency before attributing a difference to kernels |
| Rank-3 deferral | [Control](workflow-control.json) and [no-deferral](workflow-nodefer.json) both complete 0/3 short tasks; deferral runs decode around 40 tokens/s versus 35 without it | No completion-quality advantage established in either direction; preserve exact task defects and checker limitations |
| Auxiliary prefill streaming, main weight 2 | [ABBA screen](aux-prefill-screen.json): warm full prompt 461.852→423.860 tokens/s, -8.226% | Reject on this x4 auxiliary link; [trace](aux-prefill-profile.json) shows extra peer traffic and main-GPU waiting |
| Auxiliary prefill streaming, main weight 4 | [Matched ABBA screen](aux-prefill-w4-screen.json): 462.261→472.444 tokens/s, +2.203% for warm full prompts; incremental changes below 0.3% | Keep experimental; no demonstrated coding-completion gain |
| Next-layer expert prefetch into GPU0 | [Historical negative result](../../../FINDINGS.md#9-negative-results): 37.4→28.3/20.7 tokens/s with deferral, despite reducing CPU work | Reject the old mechanism; a revival needs copy-engine use, prediction confidence and a measured SM/link/DRAM budget |
| More aggressive deferral | [Historical rank-2/rank-1 tests](../../../FINDINGS.md#9-negative-results) add little throughput and worsen measured quality | Keep rank 3 for these pilots; current completion evidence does not justify stronger approximation |
| GPU speculative decoding | [Causal copy replay](causal-lookup-replay.json) does not provide a useful successful-task acceptance/cost case; future-informed oracle throughput is only a ceiling | Do not implement a GPU drafter based on idle utilization alone; measure causal acceptance and total added cost first |
| Blank Braille capture cells converted to spaces | [Tokenizer screen](capture-normalization-screen.json): 7,365→5,855 tokens with identical nonblank dots; [live batch](workflow-space.json) still 0/3 completions | Available workflow option; CPU token saving is not a measured completion gain |
| Persistent functional feedback | [Exact prefix and live checks](feedback-prefix-repair.json): old disappearing note forces 16,749 tokens from 0; retained counterfactual requires 636 new tokens. Live notes remain in stored tool messages | Fix the harness history; [API replay](feedback-cache-api-replay.json) measures 20.489→3.539 seconds on this saved request. No whole-task speedup established |
| Compact source guidance | [Composite judge8 trial](workflow-v8-compact.json): 0/3 completions with concrete rendering defects | No proven benefit; avoid treating compactness as quality |
| Native thinking score 10 | [Recovered pilot](native-score10-recovered.json): 11,615 processed tokens/300 seconds without renderer | Score alone does not bound thinking. Exclude pilot comparisons because proxy ownership failed; retain recovered observations |

The [512-token hard-budget pilot](budget512.json) forces 11 closures at the intended
position and produces code, but still times out without a passing cube. Keep it
disabled by default; shorter thought streams alone do not establish a benefit.

The [2,048-token pilot](budget2048.json) completes tetrahedron in 225.366 seconds
and cube in 179.386 seconds, with reviewed source/display quality and normal
exits. It also removes compact guidance and uses 300-second budgets. Keep it
experimental: two fresh-seed successes do not isolate the cap or establish a
reliable success rate.

Every task uses a fresh request seed and temperature/top-p 1. Historical checker
versions remain reproducible. Judge 8 adds synchronized-update support and stores
repair notes in tool history. Review submitted source and saved terminal output
before calling a task successful: automated certification cannot establish mesh
identity or general occlusion correctness.

Owned server windows restore the original executable, model identity, CPU power
limit and idle health. Proxy ownership is checked before launching an agent;
cleanup errors cannot bypass owned-proxy termination. Summaries exclude requests
started outside the agent lifetime and explicitly list unfinished requests.

The [incremental-prefill profiles](incremental-prefill-profile.json) measure
substantial weight-copy and T16 dequantization costs for 64/256-token additions.
GPU1 kernel busy fractions are about 59%/32% of those captures, so it already
computes resident experts. A new fused prefill design needs measured wall-time
and task-quality gains; the previous direct/int8 substitution was not faster
and changed quality. Unmeasured gains remain zero.

# Doom comparison

The three runs use the same normalized task, Pi 1.4.0, tool set, sampling settings, and fresh game directories. Their tmux sessions are `dsx-doom`, `nanbeige-doom-bf16`, and `deepseek-doom-102`. The manifests record pane IDs, prompt hashes, start times, and configuration. Pane sizes can change when clients attach; the recorded dimensions describe initial preparation.

Local Nanbeige uses EXL3 2.5 bpw, Q3 KV, and the bespoke Rust/CUDA runner on RTX 5090. The reference uses the original 16-bit checkpoint, Q8 KV, and TabbyAPI/ExLlamaV3 on a rented RTX PRO 6000 Blackwell. DeepSeek uses the existing DSX T16 runner on two RTX 3090 GPUs at `.102`. No run uses vLLM.

Nanbeige exposes a binary thinking switch. Pi's `low` setting enables thinking but does not impose a native low reasoning budget. DeepSeek supports native low effort. Differences in model, backend, KV precision, hardware, and reasoning control prevent this comparison from isolating weight quantization as the cause of a task failure.

The reference initially failed Triton attention compilation with Triton 3.3.0. Updating the reference environment to 3.6.0 restored generation and structured tool calls without changing weights. The reference runtime receipt records this change. API smoke tool calls were parsed but not executed.

The local live sample measured approximately 23.52 output tokens/second by retokenizing ten seconds of streamed reasoning. Its request started with 2,566 prompt tokens; the completed request's generated-token count and final populated context were still unavailable. This sample does not replace the earlier controlled short-context benchmark of 69.88 output tokens/second.

At this checkpoint, DeepSeek has a running autoplay preview, cloud Nanbeige has started making tool calls, and local Nanbeige remains in its first reasoning turn. These are interim observations, not final task scores. Preserve the complete agent sessions and rendered previews before scoring completion. The reference pod retains its independent 80-minute automatic expiry.

## Final stopped state

The user stopped optimization and requested that all changes remain on the `nanbeige` branch, without merging into `main`.

Local Nanbeige's Pi pane disappeared and the backend recorded cancellation after 43,931 output tokens. Its whole-run decode average was 29.10 TPS over 1,509.825 seconds, with 2,566 input tokens, three cached tokens, and 46,497 total prompt-plus-generated tokens. Prompt lookup accepted 4,230 of 13,581 drafted tokens in 4,675 attempts. The request produced no game file. The pane removal's cause is unknown. A new idle Pi pane was restored without restarting Doom.

The user also stopped the original 16-bit reference because it continued overthinking. Its rented instance was deleted, and a provider GET returned HTTP 404. The SSH tunnel and automatic-expiry timer are inactive. Compute rental lasted approximately 38.7 minutes, with an estimated compute cost of $1.61 before any provider-specific billing or storage adjustments.

DeepSeek produced a game and a changing autoplay preview. Its generated source and two observer captures are preserved here. The comparison was stopped before a complete common task score was assigned.

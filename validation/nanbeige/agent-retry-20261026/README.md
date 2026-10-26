# Nanbeige agent configuration retry

The user requested a new practical-use attempt, research into Pi settings, an asciinema recording, live streaming TPS, and an independent throughput/cache observer. The active local session is `nanbeige-doom`; Pi is above the preview. All model-generated game files and complete private session/request logs remain under `/home/lxe/gauntlet-nanbeige-doom/20261026-config-retry`.

The official model card recommends temperature 1.0, top-p 0.95, 65,536 maximum new tokens, retained reasoning for agent work, and XML tools. The initial observed Pi requests matched those settings. Pi's low setting only toggles this model's binary thinking switch; it does not enforce a reasoning budget. Its prior configuration therefore cannot be declared the cause of overthinking.

The first retry made incorrect tmux queries, then continued planning without saving code. Its terminal failure was a reproducible tokenizer streaming error on a malformed Unicode byte-fallback run, not a user cancellation. The new decoder matches the complete configured Replace/ByteFallback/Fuse/Strip pipeline and buffers byte runs until their meaning is stable. Regression tests reproduce the original error and compare Unicode, malformed-byte, leading-space, long-run and EOF outputs against full decoding.

A subsequent thinking-off attempt looped over echo/tmux checks after the recorder's read-only client prevented send-keys. The recorder now uses a normal headless attach with TERM set explicitly; control commands were verified. A staged write-only first milestone produced a 7,800-byte Python file at 51.54 backend TPS. Full tools were then restored in a fresh stage for execution, repairs, and combat features. This is assisted practical-use experimentation, not an unchanged gauntlet score. Saving a file does not establish task completion.

The new `/metrics/progress` endpoint counts emitted model tokens, including reasoning and tool arguments. It reports a rolling two-second rate and a whole-response decode rate excluding the first token. The tmux status bar displays those rates, output count, and prompt-plus-output sequence length. It labels queue/prefill separately so stale previous-request rates are not presented as current decode. Sequence length is not a measurement of allocated or populated KV memory.

The observer found high GPU activity, substantial prefix reuse, and a strong short-context attention partition candidate in the existing isolated measurements. Its report separates useful-code throughput from repetitive tool loops. No competing GPU benchmark ran during the gauntlet. The 4.0 bpw candidate is now downloaded and all five shard hashes verified. The matched comparison below measures its decoder override with the same original embedding and runtime 4-bit head; task success remains unmeasured.

Sources: https://huggingface.co/Nanbeige/Nanbeige4.3-33B and https://huggingface.co/Luni/Nanbeige-4.3-33B-EXL3/tree/4.00bpw. Published quantization metrics on different corpora are not directly comparable to our 128-target reference sample.

## Matched quant comparison

Both recipes allocated 262,144 context capacity with Q3 KV, CPU embedding, runtime 4-bit head, batch 4096 and lookup draft length 3. A pinned baseline binary ran sequentially without the Pi agent competing for the GPU. The 4.0 bpw decoder is Luni revision `bc301055ecdfd5d992f2afaf5f6a405a7f235572`.

| Decoder | 128-target perplexity after 4096 input tokens | 4096-input prefill seconds | 512-token short coding decode TPS | Free GPU bytes after quality allocation |
| --- | ---: | ---: | ---: | ---: |
| Local 2.5 bpw | 12.7023 | 2.1161 | 64.08 | 7,020,871,680 |
| Luni 4.0 bpw | 11.1146 | 1.9848 | 52.95 | 1,392,705,536 |

The quality sample is small. Coding output differs between quants, so its throughput also includes different lookup acceptance and output characteristics; it is one prompt, not a general task completion score. Full context allocation is proven here; filling all 262K positions with realistic history is not. The baseline 2.5 bpw server remains the default while we investigate attention scaling.

## Doom stopped and cube retry

The user stopped Doom after repeated implementation failures. One headless animation test had inflated tool feedback by about 50K tokens. The new bounded-output extension preserves raw output locally while summarizing animation and limiting ordinary tool results. This avoids accidental context growth but does not remove attention's scaling cost.

The replacement task is a rotating wireframe cube in the same `nanbeige-doom` tmux session. The first model implementation misread a Braille cell's 2-by-4 dot dimensions as the entire canvas. The repair prompt supplies explicit terminal character and dot dimensions. This defect is recorded rather than counted as a successful cube.

On the same 128 targets, comparison to saved full-precision source logits gives mean source-to-runtime KL 0.19287 for local 2.5 and 0.06081 for Luni 4.0; top-1 agreement is 82.81% and 89.06%. These numbers include runtime head and KV precision differences from the source reference, so they are not decoder-only quantization loss. Both short coding outputs contained an extractable `is_prime` function that passed all integers from -10 through 10,000 against independent trial division; this narrow check does not establish agentic reliability.

The cube repair subsequently produced a full-size rotating wireframe. Independent module checks show a 100-column, 24-row frame with 374 and 323 nonblank cells at different angles; two live tmux frames captured one second apart also differ. The original renderer packed Braille bits incorrectly and the model repaired it after explicit feedback. The parent pressed Enter once when the model omitted it during launch. This is an assisted completion, not an autonomous score. Pi was paused after the final repair for exclusive full-model attention measurements; the CPU preview and asciinema recording continue.

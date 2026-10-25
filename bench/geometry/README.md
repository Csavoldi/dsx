# Geometry coding gauntlet

This gauntlet measures how quickly one local Pi coding agent completes a usable terminal animation against the DSX server. Cube, tetrahedron, and octahedron tasks use the same [prompt](PROMPT.md): filled and distinctly shaded braille faces, correct overlap, a comfortable size with margins, and one successful self-check before finishing.

Run from a machine with Python 3, Pi, tmux, and SSH access to host 192.168.0.102:

```
python3 bench/geometry/run.py cube /absolute/new/run-directory --arm stock --reasoning off
python3 bench/geometry/run.py tetrahedron /absolute/new/run-directory --arm stock --reasoning low
```

During a run, `progress.txt` reports the current request, whether text or code/tool arguments are streaming, and the age of the last model delta. The display pane shows only the submitted animation, so it may remain blank while code is generated or repaired. To watch recorded activity without sending model requests:

```
python3 bench/geometry/status.py /absolute/run-directory --watch
```

The output directory must not exist. Each run owns an isolated agent configuration, workspace, display session, and recording proxy on port 8987. Run tasks sequentially. The runner uses the current server without restarting it; configuring a server candidate is a separate operation. The default timeout is six minutes. `--guidance PATH` appends a recorded instruction for a labeled behavior comparison.

Temperature and top-p are both 1.0, matching the [official model card's recommended sampling parameters](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/blob/main/README.md). Every run creates a fresh random seed base, and every request uses a distinct derived seed recorded in the actual upstream request and proxy metrics. Reasoning is `off` or `low`, explicitly mapped to DSX's supported API effort. Context capacity is 131,072 and the per-request output limit is 8,192 tokens.

The primary outcome is reviewed task completion and elapsed time. `summary.json` records agent time, first observed automated pass, output tokens, requests, tool calls/errors, revisions, and request timings. Engine counters provide decode and new-prompt prefill throughput; streaming deltas are not assumed to be tokens. `host-state.json` records the serving binary, flags, model identity, power limits, and GPU state. Raw prompts, SSE, Pi sessions, revised sources, PTY output, and final artifacts remain in the run directory.

The version 4 judge checks braille animation at three terminal sizes, a usable 15–65 fps range, a typical bounding box between 20% and 95% of each terminal dimension, and Ctrl-C cleanup. It records margins for review. If the agent independently supplies mesh.json, the judge also checks regular convex topology, centered coordinates, and equal edge lengths; the prompt does not require a mesh export. Exact frame rate is recorded but the task permits a smooth approximation to 30 fps. The judge runs submitted code without editing it. Automated checks alone do not establish that the renderer uses the mesh or produces a recognizable shaded solid. Save that source/display review in `review.json`; `functional_pass` remains unset until reviewed and requires a successful agent exit. `automated_pass` is reported separately.

First observed pass is sampled after file changes and may lag the first valid artifact. It does not terminate or coach the agent. A timeout or interrupted run must not count as a successful completion. Compare results only within the same task contract and judge version. The initial longer brief and stricter version 2 review remain historical receipts; they are not directly comparable to the later task contracts. Version 4 makes filled faces, distinct shading, occlusion, and margins explicit without specifying the algorithm or requiring a mesh file.

Record T16 weights, deferral, sampling, CPU policy, cache state, serving revision, and runtime flags. Separate agent-output savings from kernel throughput. Use repeated tasks and shape variants before adopting a candidate; one fast run is insufficient evidence of reliability.

`preview.py display.txt display.png` converts a saved braille screen to a PNG dot mask for review using only the standard library. It does not include ANSI colors.

`--server-log PATH` selects the actual log of a temporary serving candidate. Optional runtime diagnostics are associated with their engine request IDs. Summaries separately record unfinished requests and whether every completed client request has matching engine counters; timeout rates may cover only completed requests. A run with `excluded.json` or `interrupted.json` is ineligible for comparisons.

`--acceptance-stop` is a separate workflow experiment. It uses judge version 5 to certify full-frame clearing and differentiated face shading, waits for the agent's successful display capture of the unchanged artifact, then disables tools and asks for a final summary. Failed visual checks can return a concise description of the functional defect after the agent has inspected that revision. The feedback does not supply a rendering algorithm. Unsupported terminal controls or incremental update styles are not automatically certified. Human review of shape and overall quality remains required; this workflow is not a kernel-speed comparison.

Local checks for the completion guard (Node24 supports the TypeScript test import):

```
python3 bench/geometry/test-terminal.py
node bench/geometry/test-stop.mjs
```

The terminal checks exercise clearing, RGB colors, delayed wrapping, meaningful shading, and rejection of residual trails. The stop-guard checks require a matching certificate and successful self-check, reject stale artifact hashes, and verify that subsequent tools are disabled.

`judge_v6.py` is a standalone checker that measures framing from VT screen coordinates, including renderers that position every row explicitly. The normal runner still uses version 5 for `--acceptance-stop`; current comparisons keep that checker unchanged. Version 6 passed the reviewed GPU-sampling cube and continued to reject the failed tetrahedron and CPU-sampling cube in retrospective checks. It does not repair submissions or weaken shading, size, clearing or animation checks.

Standalone `judge_v7.py` also excludes intervals that contain only terminal
controls between consecutive cursor-home commands. A renderer may return home
at both the end and start of each redraw; those empty intervals neither redraw
the screen nor count toward FPS. Space-only redraws still count, and the same
shading, ghosting, size, resize and cleanup checks apply. Versions 5 and 6 and
original receipts remain available. Apply one judge version to all artifacts
in a comparison and report retrospective adjudication separately from the
judge used by the agent's acceptance extension.

For new comparisons, `--acceptance-stop --judge-version 7` selects the corrected
frame parser for both ongoing acceptance and final certification. The default
remains version 5 for reproducibility of existing comparisons. The completion
guard requires the selected version and an unchanged artifact/self-check hash.
Version 7 can also report a failed size/margin check after the agent inspects
that revision. This functional feedback uses corrected VT coordinates; version
5 comparisons retain their original feedback scope.

`--normalize-capture-blanks` is an opt-in workflow test. It replaces U+2800
(empty braille) with an ASCII space only in a successful `bash` result that
captures this run's named tmux pane. Rows, columns, nonblank dots, and colors are
preserved; code/read results are untouched. `capture-normalization.jsonl` saves
both versions, hashes, and the replaced-cell count. This can reduce the tokens
needed to describe an empty background. It changes model-facing tool text and
must be labeled separately from kernel speed measurements. Check the guard with
`node bench/geometry/test-capture.mjs` before using this option.

`analyze-lookup.py RAW_GENERATIONS.jsonl --run RUN_DIRECTORY --output OUTPUT.json`
replays causal copy proposals from recorded prompt/output IDs without inference.
It evaluates longer suffixes and two verification widths. The original width-4,
suffix-8 counters must match the live lookup audit exactly. Optional
`--oracle-costs bench/results/2026-10-24-102/receipts.json` adds a labeled
conditional decode-only cost model; it does not measure speculative acceptance
or a task speedup. `python3 bench/geometry/test-lookup.py` checks complete,
rejected and unfinished proposals and excludes unavailable future continuations.

An optional style-only prompt experiment is available with
`--guidance bench/geometry/COMPACT_GUIDANCE.md`. It asks for a focused
implementation without long docstrings or unrelated scaffolding. It supplies no
rendering algorithm or geometry. Label these tasks separately: the added prompt
can change generated solutions, and reduced output tokens alone do not prove
better completion. The complete task and guidance hash are recorded.

Standalone `judge_v8.py` adds support for synchronized-update mode (`CSI ?2026 h/l`).
That mode brackets when a terminal displays a redraw and does not change cell
coordinates or shading. Versions 5–7 retain their original control support;
version 8 opts into the additional mode and is selectable explicitly with
`--acceptance-stop --judge-version 8`. Retrospective artifact certification must
remain separate from the original agent timeout and completion result. Human
review still checks geometry and centering that automated extents cannot prove.

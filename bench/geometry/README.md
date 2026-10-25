# Geometry coding gauntlet

This gauntlet measures how quickly one local Pi coding agent completes a usable terminal animation against the DSX server. Cube, tetrahedron, and octahedron tasks use the same short brief: implement a recognizable rotating solid, run one smoke check, repair functional failures, and finish without aesthetic refinement.

Run from a machine with Python 3, Pi, tmux, and SSH access to host 192.168.0.102:

```
python3 bench/geometry/run.py cube /absolute/new/run-directory --arm stock --reasoning off
python3 bench/geometry/run.py tetrahedron /absolute/new/run-directory --arm stock --reasoning low
```

The output directory must not exist. Each run owns an isolated agent configuration, workspace, display session, and recording proxy on port 8987. Run tasks sequentially. The runner uses the current server without restarting it; configuring a server candidate is a separate operation. The default timeout is six minutes. `--guidance PATH` appends a recorded instruction for a labeled behavior comparison.

Temperature and top-p are both 1.0, matching the [official model card's recommended sampling parameters](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/blob/main/README.md). Every run creates a fresh random seed base, and every request uses a distinct derived seed recorded in the actual upstream request and proxy metrics. Reasoning is `off` or `low`, explicitly mapped to DSX's supported API effort. Context capacity is 131,072 and the per-request output limit is 8,192 tokens.

The primary outcome is reviewed task completion and elapsed time. `summary.json` records agent time, first observed automated pass, output tokens, requests, tool calls/errors, revisions, and request timings. Engine counters provide decode and new-prompt prefill throughput; streaming deltas are not assumed to be tokens. `host-state.json` records the serving binary, flags, model identity, power limits, and GPU state. Raw prompts, SSE, Pi sessions, revised sources, PTY output, and final artifacts remain in the run directory.

The version 3 judge checks regular convex mesh topology, centered coordinates, edge lengths, braille animation at three terminal sizes, a usable 15–65 fps range, and Ctrl-C cleanup. Exact frame rate is recorded but the task permits a smooth approximation to 30 fps. The judge runs submitted code without editing it. Automated checks alone do not establish that the renderer uses the mesh or produces a recognizable shaded solid. Save that source/display review in `review.json`; `functional_pass` remains unset until reviewed and requires a successful agent exit. `automated_pass` is reported separately.

First observed pass is sampled after file changes and may lag the first valid artifact. It does not terminate or coach the agent. A timeout or interrupted run must not count as a successful completion. Compare results only within the same task contract and judge version. The initial longer brief and stricter version 2 review remain historical receipts; they are not directly comparable to the shorter version 3 task.

Record T16 weights, deferral, sampling, CPU policy, cache state, serving revision, and runtime flags. Separate agent-output savings from kernel throughput. Use repeated tasks and shape variants before adopting a candidate; one fast run is insufficient evidence of reliability.

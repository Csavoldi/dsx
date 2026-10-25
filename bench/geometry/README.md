# Geometry coding gauntlet

This gauntlet runs one isolated Pi coding agent against an existing DSX server. The agent implements a rotating braille solid from a fresh workspace. Cube, tetrahedron, and octahedron tasks share a rendering contract and an external judge.

Run from a machine with Python 3, Pi, tmux, and SSH access to host 192.168.0.102:

```
python3 bench/geometry/run.py cube /absolute/new/run-directory --arm stock
python3 bench/geometry/run.py tetrahedron /absolute/new/run-directory --arm stock
```

The output directory must not exist. Each run owns a new display session and a local recording proxy on port 8987. Runs must be sequential. The runner uses the current server without restarting it; configuring a server candidate is a separate operation. The default timeout is 1,000 seconds. `--guidance PATH` appends a recorded task instruction for an explicitly labeled agent-behavior comparison.

The primary outcome is functional pass rate and elapsed time to completion. `summary.json` records elapsed agent time, first observed passing revision, output tokens, requests, tool calls/errors, revisions, and request timings. Engine log counters provide decode and new-prompt prefill throughput; streaming deltas are not assumed to be tokens. Raw prompts, SSE, Pi sessions, revised sources, PTY output, and final artifacts remain in the output directory. No API credentials are needed for the configured local DSX endpoint.

The judge checks regular convex mesh topology, edge lengths, braille animation at three terminal sizes, approximate frame rate, and Ctrl-C cleanup. It runs the submitted renderer without editing it. These automated checks do not establish that the renderer uses the submitted mesh or that face shading and rotation are visually correct. Inspect the saved display and source before claiming complete task achievement. First observed pass is sampled after file changes and may lag the actual first valid artifact; it does not terminate or coach the agent.

T16 weights, deferral rank, sampling, CPU policy, cache state, server revision, and runtime flags must be recorded for comparisons. Separate output-length and agent-behavior changes from kernel throughput. Compare repeated tasks and shape variants before adopting a candidate; one fast run is insufficient evidence of a reliable improvement.

# Nanbeige 4 bpw rotating cube recording

`cube-4bpw.cast` preserves the full terminal output from the fresh cube experiment, including setup, code generation, repairs, and the final animation. The original recording runs for 496.05 seconds in a 100-column by 57-row terminal. It contains terminal output only, without audio or captured keyboard input.

`cube-4bpw-4x.mp4` is a CPU-rendered copy for convenient viewing. Playback runs at four times the recorded speed, idle gaps are capped at five seconds before acceleration, and rendering is capped at 15 frames per second. The first 19.5 seconds of accelerated video (about 78 seconds of setup before the fresh implementation started) are trimmed. `cube-4bpw-full-4x.mp4` retains that opening. The full original cast remains available for real-time replay with `asciinema play cube-4bpw.cast`.

The recorder attached normally to the existing `nanbeige-doom` tmux session, with Pi in pane `%4` and the preview in pane `%3`. The separate long-running v3 recording was left active. This recording service was `nanbeige-cube-recording-20261026.service` and was stopped after the final cube was visible.

Run `./render-video.sh` in this directory to reproduce the video. Rendering uses `agg` with the Noto Sans Mono font, followed by FFmpeg's CPU `libx264` encoder. `video-ffprobe.json` records the encoded format and dimensions; `video-frame.png` is a decoded frame of the final cube preview. `SHA256SUMS` records the artifact hashes.

These files are local artifacts. No recording was uploaded to an external service.

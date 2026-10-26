# Expanded Pi cube gauntlet recording

`cube-expanded.cast` preserves the complete 735.35-second terminal recording of the fresh cube experiment. It includes the setup, generation, debugging, and final animation in the existing `nanbeige-doom` tmux session. Pi's tool calls were expanded, and its thinking display was visible. Model thinking remained off.

`cube-expanded-4x.mp4` renders the complete recording at four times its recorded speed. Idle gaps are capped at five seconds before acceleration, and rendering is capped at 15 frames per second. No part of the actual task was cut. The model's repairs and imperfect final cube framing remain visible.

The recording uses a 100-column by 57-row terminal. Pi occupies pane `%4`, and the animation preview occupies pane `%3`. The recorder attached normally, without tmux's read-only client flag, and captured output without audio or keyboard input.

Replay the original with `asciinema play cube-expanded.cast`. Run `./render-video.sh` to reproduce the MP4 using `agg` and FFmpeg's CPU `libx264` encoder. `video-frame.png` is decoded near the final valid video packet, and `video-ffprobe.json` records the encoded format. `SHA256SUMS` records artifact hashes.

The recording service was `nanbeige-cube-expanded-recording-20261026.service`. It was stopped after ten seconds of the final preview. All files remain local; nothing was uploaded externally.

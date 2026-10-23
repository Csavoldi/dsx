#!/usr/bin/env bash
# Create the amd-gauntlet tmux session: 8 panes, 2 rows x 4 columns (300x71, as Volta's pi-gauntlet), each in ~/gauntlet-amd.
set -euo pipefail
tmux kill-session -t amd-gauntlet 2>/dev/null || true
top=$(tmux new-session -d -P -F '#{pane_id}' -s amd-gauntlet -x 300 -y 71 -c ~/gauntlet-amd)
bot=$(tmux split-window -v -P -F '#{pane_id}' -t "$top" -c ~/gauntlet-amd)
for p in "$top" "$bot"; do
  b=$(tmux split-window -h -P -F '#{pane_id}' -t "$p" -c ~/gauntlet-amd -l 50%)
  tmux split-window -h -t "$p" -c ~/gauntlet-amd -l 50%
  tmux split-window -h -t "$b" -c ~/gauntlet-amd -l 50%
done
tmux list-panes -t amd-gauntlet -F '#{pane_top} #{pane_left} #{pane_width}x#{pane_height}'

#!/usr/bin/env bash
# Start the recording of the amd-gauntlet session, then send the four staged prompts, and log /stats until done.
set -euo pipefail
cd ~/gauntlet-amd-braille
mapfile -t ids < <(tmux list-panes -t amd-gauntlet -F '#{pane_top} #{pane_left} #{pane_id}' | sort -n -k1,1 -k2,2 | awk '{print $3}')
# record at the window's own size (+1 row for the status line), so the frame has no dotted margins
W=$(tmux display -p -t amd-gauntlet '#{window_width}'); H=$(( $(tmux display -p -t amd-gauntlet '#{window_height}') + 1 ))
tmux new-session -d -s amd-gauntlet-rec -x "$W" -y "$H" \
  "env -u TMUX asciinema rec -q --overwrite --window-size ${W}x${H} -c 'tmux attach -t amd-gauntlet -f ignore-size' $HOME/gauntlet-amd-braille/gauntlet.cast"
sleep 2
for i in 0 1 2 3; do tmux send-keys -t "${ids[$i]}" Enter; done
nohup ./watch.sh > watch.log 2>&1 &
echo "recording to gauntlet.cast; prompts sent to ${ids[*]:0:4}"

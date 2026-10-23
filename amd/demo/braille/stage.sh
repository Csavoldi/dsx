#!/usr/bin/env bash
# Stage the braille gauntlet (Volta's ~/gauntlet-braille, against dsx-amd) in the amd-gauntlet tmux session: pi in the
# top four panes with each task typed into its prompt but not sent; the bottom four panes are blank shells in the
# matching folders (the displays). Each TASK.md names its display pane, filled in from TASK.tmpl here.
set -euo pipefail
cd ~/gauntlet-amd-braille
mapfile -t ids < <(tmux list-panes -t amd-gauntlet -F '#{pane_top} #{pane_left} #{pane_id}' | sort -n -k1,1 -k2,2 | awk '{print $3}')
[ "${#ids[@]}" -eq 8 ] || { echo "expected 8 panes, found ${#ids[@]}"; exit 1; }
dirs=(01-cube 02-pyramid 03-torus 04-sphere)
for i in 0 1 2 3; do
  find "${dirs[$i]}" -mindepth 1 ! -name TASK.tmpl -delete
  sed "s#{DISPLAY}#${ids[$((i + 4))]}#g" "${dirs[$i]}/TASK.tmpl" > "${dirs[$i]}/TASK.md"
  tmux respawn-pane -k -t "${ids[$i]}" -c ~/gauntlet-amd-braille/${dirs[$i]}
  tmux respawn-pane -k -t "${ids[$((i + 4))]}" -c ~/gauntlet-amd-braille/${dirs[$i]}
  tmux clear-history -t "${ids[$i]}"; tmux clear-history -t "${ids[$((i + 4))]}"
done
sleep 2
for i in 0 1 2 3; do
  tmux send-keys -t "${ids[$i]}" "clear && env -u NODE_OPTIONS pi --offline --provider dsx-amd --model deepseek-v4.1-flash:off --tools read,bash,edit,write --no-extensions --no-skills --no-prompt-templates" Enter
  tmux send-keys -t "${ids[$((i + 4))]}" "clear" Enter
done
sleep 6
for i in 0 1 2 3; do
  tmux send-keys -t "${ids[$i]}" C-o
  tmux send-keys -t "${ids[$i]}" -l "$(tr -d '\n' < ${dirs[$i]}/TASK.md)"
done
for i in 0 1 2 3; do echo "${dirs[$i]}: agent ${ids[$i]}, display ${ids[$((i + 4))]}"; done

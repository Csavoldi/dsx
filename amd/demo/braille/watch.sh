#!/usr/bin/env bash
# Log the server's /stats every 5 s to stats.jsonl (without the history series) until every agent pane (top row) has been
# idle (no "Working" spinner) for 60 s, then stop the recording.
cd ~/gauntlet-amd-braille
idle=0
while :; do
  curl -s -m 4 http://127.0.0.1:9091/stats | python3 -c 'import json,sys,time; s=json.load(sys.stdin); s.pop("history",None); s["wall"]=time.time(); print(json.dumps(s))' >> stats.jsonl 2>/dev/null || true
  busy=0
  for id in $(tmux list-panes -t amd-gauntlet -F '#{pane_top} #{pane_id}' | awk '$1 == 0 {print $2}'); do
    tmux capture-pane -p -t "$id" | grep -q "Working" && busy=1
  done
  if [ $busy -eq 0 ]; then idle=$((idle + 5)); else idle=0; fi
  [ $idle -ge 60 ] && break
  sleep 5
done
tmux kill-session -t amd-gauntlet-rec 2>/dev/null || true
echo "all idle at $(date)"

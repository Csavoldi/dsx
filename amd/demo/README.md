# dsx-amd serving demo

Copies of the gauntlets run on rental 7 (RUNNER.md "Serving demo"). Live copies: `~/gauntlet-amd` (gauntlet/) and
`~/gauntlet-amd-braille` (braille/); merge `pi-provider.json` into `~/.pi/agent/models.json`.

1. On the box: `DSXR_CHUNK=256 ... dsx-amd serve --slots 8 --batch-ctx 262144 --ctx 262200 --port 8090 --admin-port 8091`
   (RUNNER.md Demo for the environment).
2. Here: `ssh -N -L 9090:127.0.0.1:8090 -L 9091:127.0.0.1:8091 dsxamd`; dashboard at http://localhost:9091.
3. gauntlet: `setup.sh` (2x4 tmux `amd-gauntlet`), `start.sh` (records `gauntlet.cast`, launches 8 agents), `watch.sh`.
   braille: `../gauntlet/setup.sh` for the panes, then `stage.sh` (agents top, displays below), `go.sh` (records, sends).

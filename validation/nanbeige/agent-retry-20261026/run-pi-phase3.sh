#!/bin/sh
set -eu
cd /home/lxe/gauntlet-nanbeige-doom/20261026-config-retry/game
export PI_CODING_AGENT_DIR=/home/lxe/gauntlet-nanbeige-doom/20261026-config-retry/agent
unset NODE_OPTIONS
exec pi --offline --provider dsx-nanbeige --model nanbeige4.3-33b --thinking off --tools read,bash,edit,write --system-prompt "$(cat ../SYSTEM.txt)" --no-extensions --extension ../bounded-output.ts --no-mcp --no-skills --no-prompt-templates --no-context-files --session-dir ../sessions "$(cat ../TASK-phase3.md)"

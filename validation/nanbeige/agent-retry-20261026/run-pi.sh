#!/bin/sh
set -eu
cd /home/lxe/gauntlet-nanbeige-doom/20261026-config-retry/game
export PI_CODING_AGENT_DIR=/home/lxe/gauntlet-nanbeige-doom/20261026-config-retry/agent
unset NODE_OPTIONS
exec pi --offline --provider dsx-nanbeige --model nanbeige4.3-33b --thinking low --tools read,bash,edit,write --append-system-prompt 'Work in small executable steps. Use absolute file paths for read, write, and edit. Save code to disk before adding optional features. The game directory is /home/lxe/gauntlet-nanbeige-doom/20261026-config-retry/game.' --no-extensions --no-mcp --no-skills --no-prompt-templates --no-context-files --session-dir ../sessions "$(cat ../TASK.md)"

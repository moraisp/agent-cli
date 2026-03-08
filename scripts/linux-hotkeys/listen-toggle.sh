#!/usr/bin/env bash

# Listen-toggle for agent-cli background-chat
#
# Triggered by compositor keybinding (e.g. Niri Mod+X).
# Each press toggles: starts recording if idle, stops recording if listening.
# No key-release detection needed — works with any compositor/hotkey tool.
#
# Prerequisites:
#   - background-chat running: agent-cli background-chat --tts &
#
# Niri config example:
#   Mod+X hotkey-overlay-title="Background Chat Toggle" {
#       spawn "/home/USER/.local/bin/agent-cli-listen-toggle";
#   }

VENV_BIN="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)/../../.venv/bin"
exec "$VENV_BIN/agent-cli" background-chat --listen-toggle --quiet

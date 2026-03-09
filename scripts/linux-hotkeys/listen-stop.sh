#!/usr/bin/env bash

# Listen-stop for agent-cli background-chat
#
# Triggered by compositor keybinding (e.g. Niri Mod+Shift+X).
# Interrupts the current turn (recording, processing, or TTS) and returns
# to idle WITHOUT triggering a new listen session.
#
# Prerequisites:
#   - background-chat running: agent-cli background-chat --tts &
#
# Niri config example:
#   Mod+Shift+X hotkey-overlay-title="Background Chat Stop" {
#       spawn "/home/USER/.local/bin/agent-cli-listen-stop";
#   }

VENV_BIN="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)/../../.venv/bin"
exec "$VENV_BIN/agent-cli" background-chat --listen-stop --quiet

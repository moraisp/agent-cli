#!/usr/bin/env bash

# Push-to-talk for agent-cli background-chat
#
# Triggered by compositor keybinding on key-down (e.g. Niri Mod+X).
# Immediately starts recording, then watches for key release via evdev
# to stop recording and trigger the LLM response.
#
# Prerequisites:
#   - python-evdev:  sudo pacman -S python-evdev
#   - input group:   sudo usermod -aG input $USER  (then re-login)
#   - background-chat running: agent-cli background-chat --tts &
#
# Usage:
#   ./push-to-talk.sh                    # default: KEY_X
#   ./push-to-talk.sh --key KEY_SPACE    # custom key

SCRIPT_DIR="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"
export PATH="$PATH:$HOME/.local/bin"

exec /usr/bin/python3 "$SCRIPT_DIR/push-to-talk.py" "$@"

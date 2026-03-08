# Linux Hotkeys

System-wide hotkeys for agent-cli voice AI features on Linux.

## Setup

```bash
./setup-linux-hotkeys.sh
```

The setup script will:
1. Install notification support if missing
2. Show you the exact hotkey bindings to add to your desktop environment
3. Provide copy-paste ready configuration for popular desktop environments

## Usage

- **`Super+Shift+R`** → Toggle voice transcription (start/stop with result)
- **`Super+Shift+A`** → Autocorrect clipboard text
- **`Super+Shift+V`** → Toggle voice edit mode for clipboard

Results appear in notifications and clipboard.

## Push-to-Talk (Background Chat)

True push-to-talk for `background-chat`: hold a key to record, release to get a response.

Works on compositors that only fire bindings on key-down (e.g. **Niri**) by using
**evdev** to detect the key release independently.

### Prerequisites

```bash
# Install python-evdev
sudo pacman -S python-evdev          # Arch/CachyOS
# sudo apt install python3-evdev     # Debian/Ubuntu
# sudo dnf install python3-evdev     # Fedora

# Add yourself to the input group (re-login after this)
sudo usermod -aG input $USER
```

### Usage

1. Start the background chat process:

    ```bash
    agent-cli background-chat --tts &
    ```

2. Test push-to-talk manually (hold X, then release):

    ```bash
    python3 push-to-talk.py --key KEY_X
    ```

3. Bind to your compositor (e.g. Niri):

    ```kdl
    binds {
        Mod+X { spawn "bash" "/path/to/linux-hotkeys/push-to-talk.sh"; }
    }
    ```

### How it works

```
Key-down (Mod+X) → Niri spawns push-to-talk.sh
  → SIGUSR1 → background-chat starts recording
  → evdev monitors for X release
Key-up (X released)
  → SIGUSR2 → background-chat stops recording → LLM → TTS
  → script exits
```

- **No key grab** — passive evdev monitoring, doesn't interfere with your compositor
- **Dedup lock** — if key repeat triggers the binding multiple times, extra instances exit silently
- **Safety timeout** — auto-stops recording after 30s if release isn't detected (configurable with `--timeout`)

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--key` | `KEY_X` | Evdev key name to monitor for release |
| `--timeout` | `30` | Seconds before auto-stop if release not detected |
| `--pid-file` | (auto) | Custom PID file path |

### Custom key

Use any evdev key name: `KEY_SPACE`, `KEY_RIGHTALT`, `BTN_EXTRA` (mouse side button), etc.

```bash
# List all available key names:
python3 -c "import evdev; print([k for k in dir(evdev.ecodes) if k.startswith(('KEY_', 'BTN_'))])"
```

### Troubleshooting

**"No input devices accessible"**
- Run `groups` and check for `input`. If missing: `sudo usermod -aG input $USER` then **re-login**.
- Verify with: `ls -la /dev/input/event*`

**"No background-chat process running"**
- Start it first: `agent-cli background-chat --tts &`

**Key release not detected**
- Run `python3 -c "import evdev; [print(evdev.InputDevice(p).name) for p in evdev.list_devices()]"` to list devices.
- Try a different `--key` value matching your actual key.

## Desktop Environment Support

The setup script provides copy-paste ready instructions for:

- **Hyprland**: Add bindings to `~/.config/hypr/hyprland.conf`
- **Sway**: Add bindings to `~/.config/sway/config`
- **i3**: Add bindings to `~/.config/i3/config`
- **GNOME**: Use Settings → Keyboard → Custom Shortcuts
- **KDE**: Use System Settings → Shortcuts → Custom Shortcuts
- **XFCE**: Use Settings Manager → Keyboard → Application Shortcuts
- **Other**: Manual hotkey configuration in your desktop environment

## Features

- **Manual configuration**: Simple setup with clear instructions for each desktop environment
- **Wayland support**: Includes clipboard syncing for Wayland compositors
- **Fallback notifications**: Uses `notify-send`, `dunstify`, or console output
- **Error handling**: Shows notifications for both success and failure cases
- **PATH handling**: Scripts automatically find agent-cli installation

## Troubleshooting

**Hotkeys not working?**
- Check your desktop's keyboard shortcut settings for conflicts
- Make sure you added the bindings to your desktop environment's config
- Verify the script paths are correct

**No notifications?**
```bash
sudo apt install libnotify-bin  # Ubuntu/Debian
sudo dnf install libnotify      # Fedora/RHEL
sudo pacman -S libnotify        # Arch
```

**Services not running?**
```bash
./start-all-services.sh
```

That's it! System-wide hotkeys for agent-cli on Linux.

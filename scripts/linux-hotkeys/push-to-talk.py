#!/usr/bin/python3
"""Push-to-talk for agent-cli background-chat using evdev.

Designed to be triggered by a compositor keybinding (e.g. Niri's Mod+X).
On launch, sends SIGUSR1 to background-chat (start recording), then monitors
evdev for the trigger key release and sends SIGUSR2 (stop recording).

Prerequisites:
    sudo pacman -S python-evdev   # or pip install evdev
    sudo usermod -aG input $USER  # then re-login

Usage:
    python3 push-to-talk.py                    # default: KEY_X
    python3 push-to-talk.py --key KEY_SPACE
    python3 push-to-talk.py --key KEY_X --timeout 15

Niri config example:
    binds {
        Mod+X { spawn "bash" "/path/to/push-to-talk.sh"; }
    }
"""

from __future__ import annotations

import argparse
import fcntl
import os
import select
import signal
import sys
import tempfile
import time
from pathlib import Path

PID_DIR = Path.home() / ".cache" / "agent-cli"
PROCESS_NAME = "background-chat"
LOCK_FILE = Path(tempfile.gettempdir()) / "agent-cli-ptt.lock"
# Minimum seconds between SIGUSR1 (start recording) and SIGUSR2 (stop recording).
# Prevents stale evdev key-release events (already buffered before Python finished
# starting) from triggering SIGUSR2 immediately, and keeps the lock held long enough
# to block Niri key-repeat instances from spawning a second SIGUSR1.
MIN_HOLD_SECS = 0.4


def _read_pid() -> int | None:
    """Read the background-chat PID from its PID file."""
    pid_file = PID_DIR / f"{PROCESS_NAME}.pid"
    if not pid_file.exists():
        return None
    try:
        pid = int(pid_file.read_text().strip())
        # Verify the process is alive
        os.kill(pid, 0)
        return pid
    except (ValueError, ProcessLookupError, PermissionError):
        return None


def _send_signal(pid: int, sig: signal.Signals) -> bool:
    """Send a signal to a process, returning True on success."""
    try:
        os.kill(pid, sig)
        return True
    except (ProcessLookupError, PermissionError) as exc:
        print(f"Failed to send {sig.name} to PID {pid}: {exc}", file=sys.stderr)
        return False


def _acquire_lock() -> int | None:
    """Acquire an exclusive lock to prevent duplicate PTT instances.

    Returns the lock fd on success, None if another instance is running.
    """
    try:
        fd = os.open(str(LOCK_FILE), os.O_CREAT | os.O_RDWR, 0o600)
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return fd
    except OSError:
        return None


def _release_lock(fd: int) -> None:
    """Release the exclusive lock."""
    try:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)
        LOCK_FILE.unlink(missing_ok=True)
    except OSError:
        pass


def _find_keyboard_devices() -> list:
    """Find all evdev devices that have key event capabilities."""
    import evdev  # type: ignore[import-untyped]

    devices = []
    for path in sorted(evdev.list_devices()):
        try:
            dev = evdev.InputDevice(path)
            caps = dev.capabilities()
            # EV_KEY = 1; check device has key capabilities
            if evdev.ecodes.EV_KEY in caps:
                devices.append(dev)
        except (PermissionError, OSError):
            continue
    return devices


def _wait_for_key_release(key_code: int, timeout: float) -> bool:
    """Monitor evdev devices for a specific key release event.

    Returns True if the release was detected, False on timeout.
    """
    import evdev  # type: ignore[import-untyped]

    devices = _find_keyboard_devices()
    if not devices:
        print(
            "No input devices accessible. Ensure you are in the 'input' group:\n"
            "  sudo usermod -aG input $USER  # then re-login",
            file=sys.stderr,
        )
        # Hold the lock for the full timeout so key-repeat can't spam SIGUSR1
        # while recording is active (background-chat will rely on VAD to stop).
        time.sleep(timeout)
        return False

    try:
        fds = {dev.fd: dev for dev in devices}
        while True:
            readable, _, _ = select.select(list(fds.keys()), [], [], timeout)
            if not readable:
                # Timeout — no release detected
                return False
            for fd in readable:
                dev = fds[fd]
                for event in dev.read():
                    # value == 0 means key release
                    if (
                        event.type == evdev.ecodes.EV_KEY
                        and event.code == key_code
                        and event.value == 0
                    ):
                        return True
    finally:
        for dev in devices:
            try:
                dev.close()
            except Exception:  # noqa: BLE001
                pass


def _resolve_key_code(key_name: str) -> int:
    """Resolve an evdev key name (e.g. 'KEY_X') to its numeric code."""
    import evdev  # type: ignore[import-untyped]

    code = getattr(evdev.ecodes, key_name, None)
    if code is None:
        # Try common prefixes
        for prefix in ("KEY_", "BTN_"):
            code = getattr(evdev.ecodes, prefix + key_name, None)
            if code is not None:
                break
    if code is None:
        print(
            f"Unknown key: {key_name}\n"
            "Use evdev key names like KEY_X, KEY_SPACE, KEY_RIGHTALT, BTN_EXTRA.\n"
            "Run 'python3 -c \"import evdev; print([k for k in dir(evdev.ecodes) if k.startswith((\\\"KEY_\\\", \\\"BTN_\\\"))])\"' to list all.",
            file=sys.stderr,
        )
        sys.exit(1)
    return code


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Push-to-talk for agent-cli background-chat.",
    )
    parser.add_argument(
        "--key",
        default="KEY_X",
        help="Evdev key name to monitor for release (default: KEY_X).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="Max seconds to wait for key release before auto-stopping (default: 30).",
    )
    parser.add_argument(
        "--pid-file",
        type=Path,
        default=None,
        help=f"Custom PID file path (default: {PID_DIR / PROCESS_NAME}.pid).",
    )
    args = parser.parse_args()

    # Prevent duplicate instances (e.g. key repeat triggering Niri binding again)
    lock_fd = _acquire_lock()
    if lock_fd is None:
        # Another PTT instance is already running — silently exit
        sys.exit(0)

    try:
        # Find the background-chat process (cheap check before importing evdev)
        pid = _read_pid()
        if pid is None:
            msg = "No background-chat process running.\nStart one first: agent-cli background-chat --tts &"
            print(msg, file=sys.stderr)
            sys.exit(1)

        # Resolve the key code (imports evdev)
        try:
            key_code = _resolve_key_code(args.key)
        except ModuleNotFoundError:
            print("python-evdev not installed. Run: sudo pacman -S python-evdev", file=sys.stderr)
            sys.exit(1)

        # Send SIGUSR1 → start listening immediately
        press_time = time.monotonic()
        if not _send_signal(pid, signal.SIGUSR1):
            sys.exit(1)

        # Wait for key release → send SIGUSR2
        _wait_for_key_release(key_code, args.timeout)

        # Enforce minimum recording window regardless of when release was detected.
        # This guards against stale evdev events (release buffered during Python
        # startup) causing SIGUSR2 to fire before any audio is captured, and keeps
        # the lock held so Niri key-repeat can't spawn a second SIGUSR1 instance.
        elapsed = time.monotonic() - press_time
        if elapsed < MIN_HOLD_SECS:
            time.sleep(MIN_HOLD_SECS - elapsed)

        # Stop recording — proceed to LLM + TTS
        _send_signal(pid, signal.SIGUSR2)
    except Exception as exc:  # noqa: BLE001
        print(f"Unexpected error: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
    finally:
        _release_lock(lock_fd)


if __name__ == "__main__":
    main()

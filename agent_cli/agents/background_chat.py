"""Background voice chat agent triggered on-demand via a global command.

Runs as a persistent background process. Listening is **not** automatic; it is
triggered on-demand:

- ``--listen``: trigger a listen session in the running process
- ``--toggle``: start the process if stopped, stop it if running
- ``--stop``: stop the running process

**Trigger mechanism:**

- Unix (Linux/macOS): sends ``SIGUSR1`` to the process
- Windows: creates a listen-trigger file that the process polls

**Behaviour when triggered:**

1. **Idle** -- starts recording immediately.
2. **Speaking (TTS active)** -- interrupts TTS and starts recording immediately.
3. **Recording/processing** -- interrupts current turn and starts a fresh recording.

After each conversation turn the process returns to idle, waiting for the next
trigger. Conversation history is persisted across triggers (same format as the
``chat`` command).
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import shutil
import signal
import subprocess
import sys
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import TYPE_CHECKING

import typer

from agent_cli import config, opts
from agent_cli.agents.chat import (
    _handle_conversation_turn,
    _load_conversation_history,
)
from agent_cli.cli import app
from agent_cli.core import process
from agent_cli.core.audio import setup_devices
from agent_cli.core.deps import requires_extras
from agent_cli.core.utils import (
    InteractiveStopEvent,
    console,
    maybe_live,
    print_command_line_args,
    print_with_style,
    setup_logging,
    stop_or_status_or_toggle,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Generator

LOGGER = logging.getLogger(__name__)

PROCESS_NAME = "background-chat"
NOTIFICATION_REPLACE_ID = "38291"
ACTIVE_NOTIFICATION_TIMEOUT_MS = 86_400_000

SYSTEM_PROMPT = """\
You are a voice assistant. Your output is read aloud by text-to-speech.

ABSOLUTE RULES (violating any of these is a failure):
1. MAXIMUM TWO SENTENCES per response. No exceptions. Ever.
2. ZERO formatting. No markdown, no bullets, no lists, no headings, no bold, no italics, no code blocks. Plain text only.
3. ZERO emojis or emoticons. Absolutely forbidden. Not a single one.
4. ZERO special characters used for formatting: no asterisks, hashes, backticks, brackets, dashes used as bullets.
5. Go directly to the answer. Do not restate the question. Do not add filler, preamble, or pleasantries.
6. Never ask follow-up questions like "Want to know more?" or "Anything else?".
7. Write numbers as spoken words when short (e.g. "three" not "3"), use digits for long numbers.

If you can answer in one word, answer in one word. If you need a sentence, use one. Two sentences is the hard ceiling.

Examples of correct responses:
Q: How many r's in strawberry? A: Three.
Q: What's the capital of France? A: Paris.
Q: What's 15 percent of 200? A: Thirty.
Q: Hello! A: Hi!
Q: Explain what a black hole is. A: A region in space where gravity is so strong nothing can escape, not even light.
Q: How do I make pasta? A: Boil salted water, cook the pasta until al dente, then drain and sauce it.

You have access to the following tools:
- read_file: Read the content of a file.
- execute_code: Execute a shell command.
- add_memory: Add important information to long-term memory for future recall.
- search_memory: Search your long-term memory for relevant information.
- update_memory: Modify existing memories by ID when information changes.
- list_all_memories: Show all stored memories with their IDs and details.
- list_memory_categories: See what types of information you've remembered.
- duckduckgo_search: Search the web for current information.

Memory guidelines:
- When the user shares personal information or preferences, offer to remember them.
- Before answering, consider searching memory for relevant context.
- Ask permission before storing sensitive information.
"""

AGENT_INSTRUCTIONS = """\
A summary of the previous conversation is provided in the <previous-conversation> tag.
The user's current message is in the <user-message> tag.

If the message continues the previous conversation, use that context.
If it is a new topic, ignore the previous conversation.
Respond in one or two sentences maximum. No formatting, no emojis, plain text only.
If an image is attached, use it to answer the user's question. Describe only what is asked about.
"""

# Regex matching voice commands that request vision / screen reading.
_VISION_PATTERN = re.compile(
    r"\b(?:see|look|look at|read this|read that|what(?:'s| is) (?:this|that|on (?:my |the )?screen)"
    r"|show me|what do you see|what am i looking at|describe (?:this|that|the screen|what you see))\b",
    re.IGNORECASE,
)


def _needs_vision(text: str) -> bool:
    """Return True if the transcribed text implies the user wants screen analysis."""
    return _VISION_PATTERN.search(text) is not None


def _capture_screen() -> bytes | None:
    """Capture the screen and return PNG bytes, or None on failure."""
    grim = shutil.which("grim")
    if grim is not None:
        # Wayland -- grim writes PNG to stdout with "-"
        result = subprocess.run(
            [grim, "-"],
            capture_output=True,
            check=False,
        )
        if result.returncode == 0 and result.stdout:
            return result.stdout

    scrot = shutil.which("scrot")
    if scrot is not None:
        # X11 -- scrot writes PNG to stdout with "-"
        result = subprocess.run(
            [scrot, "-"],
            capture_output=True,
            check=False,
        )
        if result.returncode == 0 and result.stdout:
            return result.stdout

    LOGGER.warning("No screenshot tool found (tried grim, scrot)")
    return None


def _maybe_capture_screen(text: str) -> bytes | None:
    """Capture the screen if the transcribed text implies a vision request."""
    if not _needs_vision(text):
        return None
    LOGGER.info("Vision trigger detected in: %r -- capturing screen", text)
    screenshot = _capture_screen()
    if screenshot is None:
        LOGGER.warning("Screen capture failed or no tool available")
    else:
        LOGGER.info("Screen captured: %d bytes", len(screenshot))
    return screenshot


class _BackgroundChatNotifier:
    """Manage persistent Linux desktop notifications for background-chat."""

    def __init__(self) -> None:
        self._command = shutil.which("dunstify") or shutil.which("notify-send")
        self._gdbus = shutil.which("gdbus")
        self._current_state: str | None = None
        self._notification_id: int | None = None

    def _notify(self, *, timeout_ms: int, body: str) -> None:
        if self._command is None:
            return

        command = [
            self._command,
            "-p",
            "-t",
            str(timeout_ms),
            "Background chat",
            body,
        ]
        if self._notification_id is not None:
            command[1:1] = ["-r", str(self._notification_id)]
        elif self._command.endswith("notify-send"):
            command[1:1] = ["-r", NOTIFICATION_REPLACE_ID]

        with suppress(Exception):
            subprocess.run(
                command,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
                text=True,
                capture_output=False,
            )

    def _notify_persistent(self, body: str) -> None:
        if self._gdbus is not None:
            if self._notification_id is not None:
                self.dismiss()
            command = [
                self._gdbus,
                "call",
                "--session",
                "--dest",
                "org.freedesktop.Notifications",
                "--object-path",
                "/org/freedesktop/Notifications",
                "--method",
                "org.freedesktop.Notifications.Notify",
                "Background chat",
                "0",
                "",
                "Background chat",
                body,
                "[]",
                "{}",
                str(ACTIVE_NOTIFICATION_TIMEOUT_MS),
            ]
            with suppress(Exception):
                result = subprocess.run(
                    command,
                    capture_output=True,
                    check=False,
                    text=True,
                )
                match = re.search(r"uint32\s+(\d+)", result.stdout)
                if match:
                    self._notification_id = int(match.group(1))
            return

        if self._command is None:
            return

        command = [
            self._command,
            "-p",
            "-h",
            "boolean:resident:true",
            "-t",
            str(ACTIVE_NOTIFICATION_TIMEOUT_MS),
            "Background chat",
            body,
        ]
        if self._notification_id is not None:
            command[1:1] = ["-r", str(self._notification_id)]
        elif self._command.endswith("notify-send"):
            command[1:1] = ["-r", NOTIFICATION_REPLACE_ID]

        with suppress(Exception):
            result = subprocess.run(
                command,
                capture_output=True,
                check=False,
                text=True,
            )
            notification_id = result.stdout.strip()
            if notification_id.isdigit():
                self._notification_id = int(notification_id)

    def dismiss(self) -> None:
        if self._notification_id is None:
            return
        if self._gdbus is not None:
            with suppress(Exception):
                subprocess.run(
                    [
                        self._gdbus,
                        "call",
                        "--session",
                        "--dest",
                        "org.freedesktop.Notifications",
                        "--object-path",
                        "/org/freedesktop/Notifications",
                        "--method",
                        "org.freedesktop.Notifications.CloseNotification",
                        str(self._notification_id),
                    ],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                )
        elif self._command is not None and self._command.endswith("dunstify"):
            # dunstify supports -C <id> to close a notification by ID
            with suppress(Exception):
                subprocess.run(
                    [self._command, "-C", str(self._notification_id)],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                )
        self._notification_id = None

    def update(self, state: str) -> None:
        if self._current_state == state:
            return

        self._current_state = state
        if state == "listening":
            self._notify_persistent("Listening")
        elif state == "thinking":
            self._notify_persistent("Thinking")
        elif state == "talking":
            self._notify_persistent("Talking")
        elif state == "idle":
            self.dismiss()


@contextmanager
def _signal_handling_context(  # noqa: PLR0915
    loop: asyncio.AbstractEventLoop,
    listen_event: asyncio.Event,
    global_stop: asyncio.Event,
    turn_stop: InteractiveStopEvent,
    quiet: bool,
    current_turn_task: list,
    set_state: Callable[[str], None],
) -> Generator[Callable[[], None], None, None]:
    """Register SIGUSR1/SIGTERM/SIGINT handlers for background-chat."""
    sigint_count = [0]

    def _cancel_current_turn() -> None:
        task = current_turn_task[0]
        if task is not None and not task.done():
            LOGGER.info("Cancelling current turn task")
            task.cancel()

    def _sigusr1() -> None:
        LOGGER.info("SIGUSR1 received -- triggering listen session")
        # Cancel the running turn at whatever stage it's in (recording, waiting
        # for ASR transcript, LLM, or TTS) so we can start fresh immediately.
        _cancel_current_turn()
        turn_stop.set()
        listen_event.set()

    def _sigusr2() -> None:
        LOGGER.info("SIGUSR2 received -- stop recording (push-to-talk key-up)")
        turn_stop.set()
        # Update notification immediately: recording has stopped, we're now
        # waiting for the ASR server to return the transcript.
        set_state("thinking")

    def _sigterm() -> None:
        LOGGER.info("SIGTERM received -- shutting down")
        _cancel_current_turn()
        global_stop.set()
        turn_stop.set()
        listen_event.set()

    def _sigint() -> None:
        sigint_count[0] += 1
        if sigint_count[0] == 1:
            # First Ctrl+C: interrupt the current turn only, return to idle.
            # Do NOT set global_stop or listen_event — the process keeps running
            # and waits for the next --listen trigger.
            LOGGER.info("First SIGINT -- interrupting current turn, returning to idle (Ctrl+C again or --stop to quit)")
            if not quiet:
                console.print("\n[yellow]Interrupted. Use --stop or Ctrl+C again to quit.[/yellow]")
            _cancel_current_turn()
            turn_stop.set()
        else:
            LOGGER.info("Second SIGINT -- shutting down")
            if not quiet:
                console.print("\n[yellow]Shutting down...[/yellow]")
            _cancel_current_turn()
            global_stop.set()
            turn_stop.set()
            listen_event.set()

    if sys.platform != "win32":
        loop.add_signal_handler(signal.SIGUSR1, _sigusr1)
        loop.add_signal_handler(signal.SIGUSR2, _sigusr2)
    loop.add_signal_handler(signal.SIGTERM, _sigterm)
    loop.add_signal_handler(signal.SIGINT, _sigint)

    def reset_sigint_count() -> None:
        sigint_count[0] = 0

    try:
        yield reset_sigint_count
    finally:
        if sys.platform != "win32":
            with suppress(Exception):
                loop.remove_signal_handler(signal.SIGUSR1)
            with suppress(Exception):
                loop.remove_signal_handler(signal.SIGUSR2)
        with suppress(Exception):
            loop.remove_signal_handler(signal.SIGTERM)
            loop.remove_signal_handler(signal.SIGINT)


async def _wait_for_listen(
    listen_event: asyncio.Event,
    global_stop: asyncio.Event,
) -> None:
    """Wait until a listen trigger or global stop is requested."""
    while not listen_event.is_set() and not global_stop.is_set():
        # On Windows, poll the listen trigger file
        if sys.platform == "win32" and process.check_listen_file(PROCESS_NAME):
            listen_event.set()
            break
        await asyncio.sleep(0.1)


async def _async_main(  # noqa: PLR0912, PLR0915
    *,
    provider_cfg: config.ProviderSelection,
    general_cfg: config.General,
    history_cfg: config.History,
    audio_in_cfg: config.AudioInput,
    wyoming_asr_cfg: config.WyomingASR,
    openai_asr_cfg: config.OpenAIASR,
    gemini_asr_cfg: config.GeminiASR,
    ollama_cfg: config.Ollama,
    openai_llm_cfg: config.OpenAILLM,
    gemini_llm_cfg: config.GeminiLLM,
    audio_out_cfg: config.AudioOutput,
    wyoming_tts_cfg: config.WyomingTTS,
    openai_tts_cfg: config.OpenAITTS,
    kokoro_tts_cfg: config.KokoroTTS,
    gemini_tts_cfg: config.GeminiTTS,
) -> None:
    """Main async loop for background-chat."""
    device_info = setup_devices(general_cfg, audio_in_cfg, audio_out_cfg)
    if device_info is None:
        return
    input_device_index, _, tts_output_device_index = device_info
    audio_in_cfg.input_device_index = input_device_index
    if audio_out_cfg.enable_tts:
        audio_out_cfg.output_device_index = tts_output_device_index

    # Load conversation history (shared with `chat` by default)
    conversation_history = []
    if history_cfg.history_dir:
        history_path = Path(history_cfg.history_dir).expanduser()
        history_path.mkdir(parents=True, exist_ok=True)
        os.environ["AGENT_CLI_HISTORY_DIR"] = str(history_path)
        history_file = history_path / "conversation.json"
        conversation_history = _load_conversation_history(history_file, history_cfg.last_n_messages)

    loop = asyncio.get_running_loop()
    listen_event = asyncio.Event()
    global_stop = asyncio.Event()
    turn_stop = InteractiveStopEvent()
    notifier = _BackgroundChatNotifier()

    def _set_state(state: str) -> None:
        notifier.update(state)
        process.write_state(PROCESS_NAME, state)

    pid = process.read_pid_file(PROCESS_NAME)
    if not general_cfg.quiet:
        print_with_style(
            f"Background chat ready (PID: {pid}). "
            "Waiting for listen trigger -- run with --listen or send SIGUSR1.",
            style="blue",
        )

    current_turn_task: list[asyncio.Task | None] = [None]

    with (
        _signal_handling_context(
            loop, listen_event, global_stop, turn_stop, general_cfg.quiet,
            current_turn_task, _set_state,
        ) as reset_sigint_count,
        maybe_live(not general_cfg.quiet) as live,
    ):
        try:
            while not global_stop.is_set():
                LOGGER.info("Entering idle -- waiting for listen trigger")
                _set_state("idle")
                await _wait_for_listen(listen_event, global_stop)
                if global_stop.is_set():
                    LOGGER.info("Global stop set -- exiting loop")
                    break
                LOGGER.info("Listen triggered -- starting conversation turn")
                listen_event.clear()
                if sys.platform == "win32":
                    process.clear_listen_file(PROCESS_NAME)
                turn_stop.clear()

                try:
                    # Wrap in a Task so SIGUSR1 can cancel it at any await point
                    # (including while waiting for ASR transcript, LLM, or TTS).
                    turn_task = asyncio.create_task(
                        _handle_conversation_turn(
                            stop_event=turn_stop,
                            conversation_history=conversation_history,
                            provider_cfg=provider_cfg,
                            general_cfg=general_cfg,
                            history_cfg=history_cfg,
                            audio_in_cfg=audio_in_cfg,
                            wyoming_asr_cfg=wyoming_asr_cfg,
                            openai_asr_cfg=openai_asr_cfg,
                            gemini_asr_cfg=gemini_asr_cfg,
                            ollama_cfg=ollama_cfg,
                            openai_llm_cfg=openai_llm_cfg,
                            gemini_llm_cfg=gemini_llm_cfg,
                            audio_out_cfg=audio_out_cfg,
                            wyoming_tts_cfg=wyoming_tts_cfg,
                            openai_tts_cfg=openai_tts_cfg,
                            kokoro_tts_cfg=kokoro_tts_cfg,
                            gemini_tts_cfg=gemini_tts_cfg,
                            live=live,
                            system_prompt=SYSTEM_PROMPT,
                            agent_instructions=AGENT_INSTRUCTIONS,
                            on_state_change=_set_state,
                            capture_screen_fn=_maybe_capture_screen,
                        )
                    )
                    current_turn_task[0] = turn_task
                    await turn_task
                    LOGGER.info("Conversation turn completed normally")
                except asyncio.CancelledError:
                    LOGGER.info("CancelledError during turn -- resuming idle state")
                except BaseException as exc:
                    if isinstance(exc, SystemExit):
                        raise
                    LOGGER.exception("Unexpected error during conversation turn -- resuming idle state")
                    if not general_cfg.quiet:
                        print_with_style(f"Turn failed ({type(exc).__name__}), resuming idle.", style="yellow")
                finally:
                    current_turn_task[0] = None
                    turn_stop.clear()
                    reset_sigint_count()

                if not general_cfg.quiet and not global_stop.is_set():
                    with suppress(Exception):
                        print_with_style("Idle -- waiting for listen trigger.", style="dim blue")
        except BaseException as exc:
            if not isinstance(exc, (SystemExit, KeyboardInterrupt)):
                LOGGER.exception("Unhandled exception escaped the main loop: %s", type(exc).__name__)
            raise
        finally:
            _set_state("idle")


@app.command("background-chat", rich_help_panel="Voice Commands")
@requires_extras("audio", "llm")
def background_chat(
    *,
    # --- Provider Selection ---
    asr_provider: str = opts.ASR_PROVIDER,
    llm_provider: str = opts.LLM_PROVIDER,
    tts_provider: str = opts.TTS_PROVIDER,
    # --- ASR (Audio) Configuration ---
    input_device_index: int | None = opts.INPUT_DEVICE_INDEX,
    input_device_name: str | None = opts.INPUT_DEVICE_NAME,
    asr_wyoming_ip: str = opts.ASR_WYOMING_IP,
    asr_wyoming_port: int = opts.ASR_WYOMING_PORT,
    asr_openai_model: str = opts.ASR_OPENAI_MODEL,
    asr_openai_base_url: str | None = opts.ASR_OPENAI_BASE_URL,
    asr_openai_prompt: str | None = opts.ASR_OPENAI_PROMPT,
    asr_gemini_model: str = opts.ASR_GEMINI_MODEL,
    # --- LLM Configuration ---
    llm_ollama_model: str = opts.LLM_OLLAMA_MODEL,
    llm_ollama_host: str = opts.LLM_OLLAMA_HOST,
    llm_openai_model: str = opts.LLM_OPENAI_MODEL,
    openai_api_key: str | None = opts.OPENAI_API_KEY,
    openai_base_url: str | None = opts.OPENAI_BASE_URL,
    llm_gemini_model: str = opts.LLM_GEMINI_MODEL,
    gemini_api_key: str | None = opts.GEMINI_API_KEY,
    # --- TTS Configuration ---
    enable_tts: bool = opts.ENABLE_TTS,
    output_device_index: int | None = opts.OUTPUT_DEVICE_INDEX,
    output_device_name: str | None = opts.OUTPUT_DEVICE_NAME,
    tts_speed: float = opts.TTS_SPEED,
    tts_wyoming_ip: str = opts.TTS_WYOMING_IP,
    tts_wyoming_port: int = opts.TTS_WYOMING_PORT,
    tts_wyoming_voice: str | None = opts.TTS_WYOMING_VOICE,
    tts_wyoming_language: str | None = opts.TTS_WYOMING_LANGUAGE,
    tts_wyoming_speaker: str | None = opts.TTS_WYOMING_SPEAKER,
    tts_openai_model: str = opts.TTS_OPENAI_MODEL,
    tts_openai_voice: str = opts.TTS_OPENAI_VOICE,
    tts_openai_base_url: str | None = opts.TTS_OPENAI_BASE_URL,
    tts_kokoro_model: str = opts.TTS_KOKORO_MODEL,
    tts_kokoro_voice: str = opts.TTS_KOKORO_VOICE,
    tts_kokoro_host: str = opts.TTS_KOKORO_HOST,
    tts_gemini_model: str = opts.TTS_GEMINI_MODEL,
    tts_gemini_voice: str = opts.TTS_GEMINI_VOICE,
    # --- Process Management ---
    stop: bool = opts.STOP,
    status: bool = opts.STATUS,
    toggle: bool = opts.TOGGLE,
    listen: bool = opts.LISTEN,
    listen_stop: bool = opts.LISTEN_STOP,
    listen_toggle: bool = opts.LISTEN_TOGGLE,
    # --- History Options ---
    history_dir: Path = typer.Option(  # noqa: B008
        "~/.config/agent-cli/history",
        "--history-dir",
        help="Directory for conversation history and long-term memory.",
        rich_help_panel="History Options",
    ),
    last_n_messages: int = typer.Option(
        50,
        "--last-n-messages",
        help="Number of past messages to include as context for the LLM. "
        "Set to 0 to start fresh each session.",
        rich_help_panel="History Options",
    ),
    # --- General Options ---
    save_file: Path | None = opts.SAVE_FILE,
    log_level: opts.LogLevel = opts.LOG_LEVEL,
    log_file: str | None = opts.LOG_FILE,
    list_devices: bool = opts.LIST_DEVICES,
    quiet: bool = opts.QUIET,
    config_file: str | None = opts.CONFIG_FILE,
    print_args: bool = opts.PRINT_ARGS,
) -> None:
    """Background voice chat agent triggered on-demand via a global command.

    Unlike ``chat``, this command starts in **idle** mode and only listens when
    triggered by ``--listen`` (or a hotkey bound to send ``SIGUSR1`` on Unix).

    **Trigger behaviour:**

    - **Idle** -- starts recording immediately.
    - **Speaking (TTS)** -- interrupts playback and starts recording immediately.
    - **Recording/processing** -- interrupts current turn and starts fresh.

    **Push-to-talk workflow (key-down / key-up):**

    Bind ``--listen`` to key-down and ``--listen-stop`` to key-up:

        # Key-down: start recording (interrupts TTS if active)
        agent-cli background-chat --listen

        # Key-up: stop recording, trigger LLM response + TTS
        agent-cli background-chat --listen-stop

    **Simple workflow (toggle-style):**

    Start once in the background:

        agent-cli background-chat --tts &

    Trigger a listen session (records until silence is detected):

        agent-cli background-chat --listen

    Stop the background process:

        agent-cli background-chat --stop
    """
    if print_args:
        print_command_line_args(locals())

    setup_logging(log_level, log_file, quiet=quiet)
    general_cfg = config.General(
        log_level=log_level,
        log_file=log_file,
        quiet=quiet,
        list_devices=list_devices,
        clipboard=False,
        save_file=save_file,
    )

    if stop_or_status_or_toggle(
        PROCESS_NAME,
        "background chat",
        stop,
        status,
        toggle,
        quiet=general_cfg.quiet,
    ):
        return

    if listen:
        if process.trigger_listen(PROCESS_NAME):
            if not general_cfg.quiet:
                print_with_style("Listen triggered.", style="blue")
        elif not general_cfg.quiet:
            print_with_style(
                "No background-chat process is running. "
                "Start one first with: agent-cli background-chat",
                style="yellow",
            )
        return

    if listen_stop:
        if process.trigger_listen_stop(PROCESS_NAME):
            if not general_cfg.quiet:
                print_with_style("Listen stop triggered.", style="blue")
        elif not general_cfg.quiet:
            print_with_style(
                "No background-chat process is running. "
                "Start one first with: agent-cli background-chat",
                style="yellow",
            )
        return

    if listen_toggle:
        if process.trigger_listen_toggle(PROCESS_NAME):
            if not general_cfg.quiet:
                print_with_style("Listen toggle triggered.", style="blue")
        elif not general_cfg.quiet:
            print_with_style(
                "No background-chat process is running. "
                "Start one first with: agent-cli background-chat",
                style="yellow",
            )
        return

    with process.pid_file_context(PROCESS_NAME), suppress(KeyboardInterrupt):
        cfgs = config.create_provider_configs_from_locals(locals())
        history_cfg = config.History(
            history_dir=history_dir,
            last_n_messages=last_n_messages,
        )

        asyncio.run(
            _async_main(
                provider_cfg=cfgs.provider,
                general_cfg=general_cfg,
                history_cfg=history_cfg,
                audio_in_cfg=cfgs.audio_in,
                wyoming_asr_cfg=cfgs.wyoming_asr,
                openai_asr_cfg=cfgs.openai_asr,
                gemini_asr_cfg=cfgs.gemini_asr,
                ollama_cfg=cfgs.ollama,
                openai_llm_cfg=cfgs.openai_llm,
                gemini_llm_cfg=cfgs.gemini_llm,
                audio_out_cfg=cfgs.audio_out,
                wyoming_tts_cfg=cfgs.wyoming_tts,
                openai_tts_cfg=cfgs.openai_tts,
                kokoro_tts_cfg=cfgs.kokoro_tts,
                gemini_tts_cfg=cfgs.gemini_tts,
            ),
        )

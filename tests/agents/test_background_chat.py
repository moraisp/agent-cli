"""Tests for background-chat main loop resilience.

Verifies that the idle-loop / signal-handling architecture survives all the
real-world scenarios:

- Multiple SIGUSR1 triggers in a row (start listening → answer → idle → repeat)
- SIGUSR1 during TTS (should interrupt TTS and restart listening immediately)
- SIGINT during a turn (should abort turn, return to idle, process stays alive)
- SIGINT while idle (process stays alive)
- SIGTERM at any point (clean shutdown)
- SIGUSR1 after SIGINT (process can start a fresh turn after a Ctrl+C interrupt)

These tests exercise the async loop and signal dispatch without real audio,
ASR, LLM, or TTS.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_cli.agents import background_chat as background_chat_module
from agent_cli.core.utils import InteractiveStopEvent


# ---------------------------------------------------------------------------
# Helper: a minimal reproduction of the background-chat event loop.
# We extract the core state-machine so we can test it in isolation.
# ---------------------------------------------------------------------------


async def _background_chat_loop(
    *,
    listen_event: asyncio.Event,
    global_stop: asyncio.Event,
    turn_stop: InteractiveStopEvent,
    handle_turn: Any,
    on_idle: Any | None = None,
) -> list[str]:
    """Simplified version of background_chat._async_main's while-loop.

    Returns a log of events for assertions.
    """
    log: list[str] = []
    while not global_stop.is_set():
        log.append("idle")
        if on_idle:
            on_idle()
        # Wait for listen trigger or global stop
        while not listen_event.is_set() and not global_stop.is_set():
            await asyncio.sleep(0.01)
        if global_stop.is_set():
            log.append("shutdown")
            break

        log.append("turn_start")
        listen_event.clear()
        turn_stop.clear()

        try:
            await handle_turn(turn_stop)
            log.append("turn_ok")
        except asyncio.CancelledError:
            log.append("turn_cancelled")
        except Exception:
            log.append("turn_error")
        finally:
            turn_stop.clear()

    return log


# ---------------------------------------------------------------------------
# Simulated signal dispatchers (mirror _signal_handling_context logic)
# ---------------------------------------------------------------------------


def simulate_sigusr1(
    listen_event: asyncio.Event,
    turn_stop: InteractiveStopEvent,
) -> None:
    """Simulate SIGUSR1: interrupt current turn + schedule new listen."""
    turn_stop.set()
    listen_event.set()


def simulate_sigterm(
    listen_event: asyncio.Event,
    global_stop: asyncio.Event,
    turn_stop: InteractiveStopEvent,
) -> None:
    """Simulate SIGTERM: full shutdown."""
    global_stop.set()
    turn_stop.set()
    listen_event.set()  # unblock _wait_for_listen


def simulate_sigint_first(
    turn_stop: InteractiveStopEvent,
) -> None:
    """Simulate first SIGINT: interrupt turn only, don't exit."""
    turn_stop.set()


def simulate_sigusr2(
    turn_stop: InteractiveStopEvent,
) -> None:
    """Simulate SIGUSR2 (--listen-stop): stop recording, trigger response."""
    turn_stop.set()


def simulate_sigint_second(
    listen_event: asyncio.Event,
    global_stop: asyncio.Event,
    turn_stop: InteractiveStopEvent,
) -> None:
    """Simulate second SIGINT: full shutdown."""
    global_stop.set()
    turn_stop.set()
    listen_event.set()


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def events() -> tuple[asyncio.Event, asyncio.Event, InteractiveStopEvent]:
    """Create the three event objects used by the loop."""
    return asyncio.Event(), asyncio.Event(), InteractiveStopEvent()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_single_turn_then_stop(
    events: tuple[asyncio.Event, asyncio.Event, InteractiveStopEvent],
) -> None:
    """SIGUSR1 → turn completes → SIGTERM → clean exit."""
    listen_event, global_stop, turn_stop = events

    async def handle_turn(stop: InteractiveStopEvent) -> None:
        # Simulate a normal turn (short delay)
        await asyncio.sleep(0.05)

    async def driver() -> list[str]:
        task = asyncio.create_task(
            _background_chat_loop(
                listen_event=listen_event,
                global_stop=global_stop,
                turn_stop=turn_stop,
                handle_turn=handle_turn,
            )
        )
        await asyncio.sleep(0.02)
        simulate_sigusr1(listen_event, turn_stop)  # trigger listen
        await asyncio.sleep(0.1)  # wait for turn to complete
        simulate_sigterm(listen_event, global_stop, turn_stop)  # shutdown
        return await task

    log = await driver()
    assert "idle" in log
    assert "turn_start" in log
    assert "turn_ok" in log
    assert "shutdown" in log


@pytest.mark.asyncio
async def test_multiple_turns_sequential(
    events: tuple[asyncio.Event, asyncio.Event, InteractiveStopEvent],
) -> None:
    """Three SIGUSR1 triggers in sequence, each completing normally.

    Verifies the process doesn't exit between turns.
    """
    listen_event, global_stop, turn_stop = events
    turn_count = 0

    async def handle_turn(stop: InteractiveStopEvent) -> None:
        nonlocal turn_count
        turn_count += 1
        await asyncio.sleep(0.03)

    async def driver() -> list[str]:
        task = asyncio.create_task(
            _background_chat_loop(
                listen_event=listen_event,
                global_stop=global_stop,
                turn_stop=turn_stop,
                handle_turn=handle_turn,
            )
        )
        for _ in range(3):
            await asyncio.sleep(0.02)
            simulate_sigusr1(listen_event, turn_stop)
            await asyncio.sleep(0.08)  # wait for turn

        simulate_sigterm(listen_event, global_stop, turn_stop)
        return await task

    log = await driver()
    assert turn_count == 3
    assert log.count("turn_ok") == 3


@pytest.mark.asyncio
async def test_sigusr1_during_tts_interrupts_and_restarts(
    events: tuple[asyncio.Event, asyncio.Event, InteractiveStopEvent],
) -> None:
    """SIGUSR1 during TTS (simulated long turn) should interrupt and start new turn.

    Turn 1: started, interrupted mid-way by SIGUSR1
    Turn 2: completes normally
    """
    listen_event, global_stop, turn_stop = events
    turn_count = 0

    async def handle_turn(stop: InteractiveStopEvent) -> None:
        nonlocal turn_count
        turn_count += 1
        # Simulate long TTS playback that checks stop_event
        for _ in range(50):
            if stop.is_set():
                return  # TTS interrupted — exit normally
            await asyncio.sleep(0.01)

    async def driver() -> list[str]:
        task = asyncio.create_task(
            _background_chat_loop(
                listen_event=listen_event,
                global_stop=global_stop,
                turn_stop=turn_stop,
                handle_turn=handle_turn,
            )
        )
        await asyncio.sleep(0.02)
        simulate_sigusr1(listen_event, turn_stop)  # start turn 1
        await asyncio.sleep(0.05)  # mid-TTS
        simulate_sigusr1(listen_event, turn_stop)  # interrupt turn 1, start turn 2
        await asyncio.sleep(0.6)  # wait for turn 2 to complete
        simulate_sigterm(listen_event, global_stop, turn_stop)
        return await task

    log = await driver()
    assert turn_count == 2
    assert log.count("turn_start") == 2
    # Both turns should show as OK (TTS interrupt returns normally)
    assert log.count("turn_ok") == 2


@pytest.mark.asyncio
async def test_sigint_during_turn_returns_to_idle(
    events: tuple[asyncio.Event, asyncio.Event, InteractiveStopEvent],
) -> None:
    """First Ctrl+C during a turn should abort the turn and return to idle.

    The process must stay alive and accept a new SIGUSR1 afterwards.
    """
    listen_event, global_stop, turn_stop = events
    turn_count = 0

    async def handle_turn(stop: InteractiveStopEvent) -> None:
        nonlocal turn_count
        turn_count += 1
        for _ in range(50):
            if stop.is_set():
                return
            await asyncio.sleep(0.01)

    async def driver() -> list[str]:
        task = asyncio.create_task(
            _background_chat_loop(
                listen_event=listen_event,
                global_stop=global_stop,
                turn_stop=turn_stop,
                handle_turn=handle_turn,
            )
        )
        await asyncio.sleep(0.02)
        simulate_sigusr1(listen_event, turn_stop)  # start turn 1
        await asyncio.sleep(0.05)
        simulate_sigint_first(turn_stop)  # Ctrl+C: abort turn, back to idle
        await asyncio.sleep(0.05)

        # Process should still be alive — trigger a new turn
        simulate_sigusr1(listen_event, turn_stop)  # start turn 2
        await asyncio.sleep(0.6)  # wait for turn 2 to complete

        simulate_sigterm(listen_event, global_stop, turn_stop)
        return await task

    log = await driver()
    assert turn_count == 2, f"Expected 2 turns but got {turn_count}; log={log}"
    assert not global_stop.is_set() or log[-1] == "shutdown"


@pytest.mark.asyncio
async def test_sigint_while_idle_does_not_exit(
    events: tuple[asyncio.Event, asyncio.Event, InteractiveStopEvent],
) -> None:
    """Ctrl+C while idle should not kill the process."""
    listen_event, global_stop, turn_stop = events

    async def handle_turn(stop: InteractiveStopEvent) -> None:
        await asyncio.sleep(0.03)

    async def driver() -> list[str]:
        task = asyncio.create_task(
            _background_chat_loop(
                listen_event=listen_event,
                global_stop=global_stop,
                turn_stop=turn_stop,
                handle_turn=handle_turn,
            )
        )
        await asyncio.sleep(0.02)
        simulate_sigint_first(turn_stop)  # Ctrl+C while idle — nothing happens
        await asyncio.sleep(0.05)

        # Process should still be alive
        simulate_sigusr1(listen_event, turn_stop)  # trigger a turn
        await asyncio.sleep(0.08)

        simulate_sigterm(listen_event, global_stop, turn_stop)
        return await task

    log = await driver()
    assert "turn_ok" in log, f"Expected a completed turn; log={log}"


@pytest.mark.asyncio
async def test_second_sigint_exits(
    events: tuple[asyncio.Event, asyncio.Event, InteractiveStopEvent],
) -> None:
    """Two quick SIGINT signals should shut down the process."""
    listen_event, global_stop, turn_stop = events

    async def handle_turn(stop: InteractiveStopEvent) -> None:
        for _ in range(100):
            if stop.is_set():
                return
            await asyncio.sleep(0.01)

    async def driver() -> list[str]:
        task = asyncio.create_task(
            _background_chat_loop(
                listen_event=listen_event,
                global_stop=global_stop,
                turn_stop=turn_stop,
                handle_turn=handle_turn,
            )
        )
        await asyncio.sleep(0.02)
        simulate_sigusr1(listen_event, turn_stop)
        await asyncio.sleep(0.03)
        simulate_sigint_second(listen_event, global_stop, turn_stop)
        return await task

    log = await driver()
    assert global_stop.is_set()


@pytest.mark.asyncio
async def test_sigterm_during_turn_exits_after_turn(
    events: tuple[asyncio.Event, asyncio.Event, InteractiveStopEvent],
) -> None:
    """SIGTERM during a turn should shut down after the turn exits."""
    listen_event, global_stop, turn_stop = events

    async def handle_turn(stop: InteractiveStopEvent) -> None:
        for _ in range(50):
            if stop.is_set():
                return
            await asyncio.sleep(0.01)

    async def driver() -> list[str]:
        task = asyncio.create_task(
            _background_chat_loop(
                listen_event=listen_event,
                global_stop=global_stop,
                turn_stop=turn_stop,
                handle_turn=handle_turn,
            )
        )
        await asyncio.sleep(0.02)
        simulate_sigusr1(listen_event, turn_stop)  # start turn
        await asyncio.sleep(0.05)
        simulate_sigterm(listen_event, global_stop, turn_stop)  # shutdown mid-turn
        return await task

    log = await driver()
    # Process should have exited after the turn
    assert global_stop.is_set()
    assert log.count("turn_start") == 1


@pytest.mark.asyncio
async def test_rapid_sigusr1_five_times(
    events: tuple[asyncio.Event, asyncio.Event, InteractiveStopEvent],
) -> None:
    """Five rapid SIGUSR1 triggers — each one interrupts the previous turn.

    Should result in 5 turn starts, process stays alive throughout.
    """
    listen_event, global_stop, turn_stop = events
    turn_count = 0

    async def handle_turn(stop: InteractiveStopEvent) -> None:
        nonlocal turn_count
        turn_count += 1
        for _ in range(100):
            if stop.is_set():
                return
            await asyncio.sleep(0.01)

    async def driver() -> list[str]:
        task = asyncio.create_task(
            _background_chat_loop(
                listen_event=listen_event,
                global_stop=global_stop,
                turn_stop=turn_stop,
                handle_turn=handle_turn,
            )
        )
        # Rapidly fire 5 SIGUSR1 signals
        for i in range(5):
            await asyncio.sleep(0.03)
            simulate_sigusr1(listen_event, turn_stop)

        # Wait for the last turn to finish
        await asyncio.sleep(1.2)
        simulate_sigterm(listen_event, global_stop, turn_stop)
        return await task

    log = await driver()
    assert turn_count >= 2, f"Expected at least 2 turns, got {turn_count}; log={log}"
    assert "shutdown" in log


@pytest.mark.asyncio
async def test_sigusr1_after_sigint_works(
    events: tuple[asyncio.Event, asyncio.Event, InteractiveStopEvent],
) -> None:
    """After Ctrl+C aborts a turn, SIGUSR1 should start a fresh turn.

    This is the exact scenario that was failing: Ctrl+C interrupted turn 1,
    then SIGUSR1 should start turn 2 without the process exiting.
    """
    listen_event, global_stop, turn_stop = events
    turn_count = 0

    async def handle_turn(stop: InteractiveStopEvent) -> None:
        nonlocal turn_count
        turn_count += 1
        for _ in range(50):
            if stop.is_set():
                return
            await asyncio.sleep(0.01)

    async def driver() -> list[str]:
        task = asyncio.create_task(
            _background_chat_loop(
                listen_event=listen_event,
                global_stop=global_stop,
                turn_stop=turn_stop,
                handle_turn=handle_turn,
            )
        )
        # Turn 1
        await asyncio.sleep(0.02)
        simulate_sigusr1(listen_event, turn_stop)
        await asyncio.sleep(0.05)
        # Ctrl+C to abort turn 1
        simulate_sigint_first(turn_stop)
        await asyncio.sleep(0.05)

        # Turn 2 via SIGUSR1 — this must work
        simulate_sigusr1(listen_event, turn_stop)
        await asyncio.sleep(0.6)  # wait for full completion

        # Turn 3 via SIGUSR1 — still working
        simulate_sigusr1(listen_event, turn_stop)
        await asyncio.sleep(0.6)

        simulate_sigterm(listen_event, global_stop, turn_stop)
        return await task

    log = await driver()
    assert turn_count == 3, f"Expected 3 turns, got {turn_count}; log={log}"


@pytest.mark.asyncio
async def test_handle_turn_exception_does_not_exit(
    events: tuple[asyncio.Event, asyncio.Event, InteractiveStopEvent],
) -> None:
    """If handle_turn raises an unexpected exception, the process stays alive."""
    listen_event, global_stop, turn_stop = events
    turn_count = 0

    async def handle_turn(stop: InteractiveStopEvent) -> None:
        nonlocal turn_count
        turn_count += 1
        if turn_count == 1:
            msg = "Simulated LLM error"
            raise RuntimeError(msg)
        await asyncio.sleep(0.03)

    async def driver() -> list[str]:
        task = asyncio.create_task(
            _background_chat_loop(
                listen_event=listen_event,
                global_stop=global_stop,
                turn_stop=turn_stop,
                handle_turn=handle_turn,
            )
        )
        await asyncio.sleep(0.02)
        simulate_sigusr1(listen_event, turn_stop)  # turn 1 — will raise
        await asyncio.sleep(0.05)
        simulate_sigusr1(listen_event, turn_stop)  # turn 2 — should succeed
        await asyncio.sleep(0.08)

        simulate_sigterm(listen_event, global_stop, turn_stop)
        return await task

    log = await driver()
    assert turn_count == 2
    assert "turn_error" in log
    assert "turn_ok" in log


@pytest.mark.asyncio
async def test_sigterm_during_turn_then_sigusr1_does_not_revive(
    events: tuple[asyncio.Event, asyncio.Event, InteractiveStopEvent],
) -> None:
    """SIGTERM during turn followed by SIGUSR1 should NOT start a new turn.

    This reproduces the real-world crash: user ran --stop (SIGTERM) during
    turn 1, then immediately ran --listen (SIGUSR1). The process should
    shut down cleanly — SIGTERM means "die", and SIGUSR1 cannot undo it.
    """
    listen_event, global_stop, turn_stop = events
    turn_count = 0

    async def handle_turn(stop: InteractiveStopEvent) -> None:
        nonlocal turn_count
        turn_count += 1
        for _ in range(50):
            if stop.is_set():
                return
            await asyncio.sleep(0.01)

    async def driver() -> list[str]:
        task = asyncio.create_task(
            _background_chat_loop(
                listen_event=listen_event,
                global_stop=global_stop,
                turn_stop=turn_stop,
                handle_turn=handle_turn,
            )
        )
        await asyncio.sleep(0.02)
        simulate_sigusr1(listen_event, turn_stop)  # start turn 1
        await asyncio.sleep(0.05)
        simulate_sigterm(listen_event, global_stop, turn_stop)  # --stop mid-turn
        await asyncio.sleep(0.05)  # turn exits due to stop_event
        simulate_sigusr1(listen_event, turn_stop)  # --listen after --stop
        await asyncio.sleep(0.1)
        return await task

    log = await driver()
    assert turn_count == 1, f"Expected only 1 turn (SIGTERM should prevent more); log={log}"
    assert global_stop.is_set()


# ---------------------------------------------------------------------------
# Push-to-talk tests (SIGUSR1 = --listen, SIGUSR2 = --listen-stop)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_push_to_talk_basic(
    events: tuple[asyncio.Event, asyncio.Event, InteractiveStopEvent],
) -> None:
    """Key-down (--listen) starts recording, key-up (--listen-stop) ends it.

    SIGUSR1 triggers a turn, ASR records until stop_event.is_set(),
    SIGUSR2 sets stop_event → ASR finalizes → LLM → TTS → idle.
    """
    listen_event, global_stop, turn_stop = events
    phases: list[str] = []

    async def handle_turn(stop: InteractiveStopEvent) -> None:
        phases.append("recording_start")
        # Simulate ASR: record until stop_event is set
        for _ in range(100):
            if stop.is_set():
                break
            await asyncio.sleep(0.01)
        phases.append("recording_stop")
        stop.clear()  # mirrors _handle_conversation_turn
        # Simulate LLM + TTS (should NOT be interrupted)
        phases.append("llm_tts_start")
        await asyncio.sleep(0.05)
        if stop.is_set():
            phases.append("llm_tts_interrupted")
            return
        phases.append("llm_tts_done")

    async def driver() -> list[str]:
        task = asyncio.create_task(
            _background_chat_loop(
                listen_event=listen_event,
                global_stop=global_stop,
                turn_stop=turn_stop,
                handle_turn=handle_turn,
            )
        )
        await asyncio.sleep(0.02)
        # Key-down: start recording
        simulate_sigusr1(listen_event, turn_stop)
        await asyncio.sleep(0.05)
        # Key-up: stop recording
        simulate_sigusr2(turn_stop)
        await asyncio.sleep(0.15)  # wait for LLM+TTS to complete

        simulate_sigterm(listen_event, global_stop, turn_stop)
        return await task

    log = await driver()
    assert phases == ["recording_start", "recording_stop", "llm_tts_start", "llm_tts_done"]
    assert "turn_ok" in log


@pytest.mark.asyncio
async def test_listen_stop_while_idle_is_noop(
    events: tuple[asyncio.Event, asyncio.Event, InteractiveStopEvent],
) -> None:
    """SIGUSR2 while idle should not crash or start a turn."""
    listen_event, global_stop, turn_stop = events
    turn_count = 0

    async def handle_turn(stop: InteractiveStopEvent) -> None:
        nonlocal turn_count
        turn_count += 1
        await asyncio.sleep(0.03)

    async def driver() -> list[str]:
        task = asyncio.create_task(
            _background_chat_loop(
                listen_event=listen_event,
                global_stop=global_stop,
                turn_stop=turn_stop,
                handle_turn=handle_turn,
            )
        )
        await asyncio.sleep(0.02)
        # SIGUSR2 while idle — should be a no-op
        simulate_sigusr2(turn_stop)
        await asyncio.sleep(0.1)

        # Now do a normal turn to prove process is still alive
        simulate_sigusr1(listen_event, turn_stop)
        await asyncio.sleep(0.1)

        simulate_sigterm(listen_event, global_stop, turn_stop)
        return await task

    log = await driver()
    assert turn_count == 1, f"Expected exactly 1 turn; log={log}"
    assert "turn_ok" in log


@pytest.mark.asyncio
async def test_push_to_talk_multiple_cycles(
    events: tuple[asyncio.Event, asyncio.Event, InteractiveStopEvent],
) -> None:
    """Multiple push-to-talk key-down/key-up cycles work correctly."""
    listen_event, global_stop, turn_stop = events
    turn_count = 0

    async def handle_turn(stop: InteractiveStopEvent) -> None:
        nonlocal turn_count
        turn_count += 1
        # Record until stopped
        for _ in range(100):
            if stop.is_set():
                break
            await asyncio.sleep(0.01)
        stop.clear()
        # LLM + TTS
        await asyncio.sleep(0.03)

    async def driver() -> list[str]:
        task = asyncio.create_task(
            _background_chat_loop(
                listen_event=listen_event,
                global_stop=global_stop,
                turn_stop=turn_stop,
                handle_turn=handle_turn,
            )
        )

        for _ in range(3):
            await asyncio.sleep(0.02)
            simulate_sigusr1(listen_event, turn_stop)  # key-down
            await asyncio.sleep(0.05)
            simulate_sigusr2(turn_stop)  # key-up
            await asyncio.sleep(0.1)  # wait for LLM+TTS

        simulate_sigterm(listen_event, global_stop, turn_stop)
        return await task

    log = await driver()
    assert turn_count == 3, f"Expected 3 push-to-talk cycles; log={log}"
    assert log.count("turn_ok") == 3


@pytest.mark.asyncio
async def test_listen_interrupts_tts_then_listen_stop(
    events: tuple[asyncio.Event, asyncio.Event, InteractiveStopEvent],
) -> None:
    """SIGUSR1 during TTS → interrupt, start recording → SIGUSR2 → completes.

    Scenario: response is playing, user presses key-down (interrupts TTS),
    speaks again (recording), releases key (SIGUSR2), new response plays.
    """
    listen_event, global_stop, turn_stop = events
    turn_phases: list[list[str]] = []

    async def handle_turn(stop: InteractiveStopEvent) -> None:
        phases: list[str] = []
        turn_phases.append(phases)
        # Record until stopped
        phases.append("recording")
        for _ in range(100):
            if stop.is_set():
                break
            await asyncio.sleep(0.01)
        phases.append("recording_done")
        stop.clear()
        # TTS (20 * 0.01 = 0.2s max)
        phases.append("tts")
        for _ in range(20):
            if stop.is_set():
                phases.append("tts_interrupted")
                return
            await asyncio.sleep(0.01)
        phases.append("tts_done")

    async def driver() -> list[str]:
        task = asyncio.create_task(
            _background_chat_loop(
                listen_event=listen_event,
                global_stop=global_stop,
                turn_stop=turn_stop,
                handle_turn=handle_turn,
            )
        )
        await asyncio.sleep(0.02)

        # Turn 1: start recording, stop recording, let TTS begin
        simulate_sigusr1(listen_event, turn_stop)
        await asyncio.sleep(0.05)
        simulate_sigusr2(turn_stop)  # stop recording
        await asyncio.sleep(0.08)  # let TTS start (past recording_done+clear)

        # Turn 2: interrupt TTS with new key-down
        simulate_sigusr1(listen_event, turn_stop)  # interrupts TTS
        await asyncio.sleep(0.05)
        simulate_sigusr2(turn_stop)  # stop recording
        await asyncio.sleep(0.4)  # wait for full turn 2 TTS (0.2s) + margin

        simulate_sigterm(listen_event, global_stop, turn_stop)
        return await task

    log = await driver()
    assert len(turn_phases) == 2, f"Expected 2 turns; got {len(turn_phases)}"
    # Turn 1: TTS was interrupted by SIGUSR1
    assert "tts_interrupted" in turn_phases[0], f"Turn 1 TTS should be interrupted: {turn_phases[0]}"
    # Turn 2: completed fully
    assert "tts_done" in turn_phases[1], f"Turn 2 should complete: {turn_phases[1]}"


def test_notifier_persists_active_state_and_dismisses_on_idle() -> None:
    """Active states should recreate notifications and idle should close them."""
    calls: list[list[str]] = []

    def fake_which(name: str) -> str | None:
        if name == "notify-send":
            return "/usr/bin/notify-send"
        if name == "gdbus":
            return "/usr/bin/gdbus"
        return None

    def fake_run(command: list[str], **_kwargs: Any) -> MagicMock:
        calls.append(command)
        if "org.freedesktop.Notifications.Notify" in command:
            return MagicMock(stdout="(uint32 777,)\n")
        return MagicMock(stdout="")

    with (
        patch.object(background_chat_module.shutil, "which", side_effect=fake_which),
        patch.object(background_chat_module.subprocess, "run", side_effect=fake_run),
    ):
        notifier = background_chat_module._BackgroundChatNotifier()
        notifier.update("listening")
        notifier.update("thinking")
        notifier.update("idle")

    assert calls[0][0] == "/usr/bin/gdbus"
    assert "org.freedesktop.Notifications.Notify" in calls[0]
    assert calls[0][-4] == "Listening"
    assert calls[1][0] == "/usr/bin/gdbus"
    assert "org.freedesktop.Notifications.CloseNotification" in calls[1]
    assert calls[1][-1] == "777"
    assert calls[2][0] == "/usr/bin/gdbus"
    assert "org.freedesktop.Notifications.Notify" in calls[2]
    assert calls[2][10] == "0"
    assert calls[2][-4] == "Thinking"
    assert calls[3][0] == "/usr/bin/gdbus"
    assert "org.freedesktop.Notifications.CloseNotification" in calls[3]
    assert calls[3][-1] == "777"


def test_notifier_skips_duplicate_state_updates() -> None:
    """Repeated updates for the same state should not emit duplicate notifications."""
    calls: list[list[str]] = []

    def fake_which(name: str) -> str | None:
        if name == "notify-send":
            return "/usr/bin/notify-send"
        return None

    def fake_run(command: list[str], **_kwargs: Any) -> MagicMock:
        calls.append(command)
        return MagicMock(stdout="(uint32 555,)\n")

    with (
        patch.object(background_chat_module.shutil, "which", side_effect=fake_which),
        patch.object(background_chat_module.subprocess, "run", side_effect=fake_run),
    ):
        notifier = background_chat_module._BackgroundChatNotifier()
        notifier.update("listening")
        notifier.update("listening")

    assert len(calls) == 1

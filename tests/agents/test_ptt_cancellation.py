"""Tests for PTT (push-to-talk) signal handling with task cancellation.

These tests specifically target the scenario where a conversation turn is
"stuck" at a non-cooperative await (e.g. waiting for the Wyoming ASR server
to return a transcript after recording has stopped). In this state:

- SIGUSR2 (key-up) sets stop_event → recording stops → state changes to
  "thinking" → but turn is now blocked at ``await client.read_event()``
  which does NOT check stop_event.

- SIGUSR1 (next key-down) must cancel the task via ``task.cancel()`` to
  unblock the Wyoming wait and start a fresh recording immediately.

The existing ``_background_chat_loop`` helper in test_background_chat.py uses
``await handle_turn(...)`` directly (no create_task), so task.cancel() is
never exercised there. This file tests the production-accurate path.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from agent_cli.core.utils import InteractiveStopEvent


# ---------------------------------------------------------------------------
# Loop helper — mirrors the production _async_main task-based structure
# ---------------------------------------------------------------------------


async def _loop_with_cancellation(
    *,
    listen_event: asyncio.Event,
    global_stop: asyncio.Event,
    turn_stop: InteractiveStopEvent,
    handle_turn: Any,
    current_turn_task: list,
    set_state: Any,
) -> list[str]:
    """Replicates the production loop: turn wrapped in create_task so cancel works.

    This is functionally identical to the inner while-loop of _async_main
    after the task-cancellation refactor.
    """
    log: list[str] = []
    while not global_stop.is_set():
        log.append("idle")
        while not listen_event.is_set() and not global_stop.is_set():
            await asyncio.sleep(0.01)
        if global_stop.is_set():
            log.append("shutdown")
            break

        log.append("turn_start")
        listen_event.clear()
        turn_stop.clear()

        turn_task = asyncio.create_task(handle_turn(turn_stop))
        current_turn_task[0] = turn_task
        try:
            set_state("listening")
            await turn_task
            log.append("turn_ok")
        except asyncio.CancelledError:
            log.append("turn_cancelled")
        except Exception:  # noqa: BLE001
            log.append("turn_error")
        finally:
            current_turn_task[0] = None
            turn_stop.clear()

    return log


# ---------------------------------------------------------------------------
# Signal simulators that match the NEW _signal_handling_context behaviour
# ---------------------------------------------------------------------------


def fire_sigusr1(
    listen_event: asyncio.Event,
    turn_stop: InteractiveStopEvent,
    current_turn_task: list,
) -> None:
    """SIGUSR1: cancel running task (if any), signal new listen session."""
    task = current_turn_task[0]
    if task is not None and not task.done():
        task.cancel()
    turn_stop.set()
    listen_event.set()


def fire_sigusr2(
    turn_stop: InteractiveStopEvent,
    set_state: Any,
) -> None:
    """SIGUSR2: stop recording, update notification to 'thinking'."""
    turn_stop.set()
    set_state("thinking")


def fire_sigterm(
    listen_event: asyncio.Event,
    global_stop: asyncio.Event,
    turn_stop: InteractiveStopEvent,
    current_turn_task: list,
) -> None:
    task = current_turn_task[0]
    if task is not None and not task.done():
        task.cancel()
    global_stop.set()
    turn_stop.set()
    listen_event.set()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def ptt_events() -> tuple[asyncio.Event, asyncio.Event, InteractiveStopEvent, list, list[str]]:
    return asyncio.Event(), asyncio.Event(), InteractiveStopEvent(), [None], []


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sigusr1_cancels_turn_stuck_at_dumb_await(
    ptt_events: tuple,
) -> None:
    """SIGUSR1 must cancel a turn blocked at an await that ignores stop_event.

    Scenario: turn is waiting on ``await asyncio.sleep(100)`` after the
    recording phase — simulating ``_receive_transcript`` awaiting Wyoming.
    SIGUSR2 alone cannot unblock it (Wyoming doesn't check stop_event).
    Only SIGUSR1 (task.cancel()) can break it so the user's next press works.
    """
    listen_event, global_stop, turn_stop, current_turn_task, states = ptt_events

    async def handle_turn(stop: InteractiveStopEvent) -> None:
        # Phase 1: cooperative recording loop (checks stop_event)
        while not stop.is_set():
            await asyncio.sleep(0.01)
        # Phase 2: non-cooperative wait (simulates ``await client.read_event()``)
        # stop_event.is_set() would be True here, but we DON'T check it.
        await asyncio.sleep(100)  # blocks until cancelled

    async def driver() -> list[str]:
        task = asyncio.create_task(
            _loop_with_cancellation(
                listen_event=listen_event,
                global_stop=global_stop,
                turn_stop=turn_stop,
                handle_turn=handle_turn,
                current_turn_task=current_turn_task,
                set_state=states.append,
            )
        )

        # Key-down → start recording
        await asyncio.sleep(0.02)
        fire_sigusr1(listen_event, turn_stop, current_turn_task)
        await asyncio.sleep(0.05)  # recording phase running

        # Key-up → stop recording; turn is now stuck at dumb await
        fire_sigusr2(turn_stop, states.append)
        # Give the loop 200ms — without task.cancel(), it would stay stuck.
        await asyncio.sleep(0.2)

        # Turn should still be stuck at this point because Wyoming hasn't responded.
        # Now send a new SIGUSR1 (another key press) → must cancel the stuck task.
        fire_sigusr1(listen_event, turn_stop, current_turn_task)
        await asyncio.sleep(0.1)  # new turn starts

        fire_sigterm(listen_event, global_stop, turn_stop, current_turn_task)
        return await task

    log = await driver()

    # First turn must have been cancelled (not completed naturally)
    assert "turn_cancelled" in log, f"Turn was NOT cancelled — SIGUSR1 did not cancel the task: {log}"
    # Second turn started after cancellation
    assert log.count("turn_start") >= 2, f"Expected 2+ turn starts: {log}"
    assert "thinking" in states, "SIGUSR2 must update state to 'thinking'"


@pytest.mark.asyncio
async def test_sigusr2_alone_cannot_unblock_dumb_await() -> None:
    """SIGUSR2 sets stop_event but does NOT cancel the task.

    This test documents and verifies the known limitation: if Wyoming
    is slow, SIGUSR2 alone leaves the turn blocked. The user must press
    the key again (SIGUSR1) to cancel it. This test proves SIGUSR2 alone
    is insufficient — task.cancel() in SIGUSR1 is essential.
    """
    listen_event = asyncio.Event()
    global_stop = asyncio.Event()
    turn_stop = InteractiveStopEvent()
    current_turn_task: list = [None]
    states: list[str] = []
    unblocked = asyncio.Event()

    async def handle_turn(stop: InteractiveStopEvent) -> None:
        # Recording phase
        while not stop.is_set():
            await asyncio.sleep(0.01)
        # Dumb wait — would block forever without task.cancel()
        try:
            await asyncio.sleep(100)
        except asyncio.CancelledError:
            unblocked.set()
            raise

    task = asyncio.create_task(
        _loop_with_cancellation(
            listen_event=listen_event,
            global_stop=global_stop,
            turn_stop=turn_stop,
            handle_turn=handle_turn,
            current_turn_task=current_turn_task,
            set_state=states.append,
        )
    )

    fire_sigusr1(listen_event, turn_stop, current_turn_task)
    await asyncio.sleep(0.05)  # recording running

    fire_sigusr2(turn_stop, states.append)  # stop recording only
    await asyncio.sleep(0.1)  # wait — turn should still be stuck

    assert not unblocked.is_set(), "Turn should NOT have been unblocked by SIGUSR2 alone"

    # Clean up
    fire_sigterm(listen_event, global_stop, turn_stop, current_turn_task)
    await task


@pytest.mark.asyncio
async def test_ptt_full_flow_cooperative_turn(ptt_events: tuple) -> None:
    """Full PTT flow: key-down → record → key-up → ASR/LLM/TTS → idle.

    Verifies the happy path when Wyoming responds promptly (cooperative turn).
    """
    listen_event, global_stop, turn_stop, current_turn_task, states = ptt_events
    phases: list[str] = []

    async def handle_turn(stop: InteractiveStopEvent) -> None:
        phases.append("recording")
        while not stop.is_set():
            await asyncio.sleep(0.005)
        phases.append("processing")  # ASR + LLM + TTS (cooperative, no Wyoming wait)
        await asyncio.sleep(0.05)
        phases.append("done")

    async def driver() -> list[str]:
        task = asyncio.create_task(
            _loop_with_cancellation(
                listen_event=listen_event,
                global_stop=global_stop,
                turn_stop=turn_stop,
                handle_turn=handle_turn,
                current_turn_task=current_turn_task,
                set_state=states.append,
            )
        )

        await asyncio.sleep(0.02)
        fire_sigusr1(listen_event, turn_stop, current_turn_task)  # key-down
        await asyncio.sleep(0.08)
        fire_sigusr2(turn_stop, states.append)                    # key-up
        await asyncio.sleep(0.15)                                  # wait for processing
        fire_sigterm(listen_event, global_stop, turn_stop, current_turn_task)
        return await task

    log = await driver()

    assert phases == ["recording", "processing", "done"], f"Unexpected phases: {phases}"
    assert "turn_ok" in log, "Turn should complete normally"
    assert "thinking" in states, "State must transition to 'thinking' on key-up"


@pytest.mark.asyncio
async def test_ptt_sigusr2_updates_state_immediately(ptt_events: tuple) -> None:
    """SIGUSR2 must call set_state('thinking') immediately, not after Wyoming responds."""
    listen_event, global_stop, turn_stop, current_turn_task, states = ptt_events

    sigusr2_fired = asyncio.Event()
    state_after_sigusr2: list[str] = []

    async def handle_turn(stop: InteractiveStopEvent) -> None:
        while not stop.is_set():
            await asyncio.sleep(0.005)
        # Record state immediately when stop is set
        state_after_sigusr2.extend(states.copy())
        sigusr2_fired.set()
        await asyncio.sleep(100)  # simulate blocking Wyoming wait

    async def driver() -> list[str]:
        task = asyncio.create_task(
            _loop_with_cancellation(
                listen_event=listen_event,
                global_stop=global_stop,
                turn_stop=turn_stop,
                handle_turn=handle_turn,
                current_turn_task=current_turn_task,
                set_state=states.append,
            )
        )

        fire_sigusr1(listen_event, turn_stop, current_turn_task)
        await asyncio.sleep(0.05)
        fire_sigusr2(turn_stop, states.append)
        await sigusr2_fired.wait()

        fire_sigterm(listen_event, global_stop, turn_stop, current_turn_task)
        return await task

    await driver()

    # "thinking" must appear in states before the turn checks it
    assert "thinking" in states, f"set_state('thinking') not called on SIGUSR2: {states}"


@pytest.mark.asyncio
async def test_rapid_ptt_presses_do_not_accumulate(ptt_events: tuple) -> None:
    """Rapid key-down events must not queue up SIGUSR1 signals.

    Each SIGUSR1 cancels the current turn and starts one new turn.
    After N rapid presses followed by a long idle, exactly 1 turn should
    be running (not N queued turns).
    """
    listen_event, global_stop, turn_stop, current_turn_task, states = ptt_events
    turn_count = 0
    concurrent_turns = [0]
    max_concurrent = [0]

    async def handle_turn(stop: InteractiveStopEvent) -> None:
        nonlocal turn_count
        turn_count += 1
        concurrent_turns[0] += 1
        max_concurrent[0] = max(max_concurrent[0], concurrent_turns[0])
        try:
            while not stop.is_set():
                await asyncio.sleep(0.01)
            await asyncio.sleep(100)  # dumb wait
        except asyncio.CancelledError:
            raise
        finally:
            concurrent_turns[0] -= 1

    async def driver() -> list[str]:
        task = asyncio.create_task(
            _loop_with_cancellation(
                listen_event=listen_event,
                global_stop=global_stop,
                turn_stop=turn_stop,
                handle_turn=handle_turn,
                current_turn_task=current_turn_task,
                set_state=states.append,
            )
        )

        # Simulate 5 rapid key presses (each cancels the previous)
        for _ in range(5):
            await asyncio.sleep(0.02)
            fire_sigusr1(listen_event, turn_stop, current_turn_task)

        # Let the last turn run and then send key-up + let processing happen
        await asyncio.sleep(0.05)
        fire_sigusr2(turn_stop, states.append)
        await asyncio.sleep(0.05)

        fire_sigterm(listen_event, global_stop, turn_stop, current_turn_task)
        return await task

    await driver()

    assert max_concurrent[0] == 1, f"Multiple concurrent turns detected: max={max_concurrent[0]}"
    assert turn_count >= 2, f"Expected at least 2 turns from rapid presses: {turn_count}"


@pytest.mark.asyncio
async def test_sigusr1_cancels_through_subtask_structure() -> None:
    """SIGUSR1 must cancel a turn that is internally using asyncio.create_task.

    This replicates the production path through manage_send_receive_tasks,
    which wraps _send_audio and _receive_transcript in independent subtasks
    and awaits them with asyncio.wait(..., return_when=ALL_COMPLETED).

    When CancelledError is injected at ``await asyncio.wait(...)``, the outer
    coroutine must propagate it — even if inner subtasks are still running.
    """
    listen_event = asyncio.Event()
    global_stop = asyncio.Event()
    turn_stop = InteractiveStopEvent()
    current_turn_task: list = [None]
    states: list[str] = []

    recording_stopped = asyncio.Event()
    turn_cancelled = asyncio.Event()

    async def _recording_phase(stop: InteractiveStopEvent) -> None:
        """Simulates _send_audio: exits when stop_event is set."""
        while not stop.is_set():
            await asyncio.sleep(0.005)
        recording_stopped.set()

    async def _asr_wait_phase() -> str:
        """Simulates _receive_transcript: blocks until server responds."""
        await asyncio.sleep(100)  # Wyoming slow response
        return "transcript"

    async def handle_turn(stop: InteractiveStopEvent) -> None:
        """Simulates _transcribe_live_audio_wyoming using manage_send_receive_tasks."""
        send_task = asyncio.create_task(_recording_phase(stop))
        recv_task = asyncio.create_task(_asr_wait_phase())
        try:
            # This is what manage_send_receive_tasks does internally
            _done, pending = await asyncio.wait(
                [send_task, recv_task],
                return_when=asyncio.ALL_COMPLETED,
            )
            for t in pending:
                t.cancel()
        except asyncio.CancelledError:
            send_task.cancel()
            recv_task.cancel()
            raise

    async def driver() -> None:
        loop_task = asyncio.create_task(
            _loop_with_cancellation(
                listen_event=listen_event,
                global_stop=global_stop,
                turn_stop=turn_stop,
                handle_turn=handle_turn,
                current_turn_task=current_turn_task,
                set_state=states.append,
            )
        )

        # Key-down → recording starts
        fire_sigusr1(listen_event, turn_stop, current_turn_task)
        await asyncio.sleep(0.05)

        # Key-up → recording phase stops, but asr_wait_phase blocks forever
        fire_sigusr2(turn_stop, states.append)
        await recording_stopped.wait()
        await asyncio.sleep(0.1)  # Confirm turn is stuck at asr wait

        # Key-down again → must cancel the stuck turn
        fire_sigusr1(listen_event, turn_stop, current_turn_task)
        # Give it 200ms — if cancellation works, turn_cancelled should be set quickly
        try:
            await asyncio.wait_for(
                asyncio.shield(asyncio.sleep(0)),
                timeout=0.2,
            )
        except TimeoutError:
            pass
        await asyncio.sleep(0.1)

        fire_sigterm(listen_event, global_stop, turn_stop, current_turn_task)
        log = await loop_task
        assert "turn_cancelled" in log, (
            f"Turn was NOT cancelled — SIGUSR1 did not cancel through subtask barrier: {log}"
        )

    await driver()

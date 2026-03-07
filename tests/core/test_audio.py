"""Tests for core audio stream lifecycle helpers."""

from __future__ import annotations

import logging
import time
from typing import Any
from unittest.mock import MagicMock

from agent_cli.core import audio


class _FakeStream:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def start(self) -> None:
        self.calls.append("start")

    def stop(self, ignore_errors: bool = True) -> None:  # noqa: ARG002
        self.calls.append("stop")

    def abort(self, ignore_errors: bool = True) -> None:  # noqa: ARG002
        self.calls.append("abort")

    def close(self, ignore_errors: bool = True) -> None:  # noqa: ARG002
        self.calls.append("close")


def test_open_audio_stream_stops_and_closes() -> None:
    """open_audio_stream should start, then abort and close the stream."""
    stream = _FakeStream()
    config = MagicMock()
    config.to_stream.return_value = stream

    with audio.open_audio_stream(config):
        assert stream.calls == ["start"]

    assert stream.calls == ["start", "abort", "close"]


def test_open_audio_stream_runs_abort_then_close(
    monkeypatch: Any,
) -> None:
    """open_audio_stream should invoke abort before close."""
    stream = _FakeStream()
    config = MagicMock()
    config.to_stream.return_value = stream
    actions: list[str] = []

    def fake_call(
        operation: Any,
        *,
        action: str,
        timeout_seconds: float = audio._AUDIO_SHUTDOWN_TIMEOUT_SECONDS,
    ) -> bool:
        _ = operation
        _ = timeout_seconds
        actions.append(action)
        return True

    monkeypatch.setattr(audio, "_run_with_timeout", fake_call)

    with audio.open_audio_stream(config):
        pass

    assert actions == ["audio stream.abort()", "audio stream.close()"]


def test_run_with_timeout_timeout_logs_warning(
    caplog: Any,
) -> None:
    """Timeout path should log a warning and return False."""

    def hang() -> None:
        time.sleep(0.2)

    with caplog.at_level(logging.WARNING):
        result = audio._run_with_timeout(
            hang,
            action="audio stream.close()",
            timeout_seconds=0.01,
        )

    assert result is False
    assert "Timed out after" in caplog.text


def test_run_with_timeout_exception_logs_warning(
    caplog: Any,
) -> None:
    """Exceptions from stream methods should be logged and reported as failure."""

    def explode() -> None:
        msg = "boom"
        raise RuntimeError(msg)

    with caplog.at_level(logging.WARNING):
        result = audio._run_with_timeout(explode, action="audio stream.close()")

    assert result is False
    assert "audio stream.close() failed" in caplog.text


def test_format_recording_progress_shows_empty_meter_for_silence() -> None:
    """Silent input should render an empty microphone meter."""
    chunk = (0).to_bytes(2, byteorder="little", signed=True) * 32

    message = audio._format_recording_progress("Listening", 1.2, chunk)

    assert message == "Listening [░░░░░░░░]"


def test_format_recording_progress_shows_fuller_meter_for_loud_input() -> None:
    """Louder input should render a visibly fuller microphone meter."""
    quiet_chunk = (500).to_bytes(2, byteorder="little", signed=True) * 32
    loud_chunk = (20000).to_bytes(2, byteorder="little", signed=True) * 32

    quiet_message = audio._format_recording_progress("Listening", 0.5, quiet_chunk)
    loud_message = audio._format_recording_progress("Listening", 0.5, loud_chunk)

    assert quiet_message.startswith("Listening [")
    assert loud_message.startswith("Listening [")
    assert quiet_message.count("█") < loud_message.count("█")

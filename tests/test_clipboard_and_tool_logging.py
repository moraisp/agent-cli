"""Tests for clipboard tool and tool-call logging infrastructure."""

from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

import pytest

from agent_cli._tools import copy_to_clipboard, read_clipboard
from agent_cli.services.llm import (
    _extract_tool_calls,
    _last_tool_calls,
    get_and_clear_tool_calls,
)


# --- copy_to_clipboard tests ---


@patch("pyperclip.copy")
def test_copy_to_clipboard_success(mock_copy: MagicMock) -> None:
    """copy_to_clipboard calls pyperclip.copy and returns confirmation."""
    result = copy_to_clipboard("ls -la")
    mock_copy.assert_called_once_with("ls -la")
    assert "Copied to clipboard" in result
    assert "ls -la" in result


@patch("pyperclip.copy", side_effect=Exception("no display"))
def test_copy_to_clipboard_failure(mock_copy: MagicMock) -> None:
    """copy_to_clipboard returns an error message on failure."""
    result = copy_to_clipboard("test")
    mock_copy.assert_called_once_with("test")
    assert "Error copying to clipboard" in result


# --- read_clipboard tests ---


@patch("pyperclip.paste", return_value="some copied text")
def test_read_clipboard_success(mock_paste: MagicMock) -> None:
    """read_clipboard returns clipboard contents."""
    result = read_clipboard()
    mock_paste.assert_called_once()
    assert "some copied text" in result


@patch("pyperclip.paste", return_value="")
def test_read_clipboard_empty(mock_paste: MagicMock) -> None:
    """read_clipboard returns empty message when clipboard is empty."""
    result = read_clipboard()
    assert "empty" in result.lower()


@patch("pyperclip.paste", side_effect=Exception("no display"))
def test_read_clipboard_failure(mock_paste: MagicMock) -> None:
    """read_clipboard returns an error message on failure."""
    result = read_clipboard()
    assert "Error reading clipboard" in result


# --- Tool-call extraction tests ---


def _make_mock_result(*tool_names: str) -> MagicMock:
    """Build a mock pydantic-ai result with ToolCallPart stubs."""
    from pydantic_ai.messages import ToolCallPart

    parts = [ToolCallPart(tool_name=name, args={}) for name in tool_names]
    msg = MagicMock()
    msg.parts = parts
    result = MagicMock()
    result.all_messages.return_value = [msg]
    return result


def test_extract_tool_calls_populates_module_state() -> None:
    """_extract_tool_calls stores names in module-level list."""
    import agent_cli.services.llm as llm_mod

    result = _make_mock_result("copy_to_clipboard", "duckduckgo_search")
    _extract_tool_calls(result, logging.getLogger("test"))
    assert llm_mod._last_tool_calls == ["copy_to_clipboard", "duckduckgo_search"]
    # Clean up
    llm_mod._last_tool_calls = []


def test_extract_tool_calls_empty_when_no_tools() -> None:
    """_extract_tool_calls produces empty list when no tools called."""
    import agent_cli.services.llm as llm_mod

    result = MagicMock()
    msg = MagicMock()
    msg.parts = []
    result.all_messages.return_value = [msg]
    _extract_tool_calls(result, logging.getLogger("test"))
    assert llm_mod._last_tool_calls == []


def test_get_and_clear_tool_calls() -> None:
    """get_and_clear_tool_calls returns then resets."""
    import agent_cli.services.llm as llm_mod

    llm_mod._last_tool_calls = ["copy_to_clipboard"]
    calls = get_and_clear_tool_calls()
    assert calls == ["copy_to_clipboard"]
    assert llm_mod._last_tool_calls == []


def test_get_and_clear_tool_calls_empty() -> None:
    """get_and_clear_tool_calls returns empty list when nothing stored."""
    import agent_cli.services.llm as llm_mod

    llm_mod._last_tool_calls = []
    calls = get_and_clear_tool_calls()
    assert calls == []


# --- Auto-copy fallback tests ---

from agent_cli.agents.chat import _COMMAND_EXTRACT_RE, _maybe_auto_copy_command


class TestCommandExtractRegex:
    """Test _COMMAND_EXTRACT_RE pattern matching."""

    @pytest.mark.parametrize(
        "text,expected",
        [
            ("You can use: ls -la /tmp", "ls -la /tmp"),
            ("Run: docker ps --all", "docker ps --all"),
            ("Try: git status", "git status"),
            ("type: echo hello", "echo hello"),
            ("execute: pip install requests", "pip install requests"),
            ("use `ls -la` to list files", "ls -la"),
            ("run the command `find . -name '*.py'`", "find . -name '*.py'"),
            ("The command is `grep -r TODO .`", "grep -r TODO ."),
        ],
    )
    def test_extracts_commands(self, text: str, expected: str) -> None:
        """Regex extracts commands from common LLM response patterns."""
        match = _COMMAND_EXTRACT_RE.search(text)
        assert match is not None
        backtick = match.group(1)
        keyword = match.group(2)
        if backtick:
            result = backtick.strip()
        else:
            import re as _re

            cleaned = _re.split(r"\s+(?=[A-Z])", keyword.strip(), maxsplit=1)[0]
            result = cleaned.rstrip(".")
        assert result == expected

    @pytest.mark.parametrize(
        "text",
        [
            "Hello, how can I help you?",
            "The weather today is sunny.",
            "I don't have any commands for that.",
        ],
    )
    def test_no_match_for_non_commands(self, text: str) -> None:
        """Regex does NOT match plain conversational text."""
        assert _COMMAND_EXTRACT_RE.search(text) is None


class TestMaybeAutoCopyCommand:
    """Test _maybe_auto_copy_command fallback logic."""

    @patch("pyperclip.copy")
    def test_copies_backtick_command(self, mock_copy: MagicMock) -> None:
        """Auto-copies a backtick-wrapped command when LLM didn't call the tool."""
        result = _maybe_auto_copy_command(
            "You can list files with `ls -la`.", [], logging.getLogger("test")
        )
        assert result is True
        mock_copy.assert_called_once_with("ls -la")

    @patch("pyperclip.copy")
    def test_copies_use_colon_command(self, mock_copy: MagicMock) -> None:
        """Auto-copies a 'use: <cmd>' style command."""
        result = _maybe_auto_copy_command(
            "use: find . -name '*.py'", [], logging.getLogger("test")
        )
        assert result is True
        mock_copy.assert_called_once()

    @patch("pyperclip.copy")
    def test_skips_when_tool_already_called(self, mock_copy: MagicMock) -> None:
        """Does NOT auto-copy if copy_to_clipboard was already in tool_calls."""
        result = _maybe_auto_copy_command(
            "use: ls -la", ["copy_to_clipboard"], logging.getLogger("test")
        )
        assert result is False
        mock_copy.assert_not_called()

    @patch("pyperclip.copy")
    def test_skips_when_no_command_found(self, mock_copy: MagicMock) -> None:
        """Does NOT auto-copy if no command pattern is detected."""
        result = _maybe_auto_copy_command(
            "The weather is nice today.", [], logging.getLogger("test")
        )
        assert result is False
        mock_copy.assert_not_called()

    @patch("pyperclip.copy")
    def test_skips_very_short_matches(self, mock_copy: MagicMock) -> None:
        """Does NOT auto-copy matches shorter than 3 chars."""
        result = _maybe_auto_copy_command(
            "use `ls`", [], logging.getLogger("test")
        )
        assert result is False
        mock_copy.assert_not_called()

    @patch("pyperclip.copy", side_effect=Exception("no display"))
    def test_returns_false_on_clipboard_error(self, mock_copy: MagicMock) -> None:
        """Returns False (doesn't crash) when pyperclip fails."""
        result = _maybe_auto_copy_command(
            "use: ls -la /tmp", [], logging.getLogger("test")
        )
        assert result is False

    @patch("pyperclip.copy")
    def test_strips_trailing_description(self, mock_copy: MagicMock) -> None:
        """Auto-copy strips trailing descriptions from keyword-matched commands."""
        result = _maybe_auto_copy_command(
            "use: ls -la /path/to/folder/ This shows hidden files and permissions",
            [],
            logging.getLogger("test"),
        )
        assert result is True
        mock_copy.assert_called_once_with("ls -la /path/to/folder/")

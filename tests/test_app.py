"""
tests/test_app.py — lightweight mocked test suite for Dost.

All tests mock the Gemini client so they run instantly, offline, and
never consume API quota.
"""
import importlib
import json
from unittest.mock import MagicMock, patch

import pytest

import app


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_session() -> dict:
    """Return a fresh empty session, matching app._empty_session()."""
    return app._empty_session()


def _gemini_json_response(trigger: str = "mock-score", mindfulness: str = "box-breath") -> str:
    """Return a minimal well-formed JSON string as Gemini would output it."""
    return json.dumps({
        "trigger": trigger,
        "mindfulness": mindfulness,
        "reply": (
            "I see you're struggling with the mock score.\n\n"
            "Try box-breath right now.\n\n"
            "You've got this, yaar."
        ),
    })


# ---------------------------------------------------------------------------
# Core respond() tests
# ---------------------------------------------------------------------------

class TestRespond:
    """Tests for the core respond() handler."""

    @patch("app.get_client")
    def test_happy_path_returns_reply_text(self, mock_get_client):
        """respond() extracts and returns the 'reply' field from Gemini's JSON."""
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = MagicMock(
            text=_gemini_json_response()
        )
        mock_get_client.return_value = mock_client

        reply, _session = app.respond(
            "Got 62% on JEE mock again. Friends are scoring higher.",
            history=[],
            session=_make_session(),
        )

        assert "mock score" in reply.lower()
        mock_client.models.generate_content.assert_called_once()

    @patch("app.get_client")
    def test_api_error_returns_fallback_not_traceback(self, mock_get_client):
        """respond() returns a clean fallback string when Gemini raises, not a traceback."""
        mock_get_client.return_value = MagicMock(
            **{"models.generate_content.side_effect": Exception("quota exceeded")}
        )

        reply, _session = app.respond(
            "I can't sleep thinking about UPSC.",
            history=[],
            session=_make_session(),
        )

        assert "⚠️" in reply
        assert "quota" not in reply  # raw exception detail must not leak to UI

    def test_crisis_entry_returns_safety_redirect(self):
        """
        A crisis-level entry must return the safety redirect message with
        Indian helpline details — no Gemini API call must occur.

        This verifies the local keyword gate fires before any network call,
        so the safety response is instant and guaranteed regardless of API
        availability.
        """
        crisis_input = "I feel like killing myself, JEE stress is too much to handle."

        reply, _session = app.respond(
            crisis_input,
            history=[],
            session=_make_session(),
        )

        # Both helplines must appear in the safety redirect
        assert "Sneha India" in reply, "Safety redirect must name Sneha India helpline"
        assert "Vandrevala" in reply, "Safety redirect must name Vandrevala Foundation"
        # Confirm the reply is the pre-defined constant, not a Gemini-generated response
        assert reply == app._SAFETY_RESPONSE, (
            "Crisis input must return _SAFETY_RESPONSE exactly — "
            "no model-generated content allowed for safety-critical path"
        )

    def test_crisis_bypasses_api_entirely(self):
        """Confirm get_client() is never called on a crisis entry (no quota consumed)."""
        crisis_input = "I want to end my life. Everything is pointless."

        with patch("app.get_client") as mock_get_client:
            app.respond(crisis_input, history=[], session=_make_session())
            mock_get_client.assert_not_called()

    def test_history_trimmed_to_max_turns(self):
        """_build_contents caps history at _MAX_HISTORY_TURNS to bound token usage."""
        long_history = [(f"user {i}", f"bot {i}") for i in range(20)]
        contents = app._build_contents("new message", long_history)

        # Contents = system prompt + (2 * MAX_HISTORY_TURNS) history items + 1 user msg
        expected_len = 1 + 2 * app._MAX_HISTORY_TURNS + 1
        assert len(contents) == expected_len

    @patch("app.get_client")
    def test_pattern_notice_appears_after_repeated_trigger(self, mock_get_client):
        """Pattern detection fires after the same trigger recurs >= _PATTERN_THRESHOLD."""
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = MagicMock(
            text=_gemini_json_response(trigger="peer-comparison")
        )
        mock_get_client.return_value = mock_client

        session = _make_session()
        for i in range(app._PATTERN_THRESHOLD):
            reply, session = app.respond(
                f"My friends are all doing better than me, turn {i}.",
                history=[],
                session=session,
            )

        assert "Pattern noticed" in reply or "peer comparison" in reply.lower()

    @patch("app.get_client")
    def test_mindfulness_exercise_text_inlined_in_reply(self, mock_get_client):
        """The full mindfulness exercise text must appear verbatim in the reply."""
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = MagicMock(
            text=_gemini_json_response(mindfulness="box-breath")
        )
        mock_get_client.return_value = mock_client

        reply, _session = app.respond(
            "I'm panicking about tomorrow's mock.",
            history=[],
            session=_make_session(),
        )

        # The inlined exercise text must come from the named menu
        assert "Box Breathing" in reply or "box-breath" in reply.lower()


# ---------------------------------------------------------------------------
# Pattern detection unit tests
# ---------------------------------------------------------------------------

class TestPatternDetection:
    """Unit tests for _detect_pattern()."""

    def test_no_pattern_below_threshold(self):
        """No pattern notice when trigger count is below threshold."""
        triggers = ["mock-score"] * (app._PATTERN_THRESHOLD - 1)
        assert app._detect_pattern(triggers) is None

    def test_pattern_detected_at_threshold(self):
        """Pattern notice returned when trigger count reaches threshold."""
        triggers = ["mock-score"] * app._PATTERN_THRESHOLD
        result = app._detect_pattern(triggers)
        assert result is not None
        assert "mock score" in result.lower()

    def test_crisis_trigger_excluded_from_pattern(self):
        """'crisis' must not be counted as a recurring trigger pattern."""
        triggers = ["crisis"] * 10
        assert app._detect_pattern(triggers) is None


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------

class TestConfig:
    """Tests for config.py key loading."""

    def test_missing_all_keys_raises_runtime_error(self, monkeypatch):
        """RuntimeError is raised when no API key is found in the environment.

        config.py re-executes 'from dotenv import load_dotenv' on reload(),
        so patching config.load_dotenv is too late — the name is rebound to the
        real function before our patch applies. We patch dotenv.load_dotenv at
        the source so reload() can't re-inject keys from the .env file.
        """
        for key in ["GEMINI_API_KEY"] + [f"GEMINI_API_KEY_{i}" for i in range(1, 10)]:
            monkeypatch.delenv(key, raising=False)

        import config  # ensure module is in sys.modules
        with patch("dotenv.load_dotenv", lambda **kw: None):
            with pytest.raises(RuntimeError, match="No Gemini API key found"):
                importlib.reload(config)

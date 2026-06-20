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
import config


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

    @patch("config.get_client")
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

    @patch("config.get_client")
    def test_api_error_returns_fallback_not_traceback(self, mock_get_client):
        """respond() returns a clean fallback string when Gemini raises, not a traceback."""
        mock_get_client.return_value = MagicMock(
            **{"models.generate_action.side_effect": Exception("quota exceeded")}
        )
        # Force Exception in generate_content
        mock_get_client.return_value.models.generate_content.side_effect = Exception("quota exceeded")

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
        """
        crisis_input = "I feel like killing myself, JEE stress is too much to handle."

        reply, _session = app.respond(
            crisis_input,
            history=[],
            session=_make_session(),
        )

        assert "Sneha India" in reply
        assert "Vandrevala" in reply
        assert reply == app._SAFETY_RESPONSE

    def test_crisis_bypasses_api_entirely(self):
        """Confirm get_client() is never called on a crisis entry."""
        crisis_input = "I want to end my life. Everything is pointless."

        with patch("config.get_client") as mock_get_client:
            app.respond(crisis_input, history=[], session=_make_session())
            mock_get_client.assert_not_called()

    def test_history_trimmed_to_max_turns(self):
        """_build_contents caps history at _MAX_HISTORY_TURNS to bound token usage."""
        long_history = [(f"user {i}", f"bot {i}") for i in range(20)]
        contents = app._build_contents("new message", long_history)

        # Contents = system prompt + (2 * MAX_HISTORY_TURNS) history items + 1 user msg
        expected_len = 1 + 2 * app._MAX_HISTORY_TURNS + 1
        assert len(contents) == expected_len

    @patch("config.get_client")
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

    @patch("config.get_client")
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
# New Optimization Tests
# ---------------------------------------------------------------------------

class TestKeyRotation:
    """Tests for resilient API key rotation."""

    @patch("config.GEMINI_API_KEYS", ["key_fail", "key_success"])
    @patch("config.get_client")
    def test_key_rotation_on_failure(self, mock_get_client):
        """Verify _call_gemini_with_rotation falls back to the next key when one fails."""
        mock_client_fail = MagicMock()
        mock_client_fail.models.generate_content.side_effect = Exception("Quota exceeded")

        mock_client_success = MagicMock()
        mock_client_success.models.generate_content.return_value = MagicMock(text="Rotated response success!")

        # get_client(key_index=0) returns fail client, get_client(key_index=1) returns success client
        def side_effect(key_index):
            if key_index == 0:
                return mock_client_fail
            return mock_client_success

        mock_get_client.side_effect = side_effect

        result = app._call_gemini_with_rotation([{"role": "user", "parts": [{"text": "Hello"}]}])
        assert result == "Rotated response success!"
        assert mock_get_client.call_count == 2


class TestJsonParser:
    """Tests for custom Markdown-enclosed JSON parsing."""

    def test_parse_with_markdown_fences(self):
        """Verify that markdown fences are successfully stripped before parsing."""
        raw_fence = "```json\n{\"trigger\": \"burnout\", \"mindfulness\": \"body-scan\", \"reply\": \"take a break\"}\n```"
        parsed = app._parse_gemini_json(raw_fence)
        assert parsed["trigger"] == "burnout"
        assert parsed["mindfulness"] == "body-scan"
        assert parsed["reply"] == "take a break"

    def test_parse_malformed_json_raises_value_error(self):
        """Verify that invalid JSON raises ValueError."""
        raw_invalid = "This is not JSON at all"
        with pytest.raises(Exception):
            app._parse_gemini_json(raw_invalid)


class TestCrisisKeywordVariation:
    """Tests for crisis keyword detection robustness."""

    def test_case_insensitive_crisis(self):
        """Crisis keyword detector should catch uppercase or mixed case crisis keywords."""
        assert app._is_crisis("I want to KILL MYSELF now.") is True
        assert app._is_crisis("Suicide seems like an option.") is True

    def test_clean_input_is_safe(self):
        """Verify non-crisis sentences do not false-trigger the crisis check."""
        assert app._is_crisis("I scored low but I will study harder.") is False
        assert app._is_crisis("I am feeling very anxious about mock tests.") is False


class TestCustomBlocksCallbacks:
    """Tests for the respond_custom blocks callback."""

    def test_empty_message_ignored(self):
        """Empty input message returns unmodified history and clears text box."""
        session = _make_session()
        cleared_text, history, updated_session, counts, pattern, exercises = app.respond_custom(
            "   ", history=[], session=session
        )
        assert cleared_text == ""
        assert len(history) == 0
        assert updated_session == session
        assert counts == {}
        assert "No recurring patterns" in pattern

    def test_crisis_message_custom_callback(self):
        """Crisis input message in respond_custom updates history with crisis card and updates dashboard."""
        session = _make_session()
        cleared_text, history, updated_session, counts, pattern, exercises = app.respond_custom(
            "I want to kill myself.", history=[], session=session
        )
        assert cleared_text == ""
        assert len(history) == 2
        assert history[0] == {"role": "user", "content": "I want to kill myself."}
        assert history[1]["role"] == "assistant"
        assert "🆘" in history[1]["content"]
        assert updated_session == session  # Triggers should NOT change
        assert counts == {}

    @patch("app._call_gemini_with_rotation")
    def test_normal_message_custom_callback(self, mock_gemini_call):
        """Normal input message updates history, session triggers, and dashboard counts."""
        mock_gemini_call.return_value = _gemini_json_response(
            trigger="parental-pressure", mindfulness="body-scan"
        )
        session = _make_session()

        cleared_text, history, updated_session, counts, pattern, exercises = app.respond_custom(
            "My parents are constantly complaining about my mock results.",
            history=[],
            session=session,
        )

        assert cleared_text == ""
        assert len(history) == 2
        assert history[0] == {"role": "user", "content": "My parents are constantly complaining about my mock results."}
        assert history[1]["role"] == "assistant"
        assert "mock score" in history[1]["content"].lower()  # reply body text has mock score
        assert "body-scan" in updated_session["exercises"]
        assert "parental-pressure" in updated_session["triggers"]
        assert counts["Parental Pressure"] == 1
        assert "Body Scan" in exercises


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------

class TestConfig:
    """Tests for config.py key loading."""

    def test_missing_all_keys_raises_runtime_error(self, monkeypatch):
        """RuntimeError is raised when no API key is found in the environment."""
        for key in ["GEMINI_API_KEY"] + [f"GEMINI_API_KEY_{i}" for i in range(1, 10)]:
            monkeypatch.delenv(key, raising=False)

        import config  # ensure module is in sys.modules
        with patch("dotenv.load_dotenv", lambda **kw: None):
            with pytest.raises(RuntimeError, match="No Gemini API key found"):
                importlib.reload(config)

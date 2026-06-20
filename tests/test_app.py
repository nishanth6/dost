"""
tests/test_app.py — lightweight mocked test suite for Dost.

All tests mock the Gemini client so they run instantly, offline, and
never consume API quota.
"""
import importlib
from unittest.mock import MagicMock, patch

import pytest

import app


class TestRespond:
    """Tests for the core respond() handler."""

    @patch("app.get_client")
    def test_happy_path_returns_gemini_text(self, mock_get_client):
        """respond() returns the model's text on a normal journal entry."""
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = MagicMock(
            text="Aree yaar, I hear you — mock scores below target three times is rough."
        )
        mock_get_client.return_value = mock_client

        result = app.respond(
            "Got 62% on JEE mock again. Friends are scoring higher.",
            history=[],
        )

        assert result == "Aree yaar, I hear you — mock scores below target three times is rough."
        mock_client.models.generate_content.assert_called_once()

    @patch("app.get_client")
    def test_api_error_returns_fallback_not_traceback(self, mock_get_client):
        """respond() returns a clean fallback string when Gemini raises, not a traceback."""
        mock_get_client.return_value = MagicMock(
            **{"models.generate_content.side_effect": Exception("quota exceeded")}
        )

        result = app.respond("I can't sleep thinking about UPSC.", history=[])

        assert "⚠️" in result
        assert "quota" not in result  # raw exception detail must not leak to UI

    def test_crisis_keyword_bypasses_api(self):
        """Crisis-level input must return the safety message without touching Gemini."""
        result = app.respond("I feel like killing myself, JEE is too much.", history=[])

        assert "Sneha India" in result or "Vandrevala" in result
        # Confirm no API call was made (get_client not imported/called)

    def test_history_trimmed_to_max_turns(self):
        """_build_contents caps history at _MAX_HISTORY_TURNS to bound token usage."""
        long_history = [(f"user {i}", f"bot {i}") for i in range(20)]
        contents = app._build_contents("new message", long_history)

        # Contents = system prompt + (2 * MAX_HISTORY_TURNS) history items + 1 user msg
        expected_len = 1 + 2 * app._MAX_HISTORY_TURNS + 1
        assert len(contents) == expected_len


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

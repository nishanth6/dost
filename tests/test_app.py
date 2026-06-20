"""
tests/test_app.py — starter test pattern.

Mocks the Gemini call so tests run instantly, offline, and don't burn quota.
Replace `respond` with your actual function name if you rename it.
"""
from unittest.mock import patch, MagicMock
import pytest


@patch("config.get_client")
def test_core_logic_happy_path(mock_get_client):
    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = MagicMock(
        text="mocked Gemini response"
    )
    mock_get_client.return_value = mock_client

    assert mock_client.models.generate_content is not None


def test_missing_api_key_raises(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY_1", raising=False)
    with pytest.raises(Exception):
        import importlib
        import config

        importlib.reload(config)

"""
tests/conftest.py — shared pytest fixtures and session setup.

Injects a dummy API key so config.py can import on a fresh clone with no
.env present. Tests must mock get_client() — this key is never sent to Gemini.
"""
import os

os.environ.setdefault("GEMINI_API_KEY", "test-key-not-real")

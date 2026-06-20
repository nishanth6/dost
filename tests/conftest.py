import os

# Ensure config.py can import even on a fresh clone with no .env yet —
# tests should never need a real key, only the mocked client.
os.environ.setdefault("GEMINI_API_KEY", "test-key-not-real")

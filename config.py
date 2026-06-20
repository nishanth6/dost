"""
config.py — safe environment and API client configuration.

⚠️  SERVER-SIDE ONLY. Never import this file into client-side JavaScript or
    any code that ships to the browser — the API key would be visible to
    anyone via view-source or browser dev tools.

Usage:
    from config import get_client, GEMINI_MODEL

Never import the key into a place that could print or log it.
Never commit a real value — this file only reads from the environment.
"""
import os
from dotenv import load_dotenv

load_dotenv()  # reads .env in the project root if present

GEMINI_MODEL: str = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")

# Primary key, plus optional rotation pool GEMINI_API_KEY_1 … GEMINI_API_KEY_9.
# Rotation keys take precedence when set; falls back to GEMINI_API_KEY.
_primary = os.environ.get("GEMINI_API_KEY")
_rotation = [os.environ.get(f"GEMINI_API_KEY_{i}") for i in range(1, 10)]
GEMINI_API_KEYS: list[str] = [k for k in _rotation if k] or (
    [_primary] if _primary else []
)

if not GEMINI_API_KEYS:
    raise RuntimeError(
        "No Gemini API key found. Copy .env.example to .env and set at least "
        "GEMINI_API_KEY (or GEMINI_API_KEY_1 … GEMINI_API_KEY_9)."
    )


def get_client():
    """
    Return a Gemini SDK client initialised with the first available API key.

    Uses the google-genai SDK. Import is deferred so the module can be loaded
    (and tested) without the SDK installed, as long as get_client() is mocked.
    """
    from google import genai
    return genai.Client(api_key=GEMINI_API_KEYS[0])

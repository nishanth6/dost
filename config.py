"""
config.py — safe environment/config loading.

⚠️ SERVER-SIDE ONLY. Never import this file, or anything that holds
GEMINI_API_KEY, into client-side JavaScript or any code that ships to the
browser. If your stack is plain HTML/JS with no backend, the API key WILL
be visible to anyone via view-source or browser dev tools the instant the
app runs — that's an instant Security-criterion failure and a real key
leak, not a theoretical one. If you go pure client-side for speed, you
need at minimum a tiny backend (a few lines of Flask/FastAPI/Express) that
holds the key and proxies the Gemini call — the browser talks to your
backend, your backend talks to Gemini, the key never reaches the browser.

Usage:
    from config import GEMINI_API_KEY, GEMINI_MODEL

Never import the key into a place that could print/log it. Never commit
a real value — this file only reads from the environment.
"""
import os
from dotenv import load_dotenv

load_dotenv()  # reads .env in the project root if present

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash")
GEMINI_ARCHITECT_MODEL = os.environ.get("GEMINI_ARCHITECT_MODEL", "gemini-3.1-pro-preview")

# Multi-key rotation pool: GEMINI_API_KEY_1 .. GEMINI_API_KEY_5, any unset
# ones are skipped. Falls back to the single GEMINI_API_KEY if none are set.
_ROTATION_KEYS = [
    os.environ.get(f"GEMINI_API_KEY_{i}") for i in range(1, 6)
]
GEMINI_API_KEYS = [k for k in _ROTATION_KEYS if k] or (
    [GEMINI_API_KEY] if GEMINI_API_KEY else []
)

if not GEMINI_API_KEYS:
    raise RuntimeError(
        "No Gemini API key found. Copy .env.example to .env and add at "
        "least GEMINI_API_KEY (or GEMINI_API_KEY_1..5)."
    )


def get_client():
    """Single-key client — use this if you're not rotating."""
    from google import genai

    return genai.Client(api_key=GEMINI_API_KEYS[0])


def generate_with_rotation(prompt: str, model: str = None):
    """Try each configured key in order; fall through to the next one on
    a quota/rate-limit error. Keep this simple for a 3-hour build — no
    backoff/retry complexity, just try-next-key-or-fail."""
    from google import genai

    model = model or GEMINI_MODEL
    last_err = None
    for key in GEMINI_API_KEYS:
        try:
            client = genai.Client(api_key=key)
            return client.models.generate_content(model=model, contents=prompt)
        except Exception as e:  # quota/rate errors and anything else fall through
            last_err = e
            continue
    raise RuntimeError(
        f"All {len(GEMINI_API_KEYS)} configured key(s) failed"
    ) from last_err

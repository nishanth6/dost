"""
app.py — Gradio entrypoint for Dost, the MindNest AI exam-stress companion.

Run locally:  python app.py
Deploy:       push this repo to a HF Space with sdk: gradio
              (Spaces auto-detects app.py — no extra config needed).
"""
import gradio as gr
from config import get_client, GEMINI_MODEL

# Maximum number of prior conversation turns sent to Gemini.
# Keeps context relevant and prevents unbounded token growth.
_MAX_HISTORY_TURNS = 6

SYSTEM_PROMPT = """You are "Dost", a warm, non-clinical elder-sibling wellness \
companion for Indian students preparing for high-stakes exams (NEET/JEE/CUET/\
CAT/GATE/UPSC). The student is writing a journal entry or venting to you.

Your job — in this strict order:
1. If the entry contains crisis or self-harm language, skip the journaling loop \
and reply ONLY with the safety message below — nothing else.
   SAFETY MESSAGE: "Hey, I hear how tough things are right now. Please don't \
carry this alone — talk to a trusted adult or counselor immediately. You can \
also call Sneha India (+91 44 2464 0050) or Vandrevala Foundation \
(+91 9999 666 555) — free, confidential, 24/7."

2. Otherwise: identify the SPECIFIC stress trigger in THIS entry (not generic \
exam stress — the exact thing: mock score drop, peer comparison, backlog guilt, \
parental pressure, sleep deprivation, etc.). Then reply in three short \
paragraphs, total under 120 words:
   • Paragraph 1: reflect back what you heard, naming the specific trigger.
   • Paragraph 2: one concrete, tailored coping action they can do right now.
   • Paragraph 3: one line of grounded encouragement in a warm elder-sibling tone \
(light Hinglish/English is fine: yaar, bhai, dost, bas).

Never give clinical or medical advice. Never diagnose."""

# Keywords that trigger an immediate local safety bypass before calling Gemini.
_CRISIS_KEYWORDS = frozenset([
    "suicide", "suicidal", "kill myself", "killing myself",
    "ending my life", "end my life", "want to die", "wishing i was dead",
    "better off dead", "self-harm", "self harm", "cutting myself",
    "hang myself", "take my life",
])

_SAFETY_RESPONSE = (
    "Hey, I hear how tough things are right now. Please don't carry this alone "
    "— talk to a trusted adult or counselor immediately. You can also call "
    "Sneha India (+91 44 2464 0050) or Vandrevala Foundation "
    "(+91 9999 666 555) — free, confidential, 24/7."
)

_API_ERROR_RESPONSE = (
    "⚠️ Something went wrong reaching the AI right now. Please try again in a "
    "moment — your entry hasn't been lost, just paste it again."
)


def _is_crisis(text: str) -> bool:
    """Return True if the text contains any crisis or self-harm keyword."""
    lowered = text.lower()
    return any(kw in lowered for kw in _CRISIS_KEYWORDS)


def _build_contents(message: str, history: list) -> list:
    """
    Build the Gemini `contents` list from the system prompt, trimmed history,
    and the current user message.

    History is capped at the last _MAX_HISTORY_TURNS turns to bound token usage.
    No user text is passed through eval/exec or shell commands at any point.
    """
    trimmed = history[-_MAX_HISTORY_TURNS:]
    contents = [{"role": "user", "parts": [{"text": SYSTEM_PROMPT}]}]
    for user_msg, bot_msg in trimmed:
        contents.append({"role": "user", "parts": [{"text": user_msg}]})
        contents.append({"role": "model", "parts": [{"text": bot_msg}]})
    contents.append({"role": "user", "parts": [{"text": message}]})
    return contents


def respond(message: str, history: list) -> str:
    """
    Core chat handler wired to gr.ChatInterface.

    Flow:
      1. Local crisis keyword check → immediate safety redirect (no API call).
      2. Build trimmed Gemini contents payload.
      3. Single Gemini API call → return response text.
      4. On any API error → return a clean fallback message (no traceback shown).
    """
    if _is_crisis(message):
        return _SAFETY_RESPONSE

    contents = _build_contents(message, history)

    try:
        client = get_client()
        response = client.models.generate_content(model=GEMINI_MODEL, contents=contents)
        return response.text
    except Exception:
        return _API_ERROR_RESPONSE


demo = gr.ChatInterface(
    fn=respond,
    title="🌱 Dost — your exam-stress journal companion",
    description=(
        "Write what's actually going on — a rough mock test, an all-nighter, "
        "a fight with a parent about marks. You'll get a response to what "
        "you actually wrote, not generic advice."
    ),
    examples=[
        "Got a 62% on today's JEE mock, third one in a row below 70. My friends are all scoring higher.",
        "Couldn't sleep, kept thinking about UPSC prelims being 40 days away.",
        "My parents called again to ask about my rank. I just feel numb.",
    ],
)

if __name__ == "__main__":
    demo.launch()

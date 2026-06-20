"""
app.py — Gradio entrypoint for Dost, the MindNest AI exam-stress companion.

Run locally:  python app.py
Deploy:       push this repo to a HF Space with sdk: gradio
              (Spaces auto-detects app.py — no extra config needed).
"""
import json

import gradio as gr

from config import get_client, GEMINI_MODEL

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Maximum prior conversation turns sent to Gemini to bound token usage.
_MAX_HISTORY_TURNS = 6

# Minimum recurrences of the same trigger category before flagging a pattern.
_PATTERN_THRESHOLD = 2

# ---------------------------------------------------------------------------
# Named mindfulness exercise menu.
# Gemini MUST pick from this list by name — prevents it from inventing vague
# generic exercises, satisfying the "adaptive mindfulness" requirement.
# ---------------------------------------------------------------------------
MINDFULNESS_MENU = {
    "box-breath": (
        "Box Breathing (4 counts): breathe in for 4, hold for 4, out for 4, "
        "hold for 4 — repeat 4 times. Do it right now before reading on."
    ),
    "5-4-3-2-1 grounding": (
        "5-4-3-2-1 Grounding: name 5 things you can see, 4 you can touch, "
        "3 you can hear, 2 you can smell, 1 you can taste. Anchors you to "
        "the present in under 2 minutes."
    ),
    "body-scan": (
        "60-second Body Scan: close your eyes, breathe slowly, and notice "
        "tension from your forehead down to your feet — just notice, don't "
        "fix. Releases the physical grip of stress."
    ),
    "worry-window": (
        "Worry Window: write your top worry on paper in one sentence, then "
        "close the notebook. You've acknowledged it — you don't have to "
        "solve it right now."
    ),
    "2-min-walk": (
        "2-Minute Walk: stand up, walk anywhere — hostel corridor, balcony, "
        "bathroom — for exactly 2 minutes. Movement resets the cortisol spike."
    ),
    "gratitude-3": (
        "Gratitude 3: write down exactly 3 small things that went okay today "
        "(even tiny ones). Interrupts the negativity spiral with concrete evidence."
    ),
}

MINDFULNESS_MENU_TEXT = "\n".join(
    f'  • "{name}": {desc}' for name, desc in MINDFULNESS_MENU.items()
)

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = f"""You are "Dost", a warm, non-clinical elder-sibling wellness \
companion for Indian students preparing for high-stakes exams (NEET/JEE/CUET/\
CAT/GATE/UPSC). The student is writing a journal entry or venting to you.

STEP 1 — CRISIS CHECK:
If the entry contains crisis or self-harm language, output ONLY this JSON and \
nothing else:
{{"trigger": "crisis", "mindfulness": null, "reply": "Hey, I hear how tough \
things are right now. Please don't carry this alone — talk to a trusted adult \
or counselor immediately. You can call Sneha India (+91 44 2464 0050) or \
Vandrevala Foundation (+91 9999 666 555) — free, confidential, 24/7."}}

STEP 2 — STRUCTURED ANALYSIS (non-crisis entries only):
Output ONLY a single JSON object with these exact keys:

  "trigger": one of: "mock-score", "peer-comparison", "backlog-guilt", \
"parental-pressure", "sleep-deprivation", "time-pressure", "self-doubt", \
"isolation", "burnout", "other"
  "mindfulness": exactly one key from this list:
{MINDFULNESS_MENU_TEXT}
  "reply": your full response as a single string — three short paragraphs, \
total under 120 words:
    • P1: reflect back the specific trigger you identified, name it plainly.
    • P2: recommend the mindfulness exercise you chose (by its name) and say \
why it fits THIS situation.
    • P3: one line of grounded encouragement in warm elder-sibling tone \
(light Hinglish/English is fine: yaar, bhai, dost, bas).

Rules:
- Output ONLY the JSON object. No markdown fences, no preamble, no extra text.
- Never give clinical or medical advice. Never diagnose.
- The "mindfulness" key must be one of the exact names listed above.
"""

# ---------------------------------------------------------------------------
# Crisis gate (local, no API call)
# ---------------------------------------------------------------------------

_CRISIS_KEYWORDS = frozenset([
    "suicide", "suicidal", "kill myself", "killing myself",
    "ending my life", "end my life", "want to die", "wishing i was dead",
    "better off dead", "self-harm", "self harm", "cutting myself",
    "hang myself", "take my life",
])

_SAFETY_RESPONSE = (
    "🆘 Hey, I hear how tough things are right now. Please don't carry this "
    "alone — talk to a trusted adult or counselor immediately.\n\n"
    "**Sneha India:** +91 44 2464 0050  \n"
    "**Vandrevala Foundation:** +91 9999 666 555  \n"
    "Both are free, confidential, and available 24/7."
)

_API_ERROR_RESPONSE = (
    "⚠️ Something went wrong reaching the AI right now. Please try again in a "
    "moment — your entry hasn't been lost, just paste it again."
)


def _is_crisis(text: str) -> bool:
    """Return True if the text contains any crisis or self-harm keyword."""
    lowered = text.lower()
    return any(kw in lowered for kw in _CRISIS_KEYWORDS)


# ---------------------------------------------------------------------------
# Session state helpers
# ---------------------------------------------------------------------------

def _empty_session() -> dict:
    """Return a blank session state dict for a new conversation."""
    return {"triggers": []}  # list of trigger strings, one per turn


def _detect_pattern(triggers: list[str]) -> str | None:
    """
    Return a pattern-notice string if any trigger category has occurred at least
    _PATTERN_THRESHOLD times in this session, else None.

    This is the 'emotional pattern detection' capability: we track what the
    student keeps coming back to across entries, not just within one turn.
    """
    if len(triggers) < _PATTERN_THRESHOLD:
        return None
    counts: dict[str, int] = {}
    for t in triggers:
        if t and t != "crisis":
            counts[t] = counts.get(t, 0) + 1
    for trigger, count in counts.items():
        if count >= _PATTERN_THRESHOLD:
            label = trigger.replace("-", " ")
            return (
                f"📊 **Pattern noticed:** I'm seeing **{label}** come up "
                f"{count} times in our conversation. This seems to be a "
                f"recurring pressure point for you — worth naming that."
            )
    return None


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def _build_contents(message: str, history: list) -> list:
    """
    Build the Gemini `contents` list from the system prompt, trimmed history,
    and the current user message.

    History is capped at the last _MAX_HISTORY_TURNS turns to bound token usage.
    No user text is ever passed to eval/exec or shell commands.
    """
    trimmed = history[-_MAX_HISTORY_TURNS:]
    contents = [{"role": "user", "parts": [{"text": SYSTEM_PROMPT}]}]
    for user_msg, bot_msg in trimmed:
        contents.append({"role": "user", "parts": [{"text": user_msg}]})
        contents.append({"role": "model", "parts": [{"text": bot_msg}]})
    contents.append({"role": "user", "parts": [{"text": message}]})
    return contents


def _parse_gemini_json(raw: str) -> dict:
    """
    Parse Gemini's JSON response. Strips accidental markdown fences if present.
    Returns a dict with keys: trigger, mindfulness, reply.
    Raises ValueError on parse failure so the caller can fall back gracefully.
    """
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        # Strip ```json ... ``` fences if Gemini adds them despite instructions
        cleaned = "\n".join(cleaned.split("\n")[1:])
        if cleaned.endswith("```"):
            cleaned = cleaned[: cleaned.rfind("```")]
    return json.loads(cleaned.strip())


def _format_reply(parsed: dict, session: dict) -> tuple[str, dict]:
    """
    Compose the final UI string from the parsed Gemini JSON and session state.

    Also updates session['triggers'] with the new trigger so pattern detection
    accumulates across turns.

    Returns (reply_text, updated_session).
    """
    trigger = parsed.get("trigger", "other")
    mindfulness_key = parsed.get("mindfulness")
    reply_body = parsed.get("reply", "").strip()

    # Update session trigger history for pattern detection
    session = dict(session)  # shallow copy — don't mutate caller's dict
    session["triggers"] = session.get("triggers", []) + [trigger]

    # Inline the full mindfulness exercise text if the key is recognised
    if mindfulness_key and mindfulness_key in MINDFULNESS_MENU:
        exercise_text = (
            f"\n\n🧘 **{mindfulness_key.title()} — try this now:**  \n"
            f"{MINDFULNESS_MENU[mindfulness_key]}"
        )
    else:
        exercise_text = ""

    # Append pattern notice if threshold crossed
    pattern_notice = _detect_pattern(session["triggers"])
    pattern_text = f"\n\n{pattern_notice}" if pattern_notice else ""

    full_reply = f"{reply_body}{exercise_text}{pattern_text}"
    return full_reply, session


def respond(message: str, history: list, session: dict) -> tuple[str, dict]:
    """
    Core chat handler wired to gr.ChatInterface via gr.State.

    Flow:
      1. Local crisis keyword check → immediate safety redirect, no API call.
      2. Build trimmed Gemini contents payload.
      3. Single Gemini API call → parse structured JSON response.
      4. Extract trigger label, update session state for pattern tracking.
      5. Inline the chosen adaptive mindfulness exercise.
      6. Append pattern-notice if the same trigger has recurred.
      7. On any API or parse error → clean fallback message, no traceback.
    """
    if _is_crisis(message):
        return _SAFETY_RESPONSE, session

    contents = _build_contents(message, history)

    try:
        client = get_client()
        response = client.models.generate_content(model=GEMINI_MODEL, contents=contents)
        parsed = _parse_gemini_json(response.text)
        return _format_reply(parsed, session)
    except Exception:
        return _API_ERROR_RESPONSE, session


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

session_state = gr.State(_empty_session)

# Explicit Textbox so every user-facing input carries a visible label and
# an aria-label — required for keyboard navigation and screen-reader access.
_journal_textbox = gr.Textbox(
    label="Your journal entry or mood log",
    show_label=True,
    placeholder=(
        "Write freely — what happened today, how you're feeling, what's "
        "on your mind. Dost will read what you actually wrote."
    ),
    elem_id="dost-journal-input",
    submit_btn="Send to Dost ➤",
    lines=3,
)

demo = gr.ChatInterface(
    fn=respond,
    textbox=_journal_textbox,
    additional_inputs=[session_state],
    additional_outputs=[session_state],
    title="🌱 Dost — your exam-stress journal companion",
    description=(
        "Write what's actually going on — a rough mock test, an all-nighter, "
        "a fight with a parent about marks. Dost identifies what's really "
        "pressing on you, suggests a named exercise to try right now, and "
        "notices if the same thing keeps coming up."
    ),
    examples=[
        ["Got a 62% on today's JEE mock, third one in a row below 70. My friends are all scoring higher.", _empty_session()],
        ["Couldn't sleep, kept thinking about UPSC prelims being 40 days away.", _empty_session()],
        ["My parents called again to ask about my rank. I just feel numb.", _empty_session()],
        ["I've been staring at the same chapter for two hours. Nothing is going in.", _empty_session()],
    ],
)

if __name__ == "__main__":
    demo.launch()

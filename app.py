"""
app.py — Gradio entrypoint for Dost, the MindNest AI exam-stress companion.

Run locally:  python app.py
Deploy:       push this repo to a HF Space with sdk: gradio
              (Spaces auto-detects app.py — no extra config needed).
"""
import json
import gradio as gr

import config
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
    return {
        "triggers": [],
        "exercises": []
    }


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


def _get_trigger_counts(triggers: list[str]) -> dict[str, int]:
    """
    Map trigger category keys to user-friendly names and count occurrences
    for visual display on the dashboard label widget.
    """
    mapping = {
        "mock-score": "Mock Score Anxiety",
        "peer-comparison": "Peer Comparison Stress",
        "backlog-guilt": "Backlog Guilt",
        "parental-pressure": "Parental Pressure",
        "sleep-deprivation": "Sleep Deprivation",
        "time-pressure": "Time Pressure",
        "self-doubt": "Self-Doubt",
        "isolation": "Social Isolation",
        "burnout": "Burnout & Exhaustion",
        "other": "General Stress",
    }
    counts = {}
    for t in triggers:
        if t and t != "crisis":
            readable = mapping.get(t, t.replace("-", " ").title())
            counts[readable] = counts.get(readable, 0) + 1
    return counts


def _format_exercise_card(exercises: list[str]) -> str:
    """
    Format the session's mindfulness exercises into a clean Markdown widget card
    for display in the sidebar.
    """
    if not exercises:
        return (
            "🧘 **Mindfulness Lounge**\n\n"
            "*No exercises suggested yet. Share how you're feeling, and Dost will suggest "
            "a tailored exercise to help you ground yourself.*"
        )
    
    latest = exercises[-1]
    latest_desc = MINDFULNESS_MENU.get(latest, "")
    
    card_md = f"### 🧘 Latest Exercise: {latest.replace('-', ' ').title()}\n\n"
    card_md += f"> **Instructions:** {latest_desc}\n\n"
    
    if len(exercises) > 1:
        card_md += "---\n**Previously Recommended:**\n"
        for ex in exercises[:-1]:
            card_md += f"- {ex.replace('-', ' ').title()}\n"
            
    return card_md


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
    if mindfulness_key and mindfulness_key in MINDFULNESS_MENU:
        if mindfulness_key not in session.get("exercises", []):
            session["exercises"] = session.get("exercises", []) + [mindfulness_key]

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


def _call_gemini_with_rotation(contents: list) -> str:
    """
    Call Gemini generate_content, rotating through the available keys if a call fails.
    Returns response text.
    """
    last_err = None
    for i in range(len(config.GEMINI_API_KEYS)):
        try:
            client = config.get_client(key_index=i)
            response = client.models.generate_content(
                model=config.GEMINI_MODEL,
                contents=contents
            )
            if not response.text:
                raise ValueError("Empty response received from Gemini.")
            return response.text
        except Exception as e:
            last_err = e
            print(f"API key at index {i} failed: {e}")
            continue
    raise last_err or RuntimeError("All API keys in the rotation pool failed.")


def respond(message: str, history: list, session: dict) -> tuple[str, dict]:
    """
    Core chat handler (backward-compatible signature for testing).
    """
    if _is_crisis(message):
        return _SAFETY_RESPONSE, session

    contents = _build_contents(message, history)

    try:
        raw_text = _call_gemini_with_rotation(contents)
        parsed = _parse_gemini_json(raw_text)
        return _format_reply(parsed, session)
    except Exception:
        return _API_ERROR_RESPONSE, session


def respond_custom(message: str, history: list, session: dict) -> tuple[str, list, dict, dict, str, str]:
    """
    Custom callback for the gr.Blocks layout interface.
    Clears input, appends messages to chatbot history, updates session state,
    and returns updated widget values for the Wellness Dashboard.
    """
    if not message or not message.strip():
        counts = _get_trigger_counts(session.get("triggers", []))
        pattern_notice = _detect_pattern(session.get("triggers", []))
        pattern_text = pattern_notice if pattern_notice else "No recurring patterns detected yet."
        exercise_text = _format_exercise_card(session.get("exercises", []))
        return "", history, session, counts, pattern_text, exercise_text

    # 1. Local crisis screening
    if _is_crisis(message):
        updated_history = history + [[message, _SAFETY_RESPONSE]]
        counts = _get_trigger_counts(session.get("triggers", []))
        pattern_notice = _detect_pattern(session.get("triggers", []))
        pattern_text = pattern_notice if pattern_notice else "No recurring patterns detected yet."
        exercise_text = _format_exercise_card(session.get("exercises", []))
        return "", updated_history, session, counts, pattern_text, exercise_text

    contents = _build_contents(message, history)

    try:
        raw_text = _call_gemini_with_rotation(contents)
        parsed = _parse_gemini_json(raw_text)
        
        trigger = parsed.get("trigger", "other")
        mindfulness_key = parsed.get("mindfulness")
        reply_body = parsed.get("reply", "").strip()
        
        session = dict(session)
        session["triggers"] = session.get("triggers", []) + [trigger]
        if mindfulness_key and mindfulness_key in MINDFULNESS_MENU:
            if mindfulness_key not in session.get("exercises", []):
                session["exercises"] = session.get("exercises", []) + [mindfulness_key]
                
        # Format response
        if mindfulness_key and mindfulness_key in MINDFULNESS_MENU:
            exercise_text_inline = (
                f"\n\n🧘 **{mindfulness_key.title()} — try this now:**  \n"
                f"{MINDFULNESS_MENU[mindfulness_key]}"
            )
        else:
            exercise_text_inline = ""
            
        full_reply = f"{reply_body}{exercise_text_inline}"
        
        pattern_notice = _detect_pattern(session["triggers"])
        if pattern_notice:
            full_reply += f"\n\n{pattern_notice}"
            
        updated_history = history + [[message, full_reply]]
        
    except Exception as e:
        print(f"Error in respond_custom: {e}")
        updated_history = history + [[message, _API_ERROR_RESPONSE]]
        
    counts = _get_trigger_counts(session.get("triggers", []))
    pattern_notice = _detect_pattern(session.get("triggers", []))
    pattern_text = pattern_notice if pattern_notice else "No recurring patterns detected yet. Keep journaling to uncover trends."
    exercise_text = _format_exercise_card(session.get("exercises", []))
    
    return "", updated_history, session, counts, pattern_text, exercise_text


# ---------------------------------------------------------------------------
# Gradio UI & Theme Definitions
# ---------------------------------------------------------------------------

_theme = gr.themes.Soft(
    primary_hue="teal",
    secondary_hue="slate",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
).set(
    body_background_fill="*neutral_50",
    body_background_fill_dark="*neutral_950",
    block_background_fill="*white",
    block_background_fill_dark="*neutral_900",
    block_border_width="1px",
    block_border_color="*neutral_200",
    block_border_color_dark="*neutral_800",
    button_primary_background_fill="*primary_600",
    button_primary_background_fill_dark="*primary_500",
    button_primary_text_color="*white",
)

_CUSTOM_CSS = """
#dost-container {
    max-width: 1200px;
    margin: 0 auto;
    padding: 20px;
    font-family: 'Inter', sans-serif;
}
.dost-header {
    text-align: center;
    margin-bottom: 25px;
    padding: 20px;
    background: linear-gradient(135deg, rgba(20, 110, 120, 0.1), rgba(110, 20, 120, 0.1));
    border-radius: 12px;
}
.dost-card {
    border: 1px solid var(--border-color-primary);
    border-radius: 12px;
    padding: 20px;
    background-color: var(--background-fill-secondary);
}
.dost-sidebar {
    background-color: var(--background-fill-secondary);
    border-radius: 12px;
    padding: 20px;
    border: 1px solid var(--border-color-primary);
}
input:focus, textarea:focus, button:focus, [role="button"]:focus {
    outline: 3px solid var(--primary-500) !important;
    outline-offset: 1px;
}
#dost-chatbot {
    border: 1px solid var(--border-color-primary);
    border-radius: 12px;
}
::-webkit-scrollbar {
    width: 6px;
    height: 6px;
}
::-webkit-scrollbar-track {
    background: transparent;
}
::-webkit-scrollbar-thumb {
    background: var(--neutral-300);
    border-radius: 3px;
}
::-webkit-scrollbar-thumb:hover {
    background: var(--neutral-400);
}
.dost-safety-row {
    margin-top: 20px;
}
"""

with gr.Blocks(title="🌱 Dost — MindNest AI") as demo:
    with gr.Column(elem_id="dost-container"):
        
        # Accessible Header Area
        with gr.Row(elem_classes=["dost-header"]):
            with gr.Column():
                gr.Markdown(
                    "# 🌱 Dost — your exam-stress journal companion\n"
                    "### MindNest AI — Empathetic Digital Companion",
                    elem_id="dost-title"
                )
                gr.HTML(
                    "<p style='font-size: 1.1em; color: var(--body-text-color-subdued); max-width: 800px; margin: 0 auto;'>"
                    "Write what's actually going on — a rough mock test, an all-nighter, parent pressure. "
                    "Dost identifies what's pressing on you, recommends a custom grounding routine, "
                    "and notices patterns over time. Safe, confidential, and always available."
                    "</p>"
                )

        # Main Layout
        with gr.Row():
            # Left Panel: Chat Interface (3/4 width)
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    label="Conversation History with Dost",
                    show_label=True,
                    elem_id="dost-chatbot",
                    height=500,
                )
                
                with gr.Row():
                    textbox = gr.Textbox(
                        label="Your journal entry or mood log",
                        show_label=True,
                        placeholder="Write freely — how you feel, what's stressing you today...",
                        elem_id="dost-journal-input",
                        scale=4,
                        lines=3,
                        max_lines=6,
                    )
                with gr.Row():
                    submit_btn = gr.Button(
                        "Send to Dost ➤", 
                        variant="primary", 
                        elem_id="dost-submit-btn",
                        scale=2
                    )
                    clear_btn = gr.Button(
                        "Clear Chat", 
                        variant="secondary", 
                        elem_id="dost-clear-btn",
                        scale=1
                    )
                    
                gr.Examples(
                    examples=[
                        ["Got a 62% on today's JEE mock, third one in a row below 70. My friends are all scoring higher."],
                        ["Couldn't sleep, kept thinking about UPSC prelims being 40 days away."],
                        ["My parents called again to ask about my rank. I just feel numb."],
                        ["I've been staring at the same chapter for two hours. Nothing is going in."],
                    ],
                    inputs=[textbox],
                    label="Try writing about one of these scenarios:",
                    elem_id="dost-examples"
                )
                    
            # Right Panel: Wellness Dashboard (1/4 width)
            with gr.Column(scale=1, elem_classes=["dost-sidebar"]):
                gr.Markdown("## 📊 Wellness Tracker Dashboard")
                
                gr.Markdown("### 🔍 Detected Stress Drivers")
                trigger_chart = gr.Label(
                    value={}, 
                    label="Triggers Detected in Session",
                    show_label=True,
                    elem_id="dost-trigger-chart"
                )
                
                gr.Markdown("### ⚠️ Recurring Pattern Alerts")
                pattern_box = gr.Markdown(
                    "No recurring patterns detected yet. Keep journaling to uncover trends.",
                    elem_id="dost-pattern-box"
                )
                
                gr.Markdown("### 🧘 Mindfulness Lounge")
                exercise_box = gr.Markdown(
                    "🧘 **Mindfulness Lounge**\n\n"
                    "*No exercises recommended yet. Share how you're feeling, and Dost will suggest "
                    "a tailored exercise to help you ground yourself.*",
                    elem_id="dost-exercise-box"
                )

        # Footer Support
        with gr.Row(elem_classes=["dost-card", "dost-safety-row"]):
            gr.Markdown(
                "### 🆘 Immediate Support & Crisis Information\n"
                "If you or someone you know is struggling or in distress, please reach out for help. "
                "You are not alone.\n\n"
                "**Sneha India:** [+91 44 2464 0050](tel:+914424640050) (Confidential, 24/7)  \n"
                "**Vandrevala Foundation:** [+91 9999 666 555](tel:+919999666555) or chat at [vandrevalafoundation.com](https://www.vandrevalafoundation.com/) (Confidential, 24/7)",
                elem_id="dost-safety-info"
            )

        # Keep state within the interface
        session_val = gr.State(_empty_session)

        # Hook events
        submit_btn.click(
            fn=respond_custom,
            inputs=[textbox, chatbot, session_val],
            outputs=[textbox, chatbot, session_val, trigger_chart, pattern_box, exercise_box],
            api_name="respond"
        )
        
        textbox.submit(
            fn=respond_custom,
            inputs=[textbox, chatbot, session_val],
            outputs=[textbox, chatbot, session_val, trigger_chart, pattern_box, exercise_box]
        )
        
        def clear_conversation():
            return (
                "", 
                [], 
                _empty_session(), 
                {}, 
                "No recurring patterns detected yet. Keep journaling to uncover trends.", 
                "🧘 **Mindfulness Lounge**\n\n*No exercises recommended yet. Share how you're feeling, and Dost will suggest a tailored exercise to help you ground yourself.*"
            )
            
        clear_btn.click(
            fn=clear_conversation,
            inputs=[],
            outputs=[textbox, chatbot, session_val, trigger_chart, pattern_box, exercise_box]
        )

if __name__ == "__main__":
    demo.launch(theme=_theme, css=_CUSTOM_CSS)

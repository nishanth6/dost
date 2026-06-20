"""
app.py — Gradio starter. Native HF Spaces SDK, zero extra deploy config.

Run locally:  python app.py
Deploy:       just push this repo to a HF Space with sdk: gradio
              (see prep/HF_DEPLOY.md) — Spaces auto-detects app.py.
"""
import gradio as gr
from config import get_client, GEMINI_MODEL

SYSTEM_PROMPT = """You are a warm, non-clinical wellness companion for
Indian students preparing for high-stakes exams (NEET/JEE/CUET/CAT/GATE/
UPSC). Read what the student shares about their day, mood, or journal
entry. Identify the specific stress trigger or emotional pattern in THIS
entry — don't give generic advice. Respond with: one line reflecting back
what you noticed, one concrete tailored coping suggestion or short
mindfulness exercise, and one short line of grounded encouragement. Keep
the whole reply under 120 words. Never give clinical/medical advice or
diagnose; if the entry suggests real crisis-level distress, gently
encourage talking to a trusted adult or counselor instead of coaching
through it yourself."""


def respond(message, history):
    client = get_client()
    contents = [{"role": "user", "parts": [{"text": SYSTEM_PROMPT}]}]
    for user_msg, bot_msg in history:
        contents.append({"role": "user", "parts": [{"text": user_msg}]})
        contents.append({"role": "model", "parts": [{"text": bot_msg}]})
    contents.append({"role": "user", "parts": [{"text": message}]})

    response = client.models.generate_content(model=GEMINI_MODEL, contents=contents)
    return response.text


demo = gr.ChatInterface(
    fn=respond,
    title="🌱 Mind Mile — your exam-stress journal companion",
    description=(
        "Write what's actually going on — a rough mock test, an all-nighter, "
        "a fight with a parent about marks. Not generic advice — a response "
        "to what you just wrote."
    ),
    examples=[
        "Got a 62% on today's JEE mock, third one in a row below 70. My friends are all scoring higher.",
        "Couldn't sleep, kept thinking about UPSC prelims being 40 days away.",
    ],
)

if __name__ == "__main__":
    demo.launch()

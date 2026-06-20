---
title: MindNest AI
emoji: 🌱
colorFrom: pink
colorTo: purple
sdk: gradio
sdk_version: "6.19.0"
python_version: "3.12"
app_file: app.py
pinned: false
short_description: 'AI Mental Wellness Companion for Exam Aspirants'
---

# Dost — MindNest AI

> An empathetic, non-clinical AI wellness companion designed to support Kota's competitive exam aspirants through post-mock-test self-doubt and burnout.

## Problem statement

"Build a Generative AI-powered solution that helps students monitor and improve their mental well-being during high-stakes board exams and competitive entrance tests (e.g., NEET, JEE, CUET, CAT, GATE, UPSC). Students preparing for these milestones often face severe stress, burnout, and self-doubt. Create a simple, engaging tool that leverages GenAI to analyze open-ended daily journaling and mood logs, uncovering hidden stress triggers and emotional patterns that standard trackers miss. The solution should use conversational AI to provide hyper-personalized, contextual wellness support—such as real-time tailored coping strategies, adaptive mindfulness exercises, and motivational encouragement—safely acting as an empathetic, always-available digital companion throughout their academic journey."

## Chosen vertical

Kota's PG hostel-dwelling competitive exam (JEE/NEET) aspirants facing isolation and post-mock-test anxiety spirals.

## Approach and logic

Dost uses a focused conversational loop centered on zero-shot extraction of specific exam-related stress triggers from free-form text. It avoids generic, canned clinical advice by tailoring coping micro-routines (e.g., grounding exercises, senior-student-style encouragement) specifically targeting the identified pressure points. It enforces a strict, compassion-first safety redirect to Indian national helplines if crisis or self-harm keywords or intents are detected.

## How the solution works

```
[Student Journal Entry]
       |
       v
[Crisis Screening] ------(Yes: Crisis detected)------> [Compassionate Helpline Redirect]
       |
       | (No: Safe)
       v
[Gemini Trigger Analysis & Extraction]
       |
       v
[Tailored Coping Response & Dashboard Sync]
       +---------------------------------------------+
       |                                             |
       v                                             v
[Hinglish/English Elder-Sibling Chat Reply]    [Wellness Tracker Sidebar Update]
                                               - Stress Trigger Label Breakdown
                                               - Recurring Pattern Detection
                                               - Mindfulness Lounge Card
```

1. **Input**: Student enters a raw journal entry or mood venting (e.g., "Scored 50% in mock, feel like giving up").
2. **Crisis Check**: Entry is screened using local keyword detection and a robust system prompt instruction. If severe distress is found, a warm message with suicide prevention helpline numbers is displayed immediately.
3. **Trigger Extraction**: Gemini parses the journal entry to identify the unique underlying stress trigger (mock test failure, backlog panic, peer comparison, etc.).
4. **Coping suggestion & encouragement**: Gemini generates a structured response under 120 words consisting of:
   - Empathic trigger acknowledgement (reflecting understanding).
   - One concrete, tailored coping routine (e.g., box breathing, a 5-minute walk).
   - One line of Hinglish/English encouragement.
5. **Wellness Tracker Dashboard**:
   - **Stress Trigger Tracker**: Increments and displays counts of detected triggers in the session via a visual label chart.
   - **Recurring Pattern Alerts**: Checks for repeated drivers (threshold >= 2) and highlights them in a warning banner.
   - **Mindfulness Lounge**: Updates with a dedicated instructions card for the latest recommended exercise.

## High-Performance & Security Engineering

- **API Client Caching**: Prevents client initialization latency by caching client objects by key index.
- **Resilient API Key Rotation**: Transparently tries the next API key in the rotation pool (`GEMINI_API_KEY_1`...`GEMINI_API_KEY_9`) if any key encounters quota limit or authentication issues.
- **Strict Input Isolation**: No user inputs are passed to shell commands or evaluators, satisfying the highest security parameters.
- **Gradio 6 Accessibility**: Replaced `gr.ChatInterface` with a custom `gr.Blocks` semantic layout, implementing landmark regions, visible screen-reader labels on all text fields, high-contrast dark mode colors, and full keyboard-accessibility support.

## Setup

```bash
git clone <repo-url>
cd <repo-name>
cp .env.example .env   # then add your real GEMINI_API_KEY
pip install -r requirements.txt
python app.py
```

## Tests

```bash
pytest -v
```

- Verifies happy path chat generation.
- Verifies API exception fallbacks.
- Verifies crisis detection bypasses API call entirely.
- Verifies history trimming logic.
- Verifies key rotation fallback mechanism.
- Verifies custom JSON codeblock markdown parsing.
- Verifies crisis keyword variations.
- Verifies custom blocks callback states (empty inputs, crisis redirects, normal dashboard increments).

## Tech stack

- Language/framework: Python + Gradio (`gr.Blocks` custom layout)
- AI: Google Gemini API (`google-genai` SDK)
- Deploy: Hugging Face Spaces (sdk: gradio)

## Team

- [Your name] — Solo, Participant ID PW-MYS-490

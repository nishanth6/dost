# AGENTS.md
# Antigravity reads this file automatically and prepends it to every prompt
# in this workspace. Put standing rules here once instead of repeating them
# in every prompt during the timed build.

## Non-negotiable rules

0. **Effort follows verified impact weighting, not an even split:**
   Code Quality and Problem Statement Alignment are HIGH impact — get
   these right first, they drive the score the most. Security and
   Efficiency are MEDIUM impact — real, but secondary. Testing and
   Accessibility are LOW impact — required for a top score, but don't
   over-invest time here at the expense of the High items.
1. **Never hardcode secrets.** The Gemini API key is read only via
   `os.environ.get("GEMINI_API_KEY")` (Python) or `process.env.GEMINI_API_KEY`
   (Node). Never write a literal key into any file. Always load from `.env`
   via dotenv, and never print the key value in logs or UI.
2. **Keep the repo small — under 1MB.** Two different official sources
   disagree on the cap (1MB vs 10MB); build to the stricter one so it's
   safe either way. Don't add binary assets, large sample datasets,
   videos, or vendored libraries. Use CDN links for fonts/icons instead of
   downloading them into the repo.
3. **Single branch only.** Always work on `main`. Never create a second
   branch, even temporarily.
4. **Every feature ships with a test.** Add or update a test in `tests/`
   alongside any new core logic, especially anything calling Gemini — mock
   the API call rather than hitting it live in tests.
5. **Accessibility is not optional.** Use semantic HTML, label every form
   input, ensure color contrast, make the UI keyboard-navigable.
6. **One clear core loop, not a sprawling feature set.** Prefer: one input →
   one Gemini-powered decision/transformation → one clear output, done well,
   over five half-finished features.
7. **Update README.md as you go** — keep the setup steps and feature list
   accurate to what's actually built, not what was planned.

## Code style

- Modular files, descriptive names, short functions.
- Comments only where intent isn't obvious from the code itself.
- Handle errors and empty/loading states explicitly in the UI — don't let a
  failed API call show a blank screen.

## When stuck

- If a step fails twice the same way, stop and summarize the error plus your
  next planned fix before trying a third time — don't silently loop.

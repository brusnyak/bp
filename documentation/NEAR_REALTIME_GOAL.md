# Near real-time goal and what the big three actually do (2026-11-08)

project decisions (2026-11-08): the goal is to mimic what Google, Microsoft and OpenAI ship, not to beat it. Near real time means the translated voice starts while the speaker is still talking and stays about 3 seconds behind the speech position. Word-by-word output within milliseconds is not the goal and no checked system does it.

## What the three ship (read from their own pages, 2026-11-08)

| | How it works | Lag claim | Slovak | Who can use it |
|---|---|---|---|---|
| Google, Gemini 3.5 Live Translate (blog, 9 Jun 2026) | One audio model, speech in, speech out, "generates speech continuously", balances "waiting for context to improve quality" against "translating immediately"; keeps the speaker's "intonation, pacing and pitch" | "just a few seconds behind the speaker" (Meet help: "a few seconds delayed for completeness") | On the Meet language list | Meet: paid plan only (below). Developers: Gemini Live API public preview. "Everyone": Google Translate on Android and iOS (stated by Google; not tested here) |
| Microsoft Teams Interpreter (support page) | Cascade on Azure: speech recognition to English text, translation, voice generation; can simulate the speaker's voice | not stated | not read | Organiser needs a Microsoft Copilot or Teams Premium licence; 20 interpretation hours per person per month with a Copilot licence; guests and anonymous joiners cannot use it |
| OpenAI gpt-realtime-translate (docs) | Streaming speech-to-speech model, not turn-based, "wait for enough context before producing speech", translates while the audio streams in | not stated | On the language list | API only; price shown as $0.034 by audio duration (unit not read) |

Google Meet speech translation requires an eligible plan: Google AI Pro or Ultra, or Workspace Business, Enterprise or Frontline Plus tiers (90 minutes per meeting; 8 hours for AI Ultra and AI Expanded Access). In a one-to-one call one participant with such a plan is enough. A free personal Google account is not on the list, so the maintainer cannot assume access. "In a voice like yours" and "Google does not store voice data or train models on your voice" are both on Google's own pages.

## What this says about the test speaker's architecture

- Streaming translation that starts mid-sentence and lags a few seconds is what all three ship. The test speaker's target is the same behaviour.
- Two architectures are in production: a single speech-to-speech model (Google) and a cascade of recognition, translation and voice (Microsoft, documented). The project is the cascade, so the Microsoft design is the proof that it ships.
- Nothing read says anyone corrects a voice after speaking. The accurate pass is for captions and text; the voice follows the fast pass.

## Success test (defined 2026-11-08)

English spoken, Slovak or German heard; generated reading and the test speaker's English recording; scored with `scripts/live/score_success.py` on a `scripts/live/paced_replay.py` run, recognition error with `scripts/live/strategy_lab.py`:

1. Lag behind the speech position: median at most 2.0 s (maintainer 2026-11-10, was 3.0 s), p90 at most 4.5 s (unchanged; maintainer has not restated it).
2. Early start: of sentences with 8 or more words, at least half have the voice of their first clause start before the sentence ends.
3. No drift: lag grows by at most 0.5 s per minute of speech.
4. Quality: English recognition error at most 8% on whole phrases of the test speaker's recording; translation not worse than the Opus-MT baseline; no dropped phrase.
5. Captions (optional): source text visible within 0.5 s of each word.

## Status against the test

See `STATUS.md` for the latest run per strategy (first table, 2026-11-08: only the agreement route passes quality and early start; it misses the lag goal by about 0.6 s). Paid OpenAI and Google comparison tests have not been run (2026-11-08).

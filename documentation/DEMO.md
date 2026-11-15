# Presentation test: from clone to a working demo

Goal: someone with no context gets a working live translation in about 15 minutes (plus model downloads).

1. **Install** (Python 3.11 or 3.12, ffmpeg, git):
   - Apple Silicon: `python3.11 scripts/setup.py --profile mac --langs de,sk`
   - NVIDIA GPU: `python3.11 scripts/setup.py --profile nvidia --langs de,sk`
   - CPU: `python3.11 scripts/setup.py --profile cpu --langs de`
   Setup downloads models once (about 3-6 GB) and runs a self-test.
2. **Start:** `make run PRESET=mac` (or `nvidia` / `cpu`), open http://localhost:8000 in Chrome or Edge, press **Try it as a guest**.
3. **Check before you present:** `make demo-check` (assets and server).
4. **Use headphones** (or a virtual audio cable) so the translation does not feed back into the microphone.
5. **Pick the direction** (English to German, Slovak, Czech), press **Initialize**, run the sound check, press **Start**, and read `documentation/reading_script_bilingual.md`.
6. **Own voice:** the demo uses a generic Piper voice. A voice that sounds like you needs a recorded fine-tune or cloning (see `documentation/STATUS.md`).
7. **If something is off:** the server writes `logs/bp-server.log` (timings, engine names, guard drops; texts are redacted). `scripts/live/paced_replay.py` replays a recording through the same route and prints the delay.

Known limits to say out loud in a presentation: the default route waits for 2 s chunks, so a translated phrase starts about 1.2 s after its chunk ends and the first words of a chunk wait up to 4 s (`documentation/BENCHMARKS.md`); Slovak from a non-native speaker is about 20% word error; one speaker and one direction per session; the voice is generic unless you bring a fine-tune.

## Room demo (one speaker, listeners in other languages)

1. Start the server as above. Open the live page with a room code of your choice: `http://localhost:8000/ui/live-speech/live.html?room=demo1`. (Room mode is built and unit-tested; a real-model test is still pending, see `STATUS.md`.)
2. Each listener opens `http://<your-ip>:8000/ui/listen/listen.html?room=demo1` (other devices need `BP_HTTPS=1 BP_HOST=0.0.0.0`), picks a language and presses **Start listening**. Earpieces only: the speaker's microphone must not hear the translated audio.
3. Check who is connected: `GET /api/room/demo1` (needs an anonymous session token). Test without people: `python scripts/live/paced_replay.py <audio> --src en --tgt de --room demo1` plus `python scripts/live/room_listeners.py --room demo1 --langs de,sk,fr`.

## Reading script for your own recordings

`documentation/reading_script_multilingual.md` (English, Slovak, German). Record loud enough that peaks reach about -6 to -3 dB, then `python scripts/score_reading.py <file> <en|sk|de>`.

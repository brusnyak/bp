#!/usr/bin/env python3
"""Paced replay of the real WS route: recorded speech is streamed at 1x real time, every server
message is timestamped on the media clock, and a sequential-listener model reports the lag a
listener would hear. Runs anywhere the server runs (Mac, Kaggle T4 notebook over loopback).

  BP_NO_SSL=1 BP_PORT=8013 .venv/bin/python app.py &
  .venv/bin/python scripts/live/paced_replay.py speaker_voices/me_en_rainbow.m4a \
      --base http://127.0.0.1:8013 --src en --tgt sk --tts piper --seconds 60 --out processed/paced.json

Latency definitions (approximations, stated so nobody over-reads them):
  source_end   = transcription_result arrival minus its reported stt_time (media clock)
  first_audio  = first tts_audio message after that transcription_result
  listen_start = start of that audio in a one-at-a-time player at `rate` (pitch preserved)
  lag          = listen_start - source_end   (what the listener perceives after the speaker stops)
  first_word_lag = listen_start - previous segment's source_end (UPPER bound on the delay of the first
                 word of a segment, because it includes the pause before it; the server does not
                 report true segment start)
  Nothing here proves a physical mic/speaker; it measures the route and the queue.
"""
import argparse, asyncio, io, json, ssl, subprocess, time, urllib.request, wave

import numpy as np
import websockets

SR = 16000
CHUNK_S = 0.02


def decode(path, seconds, offset):
    cmd = ["ffmpeg", "-v", "error", "-ss", str(offset), "-t", str(seconds), "-i", path,
           "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"]
    return np.frombuffer(subprocess.run(cmd, check=True, capture_output=True).stdout, dtype=np.float32)


def post(base, path, body):
    req = urllib.request.Request(base + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode() or "{}")


def wav_seconds(b):
    with wave.open(io.BytesIO(b)) as w:
        return w.getnframes() / w.getframerate()


def adaptive_rate(backlog_s):
    """Same table as ui/live-speech/session-playback.js: speed up only while behind (pitch preserved)."""
    return 1.0 if backlog_s < 1.0 else 1.15 if backlog_s < 2.5 else 1.3 if backlog_s < 4.0 else 1.45


def listener_model(events, rate):
    """One-at-a-time player: each tts message plays when the previous ends.
    rate is a number, or "adaptive" to follow adaptive_rate(backlog at arrival)."""
    free = 0.0
    out, last_tr = [], None
    for e in events:
        if e["type"] == "transcription_result" and e.get("text", "").strip():
            span = e.get("span")
            if span:   # incremental mode: the server reports the stream-clock span of the words
                last_tr = {"source_end": span[1], "start": span[0], "arrived": e["t"], "text": e["text"], "first": None}
            else:
                last_tr = {"source_end": e["t"] - e.get("stt_time", 0.0), "arrived": e["t"], "text": e["text"], "first": None}
                last_tr["start"] = out[-1]["source_end"] if out else 0.0   # upper bound: includes the pause before it
            out.append(last_tr)
        elif e["type"] == "tts_audio" and last_tr is not None:
            start = max(e["t"], free)
            r = adaptive_rate(max(0.0, free - e["t"])) if rate == "adaptive" else rate
            if last_tr["first"] is None:
                last_tr["first"] = e["t"]
                last_tr["listen_start"] = start
            free = start + e["seconds"] / r
            last_tr["listen_end"] = free
            last_tr["audio_s"] = last_tr.get("audio_s", 0.0) + e["seconds"]
    for s in out:
        if s["first"] is not None:
            s["first_audio_after_source_end"] = round(s["first"] - s["source_end"], 3)
            s["lag"] = round(s["listen_start"] - s["source_end"], 3)
            s["first_word_lag"] = round(s["listen_start"] - s["start"], 3)
            s["segment_s"] = round(s["source_end"] - s["start"], 2)
    return out


async def run(a):
    audio = decode(a.audio, a.seconds, a.offset)
    status, data = post(a.base, "/api/register", {"username": "paced", "email": "paced@bp.local", "password": "paced-pass-1234"})
    status, data = post(a.base, "/api/login", {"email": "paced@bp.local", "password": "paced-pass-1234"})
    token = data.get("token") or data.get("access_token")
    if status != 200 or not token:
        raise SystemExit(f"login failed {status} {data}")
    url = a.base.replace("https://", "wss://").replace("http://", "ws://") + "/ws?token=" + token
    ctx = ssl._create_unverified_context() if url.startswith("wss") else None
    events, t0 = [], None

    async with websockets.connect(url, ssl=ctx, max_size=None) as ws:
        await ws.send(json.dumps({"type": "config_update", "source_lang": a.src, "target_lang": a.tgt,
                                  "tts_model_choice": a.tts}))
        while True:
            m = json.loads(await asyncio.wait_for(ws.recv(), timeout=a.load_timeout))
            if m.get("type") == "error":
                raise SystemExit("preparation failed: " + m.get("message", ""))
            if m.get("type") == "models_loading_status" and m.get("fully_loaded"):
                break
        await ws.send(json.dumps({"type": "start"}))

        async def recv():
            async for msg in ws:
                t = time.perf_counter() - t0
                if isinstance(msg, bytes):
                    try:
                        events.append({"t": t, "type": "tts_audio", "bytes": len(msg), "seconds": wav_seconds(msg)})
                    except Exception:
                        events.append({"t": t, "type": "tts_audio", "bytes": len(msg), "seconds": 0.0})
                else:
                    d = json.loads(msg)
                    k = d.get("type")
                    if k == "transcription_result":
                        events.append({"t": t, "type": k, "text": d.get("transcribed", ""),
                                       "stt_time": (d.get("metrics") or {}).get("stt_time", 0.0),
                                       "span": d.get("source_span")})
                    elif k == "translation_result":
                        events.append({"t": t, "type": k, "text": d.get("translated", ""),
                                       "mt_time": (d.get("metrics") or {}).get("mt_time", 0.0)})
                    elif k in ("final_metrics", "error"):
                        events.append({"t": t, "type": k, **{x: d[x] for x in ("metrics", "message") if x in d}})

        t0 = time.perf_counter()
        rx = asyncio.create_task(recv())
        n = int(SR * CHUNK_S)
        for i in range(0, len(audio), n):
            await ws.send(audio[i:i + n].tobytes())
            due = t0 + (i + n) / SR                      # absolute 1x schedule: no drift
            await asyncio.sleep(max(0.0, due - time.perf_counter()))
        silence = np.zeros(n, dtype=np.float32).tobytes()
        end_media = time.perf_counter() - t0
        for _ in range(int(a.tail / CHUNK_S)):          # tail silence in real time, closes the last segment
            await ws.send(silence)
            await asyncio.sleep(CHUNK_S)
        await asyncio.sleep(a.settle)
        await ws.send(json.dumps({"type": "stop"}))
        await asyncio.sleep(0.5)
        rx.cancel()

    report = {"audio": a.audio, "seconds": a.seconds, "src": a.src, "tgt": a.tgt, "tts": a.tts,
              "input_end_s": round(end_media, 2), "rates": {}}
    for rate_arg in a.rates:
        rate = rate_arg if rate_arg == "adaptive" else float(rate_arg)
        segs = listener_model(events, rate)
        lags = [s["lag"] for s in segs if "lag" in s]
        report["rates"][rate_arg] = {
            "segments": len(segs), "with_audio": len(lags),
            "first_audio_after_source_end_s": [s.get("first_audio_after_source_end") for s in segs],
            "lag_s": lags,
            "lag_max_s": max(lags) if lags else None,
            "first_word_lag_s": [s.get("first_word_lag") for s in segs],
            "first_word_lag_max_s": max([s["first_word_lag"] for s in segs if "first_word_lag" in s] or [None]),
            "segment_s": [s.get("segment_s") for s in segs],
            "lag_growth_s": (lags[-1] - lags[0]) if len(lags) > 1 else 0.0,
            "last_listen_end_vs_input_end_s": round(max([s.get("listen_end", 0) for s in segs] + [0]) - end_media, 2)}
    report["tts_messages"] = sum(e["type"] == "tts_audio" for e in events)
    report["events"] = events
    if a.out:
        json.dump(report, open(a.out, "w"), indent=1)
    summary = {k: v for k, v in report.items() if k != "events"}
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("audio")
    p.add_argument("--base", default="http://127.0.0.1:8013")
    p.add_argument("--src", default="en")
    p.add_argument("--tgt", default="sk")
    p.add_argument("--tts", default="piper")
    p.add_argument("--seconds", type=float, default=60)
    p.add_argument("--offset", type=float, default=0)
    p.add_argument("--tail", type=float, default=1.5)
    p.add_argument("--settle", type=float, default=15)
    p.add_argument("--load-timeout", type=float, default=300)
    p.add_argument("--rates", nargs="+", default=["1.0", "1.25", "adaptive"])
    p.add_argument("--out")
    asyncio.run(run(p.parse_args()))

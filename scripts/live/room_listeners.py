#!/usr/bin/env python3
"""Connect N listeners (languages given) to a room and report what each received. Used by the Kaggle room test and for manual checks.

    python scripts/live/room_listeners.py --base http://127.0.0.1:8013 --room ID --langs de,de,sk,fr --seconds 90 --out listeners.json
Reports per listener: captions, audio chunks, bytes, seconds of audio, first-caption time, and whether audio chunks per language are identical.
"""
import argparse, asyncio, hashlib, io, json, time, urllib.request, wave

import websockets


def post(base, path, body):
    req = urllib.request.Request(base + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=20) as r: return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e: return e.code, json.loads(e.read().decode() or "{}")


async def listen(base, room, lang, token, seconds, t0, out):
    url = base.replace("http://", "ws://").replace("https://", "wss://") + f"/ws/listen?room={room}&lang={lang}&token={token}"
    rec = {"lang": lang, "captions": 0, "chunks": 0, "bytes": 0, "audio_s": 0.0, "first_caption_s": None, "hashes": [], "errors": [], "sample": None}
    out.append(rec)
    async with websockets.connect(url, max_size=None) as ws:
        end = time.time() + seconds
        while time.time() < end:
            try: m = await asyncio.wait_for(ws.recv(), timeout=max(0.1, end - time.time()))
            except asyncio.TimeoutError: break
            if isinstance(m, bytes):
                rec["chunks"] += 1; rec["bytes"] += len(m); rec["hashes"].append(hashlib.md5(m).hexdigest())
                with wave.open(io.BytesIO(m)) as w: rec["audio_s"] += w.getnframes() / w.getframerate()
            else:
                d = json.loads(m)
                if d.get("type") == "room_caption":
                    rec["captions"] += 1; rec["first_caption_s"] = rec["first_caption_s"] or round(time.time() - t0, 2)
                    rec["sample"] = rec["sample"] or d.get("translated")
                elif d.get("type") == "error": rec["errors"].append(d.get("message"))


async def main(a):
    status, data = post(a.base, "/api/register", {"username": "roomtest", "email": "roomtest@bp.local", "password": "roomtest-pass-1234"})
    status, data = post(a.base, "/api/login", {"email": "roomtest@bp.local", "password": "roomtest-pass-1234"})
    token = data.get("token") or data.get("access_token")
    if not token: raise SystemExit(f"login failed {status} {data}")
    out, t0 = [], time.time()
    await asyncio.gather(*(listen(a.base, a.room, l, token, a.seconds, t0, out) for l in a.langs.split(",")))
    by = {}
    for r in out: by.setdefault(r["lang"], []).append(r)
    same = {l: len({tuple(r["hashes"]) for r in rs}) == 1 for l, rs in by.items()}
    summary = {"listeners": [{k: v for k, v in r.items() if k != "hashes"} for r in out], "identical_audio_within_language": same}
    json.dump(summary, open(a.out, "w"), ensure_ascii=False, indent=1); print(json.dumps(summary, ensure_ascii=False)[:600])

if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--base", default="http://127.0.0.1:8013"); p.add_argument("--room", required=True)
    p.add_argument("--langs", default="de,sk"); p.add_argument("--seconds", type=float, default=90); p.add_argument("--out", default="listeners.json")
    asyncio.run(main(p.parse_args()))

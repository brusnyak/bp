"""Model-free, through the real app: a speaker account creates a room, a listener with only a guest token (no account) joins it by link, and a made-up room id is refused."""
import uuid

API = "/api"


def _speaker_token(client):
    email = f"spk_{uuid.uuid4().hex[:8]}@x.com"
    client.post(f"{API}/register", json={"username": f"spk_{uuid.uuid4().hex[:8]}", "email": email, "password": "passpass1"})
    return client.post(f"{API}/login", json={"email": email, "password": "passpass1"}).json()["token"]


def test_guest_listener_joins_a_room_the_speaker_created(test_client):
    c = test_client
    tok = _speaker_token(c)
    made = c.post(f"{API}/room", headers={"Authorization": f"Bearer {tok}"})
    assert made.status_code == 200
    rid = made.json()["room_id"]
    assert rid in made.json()["listen_path"]
    guest = c.post(f"{API}/guest").json()["token"]                          # what the listener page gets with no login
    with c.websocket_connect(f"/ws/listen?room={rid}&lang=de&token={guest}") as ws:
        joined = ws.receive_json()
        assert joined["type"] == "joined" and joined["room_id"] == rid and joined["lang"] == "de"


def test_unknown_room_is_refused_not_created(test_client):
    c = test_client
    guest = c.post(f"{API}/guest").json()["token"]
    with c.websocket_connect(f"/ws/listen?room=guess{uuid.uuid4().hex[:6]}&lang=de&token={guest}") as ws:
        msg = ws.receive_json()
        assert msg["type"] == "error" and "no such room" in msg["message"]


def test_start_room_gives_a_phone_reachable_link_and_a_qr_code(test_client):
    c = test_client
    h = {"Authorization": f"Bearer {_speaker_token(c)}"}
    made = c.post(f"{API}/room", headers=h).json()
    assert made["listen_url"].endswith(f"/ui/listen/listen.html?room={made['room_id']}") and "lan_ready" in made
    qr = c.get(f"{API}/room/{made['room_id']}/qr", headers=h)
    assert qr.status_code == 200 and qr.headers["content-type"].startswith("image/svg+xml") and "<svg" in qr.text
    assert c.get(f"{API}/room/nosuchroom/qr", headers=h).status_code == 404


def test_localhost_link_is_swapped_for_the_lan_address():
    from backend.room import listen_url
    assert listen_url("http://localhost:8000/", "abc", "192.168.1.20") == "http://192.168.1.20:8000/ui/listen/listen.html?room=abc"
    assert listen_url("https://meet.example.com/", "a b", "192.168.1.20") == "https://meet.example.com/ui/listen/listen.html?room=a%20b"
    assert listen_url("http://localhost:8000/", "abc", None) == "http://localhost:8000/ui/listen/listen.html?room=abc"   # offline: nothing better to offer

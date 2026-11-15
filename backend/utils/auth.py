"""Anonymous session signatures and network rate-limit policy; no identity store."""
import os
import secrets
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

import jwt
from fastapi import HTTPException

JWT_ALGORITHM = "HS256"
JWT_EXPIRE_HOURS = 2
JWT_SECRET_FILE = Path(__file__).resolve().parents[2] / ".jwt_secret"


def _jwt_secret() -> str:
    secret = os.environ.get("JWT_SECRET")
    if secret:
        return secret
    # Publish a complete, mode-0600 file atomically. Concurrent first starts all
    # read the winner, rather than signing with different process-local keys.
    if not JWT_SECRET_FILE.exists():
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", dir=JWT_SECRET_FILE.parent, delete=False) as out:
                temporary = Path(out.name)
                out.write(secrets.token_urlsafe(48))
            try:
                os.link(temporary, JWT_SECRET_FILE)
            except FileExistsError:
                pass
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    secret = JWT_SECRET_FILE.read_text(encoding="utf-8").strip()
    if len(secret) < 32:
        raise RuntimeError("Invalid local signing secret; supply JWT_SECRET or restore .jwt_secret")
    return secret


def create_access_token(guest_id: str, hours: float = JWT_EXPIRE_HOURS) -> str:
    now = int(time.time())
    return jwt.encode({"sub": guest_id, "iat": now, "exp": now + int(hours * 3600)},
                      _jwt_secret(), algorithm=JWT_ALGORITHM)


def decode_access_token(token: str) -> str:
    try:
        payload = jwt.decode(token, _jwt_secret(), algorithms=[JWT_ALGORITHM],
                             options={"require": ["sub", "iat", "exp"]})
        guest_id = payload["sub"]
        if not isinstance(guest_id, str) or uuid.UUID(guest_id).hex != guest_id:
            raise ValueError("Invalid guest id")
        return guest_id
    except (jwt.PyJWTError, ValueError, TypeError):
        raise HTTPException(status_code=401, detail="Invalid or expired session token")


@dataclass(frozen=True)
class GuestSession:
    id: str


def session_from_token(token: str) -> GuestSession:
    return GuestSession(decode_access_token(token))


_LOOPBACK_HOSTS = ("127.0.0.1", "localhost", "::1")


def rate_limit_enabled() -> bool:
    """Explicit flags win; otherwise limit whenever listening beyond loopback."""
    flag = os.environ.get("BP_ENABLE_RATE_LIMIT")
    if flag in ("1", "0"):
        return flag == "1"
    if os.environ.get("BP_DISABLE_RATE_LIMIT") == "1":
        return False
    return os.environ.get("BP_HOST", "localhost") not in _LOOPBACK_HOSTS


def rate_limit(limit: str):
    """Per-client sliding window, shared by requests in one app; no optional dependency."""
    import functools
    import threading
    from collections import deque
    count_text, unit = limit.split("/", 1)
    count = int(count_text)
    window = {"minute": 60, "hour": 3600}[unit]
    lock = threading.Lock()

    def decorate(function):
        @functools.wraps(function)
        async def limited(*args, **kwargs):
            request = kwargs["request"]
            if rate_limit_enabled():
                now = time.monotonic()
                key = (function.__name__, request.client.host if request.client else "unknown")
                with lock:
                    state = getattr(request.app.state, "session_rate_limits", None)
                    if state is None:
                        state = request.app.state.session_rate_limits = {}
                    for old_key in list(state):
                        if not state[old_key] or state[old_key][-1] <= now - window:
                            del state[old_key]
                    if key not in state and len(state) >= 4096:
                        raise HTTPException(429, "Local service capacity reached", headers={"Retry-After": "60"})
                    times = state.setdefault(key, deque())
                    while times and times[0] <= now - window:
                        times.popleft()
                    if len(times) >= count:
                        raise HTTPException(429, "Too many requests", headers={"Retry-After": str(max(1, int(window - (now - times[0]))))})
                    times.append(now)
            return await function(*args, **kwargs)
        return limited
    return decorate

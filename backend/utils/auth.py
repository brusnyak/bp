import logging
import os
import secrets
import time

import jwt
from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerifyMismatchError
from fastapi import HTTPException

# Session tokens: HS256 JWTs signed with JWT_SECRET (set it in .env; scripts/setup.py generates one).
# If unset we mint a random per-process secret: sessions die on restart, but there is no
# published fallback value that lets anyone forge tokens.
JWT_ALGORITHM = "HS256"
JWT_EXPIRE_HOURS = 24
_ephemeral_secret: str | None = None


def _jwt_secret() -> str:
    global _ephemeral_secret
    secret = os.environ.get("JWT_SECRET")
    if secret:
        return secret
    if _ephemeral_secret is None:
        _ephemeral_secret = secrets.token_urlsafe(48)
        logging.warning("Backend: JWT_SECRET unset - using a random per-process secret (logins reset on restart).")
    return _ephemeral_secret


ph = PasswordHasher()


def get_password_hash(password: str) -> str:
    try:
        return ph.hash(password)
    except Exception as e:
        logging.error(f"Backend: Unexpected error during hashing: {type(e).__name__}")
        raise HTTPException(status_code=500, detail="Internal Server Error during password hashing")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    try:
        ph.verify(hashed_password, plain_password)
        return True
    except (VerifyMismatchError, InvalidHashError):
        # InvalidHashError: accounts created via Google login have an empty password hash.
        return False
    except Exception as e:
        logging.error(f"Backend: Unexpected error during password verification: {type(e).__name__}")
        raise HTTPException(status_code=500, detail="Internal Server Error during password verification")


def create_access_token(email: str) -> str:
    """Mint a session JWT for an already-authenticated email."""
    now = int(time.time())
    return jwt.encode(
        {"sub": email, "iat": now, "exp": now + JWT_EXPIRE_HOURS * 3600},
        _jwt_secret(),
        algorithm=JWT_ALGORITHM,
    )


def decode_access_token(token: str) -> str:
    """Return the email in a valid session JWT, else raise 401."""
    try:
        payload = jwt.decode(token, _jwt_secret(), algorithms=[JWT_ALGORITHM])
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid authentication credentials")
    email = payload.get("sub")
    if not email:
        raise HTTPException(status_code=401, detail="Invalid authentication credentials")
    return email

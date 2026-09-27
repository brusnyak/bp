import logging
import os
from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError
from fastapi import HTTPException
from jose import jwt, JWTError

# Session tokens: real HS256 JWTs (no mock strings anywhere). Secret from env;
# the dev fallback is LOUD and must never survive to any shared deployment.
JWT_ALGORITHM = "HS256"
JWT_EXPIRE_HOURS = 24


def _jwt_secret() -> str:
    secret = os.environ.get("JWT_SECRET")
    if not secret:
        logging.warning("Backend: JWT_SECRET unset — using insecure dev fallback. Set JWT_SECRET in .env.")
        return "dev-only-insecure-secret-change-me"
    return secret

# Password hasher instance
ph = PasswordHasher()

def get_password_hash(password: str) -> str:
    logging.debug(f"Backend: Hashing password of length: {len(password)} characters. Password (first 10 chars): {password[:10]}")
    try:
        return ph.hash(password)
    except Exception as e:
        logging.error(f"Backend: Unexpected error during hashing: {e}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error during password hashing: {e}")

def verify_password(plain_password: str, hashed_password: str) -> bool:
    try:
        ph.verify(hashed_password, plain_password)
        return True
    except VerifyMismatchError:
        return False
    except Exception as e:
        logging.error(f"Backend: Unexpected error during password verification: {e}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error during password verification: {e}")


def create_access_token(email: str) -> str:
    """Mint a session JWT for an already-authenticated email."""
    import time
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
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid authentication credentials")
    email = payload.get("sub")
    if not email:
        raise HTTPException(status_code=401, detail="Invalid authentication credentials")
    return email

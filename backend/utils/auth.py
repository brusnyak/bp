import logging
from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError
from fastapi import HTTPException

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

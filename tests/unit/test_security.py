import sys
from datetime import timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from src.core.security import (
    create_access_token,
    decode_access_token,
    get_password_hash,
    verify_password,
)


class TestPasswordHashing:
    def test_hash_is_not_plaintext(self):
        hashed = get_password_hash("myPassword123")
        assert hashed != "myPassword123"

    def test_correct_password_verifies(self):
        hashed = get_password_hash("myPassword123")
        assert verify_password("myPassword123", hashed) is True

    def test_wrong_password_fails(self):
        hashed = get_password_hash("myPassword123")
        assert verify_password("wrongPassword", hashed) is False

    def test_same_password_hashes_differently_each_time(self):
        # bcrypt salts per-call; two hashes of the same password must differ.
        h1 = get_password_hash("samePassword")
        h2 = get_password_hash("samePassword")
        assert h1 != h2
        assert verify_password("samePassword", h1)
        assert verify_password("samePassword", h2)


class TestJWT:
    def test_token_roundtrip(self):
        token = create_access_token(data={"sub": "user123", "persona": "mother"})
        payload = decode_access_token(token)
        assert payload is not None
        assert payload["sub"] == "user123"
        assert payload["persona"] == "mother"

    def test_expired_token_is_rejected(self):
        token = create_access_token(
            data={"sub": "user123"}, expires_delta=timedelta(seconds=-1)
        )
        assert decode_access_token(token) is None

    def test_garbage_token_is_rejected(self):
        assert decode_access_token("not.a.valid.jwt") is None

    def test_tampered_token_is_rejected(self):
        token = create_access_token(data={"sub": "user123"})
        tampered = token[:-2] + ("aa" if not token.endswith("aa") else "bb")
        assert decode_access_token(tampered) is None

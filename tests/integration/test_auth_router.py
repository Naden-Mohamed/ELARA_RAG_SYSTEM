import pytest


class TestRegister:
    @pytest.mark.asyncio
    async def test_register_creates_user_and_returns_token(self, client):
        resp = await client.post(
            "/auth/register",
            json={
                "email": "new@example.com",
                "password": "password123",
                "full_name": "New User",
                "persona": "mother",
            },
        )
        body = resp.json()
        assert body["status"] == "success"
        assert "access_token" in body["data"]
        assert body["data"]["persona"] == "mother"

    @pytest.mark.asyncio
    async def test_register_rejects_duplicate_email(self, client, registered_user):
        _, _, payload = registered_user
        resp = await client.post("/auth/register", json=payload)
        body = resp.json()
        assert body["status"] == "failed"
        assert "already registered" in body["error"].lower()

    @pytest.mark.asyncio
    async def test_register_rejects_invalid_email_format(self, client):
        resp = await client.post(
            "/auth/register",
            json={
                "email": "not-an-email",
                "password": "password123",
                "full_name": "Bad Email",
            },
        )
        assert (
            resp.status_code == 422
        )  # pydantic validation, handled by FastAPI directly

    @pytest.mark.asyncio
    async def test_register_rejects_short_password(self, client):
        resp = await client.post(
            "/auth/register",
            json={
                "email": "shortpw@example.com",
                "password": "123",
                "full_name": "Short PW",
            },
        )
        assert resp.status_code == 422


class TestLogin:
    @pytest.mark.asyncio
    async def test_login_with_correct_credentials_succeeds(
        self, client, registered_user
    ):
        _, _, payload = registered_user
        resp = await client.post(
            "/auth/login",
            json={
                "email": payload["email"],
                "password": payload["password"],
            },
        )
        body = resp.json()
        assert body["status"] == "success"
        assert "access_token" in body["data"]

    @pytest.mark.asyncio
    async def test_login_with_wrong_password_fails(self, client, registered_user):
        _, _, payload = registered_user
        resp = await client.post(
            "/auth/login",
            json={
                "email": payload["email"],
                "password": "wrong-password",
            },
        )
        body = resp.json()
        assert body["status"] == "failed"

    @pytest.mark.asyncio
    async def test_login_with_unknown_email_fails(self, client):
        resp = await client.post(
            "/auth/login",
            json={
                "email": "ghost@example.com",
                "password": "whatever123",
            },
        )
        body = resp.json()
        assert body["status"] == "failed"

    @pytest.mark.asyncio
    async def test_email_is_case_insensitive(self, client, registered_user):
        _, _, payload = registered_user
        resp = await client.post(
            "/auth/login",
            json={
                "email": payload["email"].upper(),
                "password": payload["password"],
            },
        )
        body = resp.json()
        assert body["status"] == "success"


class TestMe:
    @pytest.mark.asyncio
    async def test_me_requires_auth(self, client):
        resp = await client.get("/auth/me")
        assert resp.status_code in (
            401,
            403,
        )  # HTTPBearer raises 403 when header missing entirely

    @pytest.mark.asyncio
    async def test_me_returns_current_user_without_password_hash(
        self, client, registered_user
    ):
        headers, user_id, payload = registered_user
        resp = await client.get("/auth/me", headers=headers)
        body = resp.json()
        assert body["data"]["email"] == payload["email"]
        assert "hashed_password" not in body["data"]

    @pytest.mark.asyncio
    async def test_me_rejects_invalid_token(self, client):
        resp = await client.get(
            "/auth/me", headers={"Authorization": "Bearer garbage.token.here"}
        )
        assert resp.status_code == 401

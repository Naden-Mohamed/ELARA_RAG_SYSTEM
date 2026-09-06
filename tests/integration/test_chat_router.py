import pytest

pytestmark = pytest.mark.integration


class TestSendMessage:
    @pytest.mark.asyncio
    async def test_send_requires_auth(self, client):
        resp = await client.post("/chat/send", json={"query": "hello"})
        assert resp.status_code in (401, 403)

    @pytest.mark.asyncio
    async def test_send_creates_new_chat_when_no_chat_id_given(
        self, client, registered_user
    ):
        headers, _, _ = registered_user
        resp = await client.post(
            "/chat/send", headers=headers, json={"query": "What is BPCR?"}
        )
        body = resp.json()
        assert body["status"] == "success"
        assert body["data"]["chat_id"]
        assert body["data"]["answer"]

    @pytest.mark.asyncio
    async def test_send_persists_user_and_assistant_messages(
        self, client, registered_user
    ):
        headers, _, _ = registered_user
        resp = await client.post(
            "/chat/send", headers=headers, json={"query": "Hello there"}
        )
        chat_id = resp.json()["data"]["chat_id"]

        history_resp = await client.get(f"/chat/{chat_id}/history", headers=headers)
        messages = history_resp.json()["data"]["messages"]
        roles = [m["role"] for m in messages]
        assert roles == ["user", "assistant"]

    @pytest.mark.asyncio
    async def test_reusing_own_chat_id_continues_same_conversation(
        self, client, registered_user
    ):
        headers, _, _ = registered_user
        first = await client.post(
            "/chat/send", headers=headers, json={"query": "First message"}
        )
        chat_id = first.json()["data"]["chat_id"]

        second = await client.post(
            "/chat/send",
            headers=headers,
            json={"query": "Second message", "chat_id": chat_id},
        )
        assert second.json()["data"]["chat_id"] == chat_id


class TestChatIsolationBetweenUsers:
    """
    Regression tests for the IDOR fix: a user must not be able to read or write
    into another user's chat by guessing/reusing their chat_id.
    """

    @pytest.mark.asyncio
    async def test_cannot_send_message_into_another_users_chat(
        self, client, registered_user, second_user
    ):
        headers_a, _, _ = registered_user
        headers_b, _, _ = second_user

        owned_by_a = await client.post(
            "/chat/send", headers=headers_a, json={"query": "This is my private chat"}
        )
        chat_id = owned_by_a.json()["data"]["chat_id"]

        # user B tries to piggyback on user A's chat_id
        hijack_attempt = await client.post(
            "/chat/send",
            headers=headers_b,
            json={"query": "trying to read your chat", "chat_id": chat_id},
        )
        body = hijack_attempt.json()
        assert body["status"] == "failed"
        assert hijack_attempt.json()["status_code"] == 404

    @pytest.mark.asyncio
    async def test_cannot_read_another_users_chat_history(
        self, client, registered_user, second_user
    ):
        headers_a, _, _ = registered_user
        headers_b, _, _ = second_user

        owned_by_a = await client.post(
            "/chat/send", headers=headers_a, json={"query": "secret medical question"}
        )
        chat_id = owned_by_a.json()["data"]["chat_id"]

        resp = await client.get(f"/chat/{chat_id}/history", headers=headers_b)
        body = resp.json()
        assert body["status"] == "failed"
        assert (
            "not found" in body["error"].lower()
            or "unauthorized" in body["error"].lower()
        )

    @pytest.mark.asyncio
    async def test_my_chats_only_returns_own_chats(
        self, client, registered_user, second_user
    ):
        headers_a, _, _ = registered_user
        headers_b, _, _ = second_user

        await client.post(
            "/chat/send", headers=headers_a, json={"query": "A's question"}
        )
        await client.post(
            "/chat/send", headers=headers_b, json={"query": "B's question"}
        )

        resp_a = await client.get("/chat/my-chats", headers=headers_a)
        chats_a = resp_a.json()["data"]
        assert len(chats_a) == 1


class TestPagination:
    @pytest.mark.asyncio
    async def test_history_pagination_has_more_flag(self, client, registered_user):
        headers, _, _ = registered_user
        first = await client.post(
            "/chat/send", headers=headers, json={"query": "msg 1"}
        )
        chat_id = first.json()["data"]["chat_id"]
        for i in range(2, 5):
            await client.post(
                "/chat/send",
                headers=headers,
                json={"query": f"msg {i}", "chat_id": chat_id},
            )

        resp = await client.get(
            f"/chat/{chat_id}/history?page=1&page_size=2", headers=headers
        )
        body = resp.json()["data"]
        assert body["page_size"] == 2
        assert len(body["messages"]) == 2
        assert body["has_more"] is True

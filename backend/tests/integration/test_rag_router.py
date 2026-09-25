import pytest


class TestSearchRequiresAuth:
    @pytest.mark.asyncio
    async def test_search_without_token_is_rejected(self, client):
        resp = await client.post(
            "/rag/search", json={"text": "pregnancy symptoms", "limit": 5}
        )
        assert resp.status_code in (401, 403)


class TestPushRequiresAuth:
    @pytest.mark.asyncio
    async def test_push_without_token_is_rejected(self, client):
        resp = await client.post(
            "/rag/push", json={"document_id": "64b64b64b64b64b64b64b64"}
        )
        assert resp.status_code in (401, 403)


class TestSearchAndRetrieval:
    @pytest.mark.asyncio
    async def test_search_returns_most_similar_chunk_first(
        self, client, registered_user, fake_vectordb, fake_embedding_service
    ):
        headers, _, _ = registered_user

        # Seed the fake vector store directly, bypassing the ingest pipeline,
        # to isolate the /rag/search retrieval logic itself.
        texts = ["pregnancy nutrition advice", "car engine maintenance tips"]
        vectors = [fake_embedding_service._vec(t) for t in texts]
        await fake_vectordb.insert_many(
            "ELARA",
            texts=texts,
            vectors=vectors,
            metadatas=[{"doc_name": "nutrition.pdf"}, {"doc_name": "cars.pdf"}],
        )

        resp = await client.post(
            "/rag/search",
            headers=headers,
            json={"text": "pregnancy nutrition advice", "limit": 2},
        )
        body = resp.json()
        assert body["status_code"] == 200
        top_result = body["data"]["search_results"]["points"][0]
        assert top_result["payload"]["doc_name"] == "nutrition.pdf"


class TestTestPromptEndpoint:
    @pytest.mark.asyncio
    async def test_test_prompt_uses_provided_context_chunks(
        self, client, fake_llm_service
    ):
        resp = await client.post(
            "/rag/test-prompt",
            json={
                "query": "What is BPCR?",
                "persona": "doctor",
                "language": "en",
                "context_chunks": [
                    {
                        "chunk_id": "c1",
                        "doc_name": "who.pdf",
                        "page_number": 1,
                        "section": "Intro",
                        "text": "BPCR stands for Birth Preparedness...",
                    }
                ],
            },
        )
        body = resp.json()
        assert body["status"] == "success"
        assert body["data"]["answer"]
        # confirm our fake LLM actually received the custom chunk, not the module-level mock default
        assert fake_llm_service.calls[-1]["chunks"][0].doc_name == "who.pdf"

    @pytest.mark.asyncio
    async def test_test_prompt_falls_back_to_mock_chunks_when_none_given(
        self, client, fake_llm_service
    ):
        resp = await client.post(
            "/rag/test-prompt",
            json={
                "query": "What is BPCR?",
                "persona": "doctor",
                "language": "en",
            },
        )
        assert resp.json()["status"] == "success"
        assert len(fake_llm_service.calls[-1]["chunks"]) > 0

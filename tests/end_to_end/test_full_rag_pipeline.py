"""
True end-to-end test: exercises the deployed app over real HTTP against
real MongoDB + real Qdrant (started via docker-compose.test.yml — see repo root).

If GROQ_API_KEY is a real key, generation calls hit the real Groq API too —
otherwise this only validates upload -> ingest -> push -> search wiring and
skips the generation assertions.
"""
import os
import time
import io
import uuid

import httpx
import pytest

pytestmark = pytest.mark.e2e

RUN_E2E = os.environ.get("RUN_E2E_TESTS") == "1"
BASE_URL = os.environ.get("E2E_BASE_URL", "http://localhost:8000")

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(not RUN_E2E, reason="Set RUN_E2E_TESTS=1 to run true e2e tests against live services."),
]


def _make_pdf_bytes(text_marker: str) -> bytes:
    pypdf = pytest.importorskip("pypdf")
    from reportlab.pdfgen import canvas  
    buf = io.BytesIO()
    c = canvas.Canvas(buf)
    c.drawString(100, 750, f"ELARA E2E TEST DOCUMENT: {text_marker}")
    c.drawString(100, 730, "Continuous companionship during labour improves maternal outcomes.")
    c.save()
    return buf.getvalue()


@pytest.fixture(scope="module")
def e2e_client():
    with httpx.Client(base_url=BASE_URL, timeout=60.0) as c:
        yield c


@pytest.fixture(scope="module")
def e2e_user(e2e_client):
    email = f"e2e_{uuid.uuid4().hex[:8]}@example.com"
    resp = e2e_client.post("/auth/register", json={
        "email": email, "password": "e2e-test-password", "full_name": "E2E Tester", "persona": "doctor",
    })
    resp.raise_for_status()
    token = resp.json()["data"]["access_token"]
    return {"Authorization": f"Bearer {token}"}


class TestFullIngestionAndRetrievalPipeline:
    def test_health_check(self, e2e_client):
        resp = e2e_client.get("/api/health")
        assert resp.status_code == 200

    def test_upload_ingest_push_search_roundtrip(self, e2e_client, e2e_user):
        marker = uuid.uuid4().hex[:8]
        pdf_bytes = _make_pdf_bytes(marker)

        # 1. Upload
        upload_resp = e2e_client.post(
            "/data/upload", headers=e2e_user,
            files={"file": (f"e2e_{marker}.pdf", pdf_bytes, "application/pdf")},
        )
        upload_resp.raise_for_status()
        document_id = upload_resp.json()["data"]["document_id"]
        assert document_id

        # 2. Ingest (chunk + store chunk records in Mongo)
        ingest_resp = e2e_client.post(
            "/data/ingest", headers=e2e_user, params={"document_id": document_id}
        )
        ingest_resp.raise_for_status()
        assert ingest_resp.json()["data"]["inserted_chunks_count"] > 0

        # 3. Push (embed + store vectors in Qdrant)
        push_resp = e2e_client.post(
            "/rag/push", headers=e2e_user, json={"document_id": document_id}
        )
        push_resp.raise_for_status()
        assert push_resp.json()["data"]["chunk_count"] > 0

        # 4. Search — the marker text should now be retrievable end-to-end
        search_resp = e2e_client.post(
            "/rag/search", headers=e2e_user,
            json={"text": "companionship during labour", "limit": 5},
        )
        search_resp.raise_for_status()
        results = search_resp.json()["data"]["search_results"]["points"]
        assert any(marker in p["payload"].get("text", "") for p in results)

    @pytest.mark.skipif(
        not os.environ.get("GROQ_API_KEY") or os.environ.get("GROQ_API_KEY") == "test-groq-key",
        reason="Real GROQ_API_KEY required for live generation e2e assertions.",
    )
    def test_chat_send_produces_grounded_answer(self, e2e_client, e2e_user):
        resp = e2e_client.post(
            "/chat/send", headers=e2e_user,
            json={"query": "Is a labour companion recommended?", "language": "en"},
        )
        resp.raise_for_status()
        body = resp.json()["data"]
        assert body["answer"]
        assert isinstance(body["citations"], list)

"""
Shared fixtures for unit, integration, and e2e tests.

Design:
- Required env vars are set BEFORE any app import, so `core.config.get_settings()`
  (which validates JWT_SECRET_KEY etc.) never fails at import time.
- `mongo_db` uses mongomock-motor: an in-memory Motor-compatible client, so
  integration tests don't need a real MongoDB instance.
- `fake_vectordb` is a minimal in-memory stand-in for db.qdrant_vectordb.Qdrant,
  implementing the same async method signatures the routers call, backed by
  brute-force cosine similarity. This exercises the real router/service code
  without needing a live Qdrant instance.
- `fake_llm_service` monkeypatches the module-level `llm_service` globals in
  chat_router / rag_router (that's how the app currently wires it) so no real
  Groq API call ever happens in tests.
"""
import os
import sys
import math
import asyncio
from pathlib import Path
from datetime import datetime, timezone

import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport

# --- 1. Make `src/` importable, and set required env vars before any app import ---
SRC_DIR = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

os.environ.setdefault("JWT_SECRET_KEY", "test-secret-key-please-do-not-use-in-prod-32chars")
os.environ.setdefault("MONGODB_URI", "mongodb://localhost:27017")
os.environ.setdefault("MONGODB_DB_NAME", "elara_test")
os.environ.setdefault("QDRANT_URL", "http://localhost:6333")
os.environ.setdefault("GROQ_API_KEY", "test-groq-key")
os.environ.setdefault("EMBEDDING_MODEL_ID", "sentence-transformers/all-MiniLM-L6-v2")
os.environ.setdefault("EMBEDDING_MODEL_SIZE", "384")


def _stub_heavy_optional_dependency(module_name: str, attrs: dict):
    """
    DEMO-ENVIRONMENT-ONLY SHIM.

    docling / docling_core / sentence_transformers are real project dependencies
    (see pyproject.toml) and ARE installed in real CI via `uv sync`, matching
    what the running app actually needs at startup. They're multi-GB installs
    (torch, etc.), so this sandbox doesn't install them. If they're missing,
    inject a minimal fake module so `import main` doesn't explode — this block
    is a no-op the moment the real packages are present, and should be deleted
    once CI always has them installed (i.e. never needed outside quick local
    unit/integration runs where you deliberately skip the heavy ML deps).
    """
    import sys
    import types
    try:
        __import__(module_name)
    except ImportError:
        mod = types.ModuleType(module_name)
        for attr_name, attr_value in attrs.items():
            setattr(mod, attr_name, attr_value)
        sys.modules[module_name] = mod


class _DummyClass:
    def __init__(self, *a, **k):
        pass


_stub_heavy_optional_dependency("docling_core", {})
_stub_heavy_optional_dependency("docling_core.transforms", {})
_stub_heavy_optional_dependency("docling_core.transforms.chunker", {})
_stub_heavy_optional_dependency("docling_core.transforms.chunker.tokenizer", {})
_stub_heavy_optional_dependency(
    "docling_core.transforms.chunker.tokenizer.huggingface", {"HuggingFaceTokenizer": _DummyClass}
)
_stub_heavy_optional_dependency(
    "docling_core.transforms.chunker.hybrid_chunker", {"HybridChunker": _DummyClass}
)
_stub_heavy_optional_dependency("docling", {})
_stub_heavy_optional_dependency("docling.document_converter", {
    "DocumentConverter": _DummyClass, "PdfFormatOption": _DummyClass,
})
_stub_heavy_optional_dependency("docling.datamodel", {})
_stub_heavy_optional_dependency("docling.datamodel.pipeline_options", {"PdfPipelineOptions": _DummyClass})
_stub_heavy_optional_dependency("docling.datamodel.base_models", {"InputFormat": _DummyClass})
_stub_heavy_optional_dependency("transformers", {"AutoTokenizer": _DummyClass})
_stub_heavy_optional_dependency("sentence_transformers", {
    "SentenceTransformer": _DummyClass, "CrossEncoder": _DummyClass,
})


# ---------------------------------------------------------------------------
# Fake Qdrant: in-memory, same call surface as db.qdrant_vectordb.Qdrant
# ---------------------------------------------------------------------------
class _FakePoint:
    def __init__(self, id, payload, score=1.0):
        self.id = id
        self.payload = payload
        self.score = score


class _FakeSearchResult:
    def __init__(self, points):
        self.points = points


class FakeQdrant:
    """In-memory stand-in for db.qdrant_vectordb.Qdrant used in integration tests."""

    def __init__(self):
        self.collections: dict[str, list[dict]] = {}
        self.client = "connected"  # truthy, mirrors real client presence check

    async def connect(self):
        self.client = "connected"

    def disconnect(self):
        self.client = None

    async def is_collection_exists(self, collection_name: str) -> bool:
        return collection_name in self.collections

    async def create_collection(self, collection_name, embedding_size, do_reset=0):
        if collection_name in self.collections and do_reset:
            self.collections[collection_name] = []
        elif collection_name not in self.collections:
            self.collections[collection_name] = []
        return True

    async def delete_collection(self, collection_name):
        self.collections.pop(collection_name, None)
        return True

    async def get_collection_info(self, collection_name):
        if collection_name not in self.collections:
            return None
        return {"points_count": len(self.collections[collection_name])}

    async def insert_many(self, collection_name, texts, vectors, record_ids=None,
                           metadatas=None, batch_size=50):
        if collection_name not in self.collections:
            await self.create_collection(collection_name, len(vectors[0]) if vectors else 0)
        metadatas = metadatas or [{} for _ in texts]
        record_ids = record_ids or [str(i) for i in range(len(texts))]
        for rid, text, vec, meta in zip(record_ids, texts, vectors, metadatas):
            self.collections[collection_name].append({
                "id": rid, "text": text, "vector": vec, "payload": {**meta, "text": text},
            })
        return True

    async def search_by_vector(self, collection_name, vector, top_k=5):
        points = self.collections.get(collection_name, [])
        if not points:
            return _FakeSearchResult([])

        def cosine(a, b):
            dot = sum(x * y for x, y in zip(a, b))
            na = math.sqrt(sum(x * x for x in a)) or 1e-9
            nb = math.sqrt(sum(y * y for y in b)) or 1e-9
            return dot / (na * nb)

        scored = sorted(points, key=lambda p: cosine(p["vector"], vector), reverse=True)
        top = scored[:top_k]
        return _FakeSearchResult([_FakePoint(p["id"], p["payload"], 0.99) for p in top])


# ---------------------------------------------------------------------------
# Fake embedding service: deterministic, no real model download/inference
# ---------------------------------------------------------------------------
class FakeEmbeddingService:
    """Deterministic pseudo-embeddings so retrieval ordering is testable."""

    embedding_size = 8

    def embed_text(self, text, document_type: str = ""):
        if isinstance(text, str):
            text = [text]
        return [self._vec(t) for t in text]

    @staticmethod
    def _vec(text: str):
        # Deterministic hash-based pseudo-embedding, stable across runs.
        v = [0.0] * 8
        for i, ch in enumerate(text.lower()):
            v[i % 8] += ord(ch)
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norm for x in v]


# ---------------------------------------------------------------------------
# Fake LLM service: no real Groq call
# ---------------------------------------------------------------------------
class FakeLLMService:
    """Mimics services.llm_service.LLMService's public async methods."""

    def __init__(self, canned_answer: str = "This is a test answer [Doc: test.pdf, Page: 1, Sec: Intro]."):
        self.canned_answer = canned_answer
        self.calls = []

    async def generate_rag_response(self, query, chunks, persona, language):
        self.calls.append({"query": query, "chunks": chunks, "persona": persona, "language": language})
        return self.canned_answer, 0.01, ["[Doc: test.pdf, Page: 1, Sec: Intro]"]

    async def generate_chat_response(self, query, chunks, persona, language, history,
                                      mother_profile=None, dynamic_memories=None):
        self.calls.append({
            "query": query, "chunks": chunks, "persona": persona, "language": language,
            "history": history,
        })
        return self.canned_answer, 0.01, ["[Doc: test.pdf, Page: 1, Sec: Intro]"]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest_asyncio.fixture
async def mongo_db():
    """In-memory Motor-compatible DB via mongomock-motor. Fresh per test."""
    from mongomock_motor import AsyncMongoMockClient
    client = AsyncMongoMockClient()
    yield client["elara_test"]
    client.close()


@pytest.fixture
def fake_vectordb():
    return FakeQdrant()


@pytest.fixture
def fake_embedding_service():
    return FakeEmbeddingService()


@pytest.fixture
def fake_llm_service():
    return FakeLLMService()


@pytest_asyncio.fixture
async def app(mongo_db, fake_vectordb, fake_embedding_service, fake_llm_service, monkeypatch):
    """
    FastAPI app with test doubles injected directly into app.state,
    bypassing the real lifespan (no live Mongo/Qdrant/model download).
    Also monkeypatches the module-level llm_service globals used by the
    routers today (see routers/chat_router.py and routers/rag_router.py).
    """
    import main as main_module
    from routers import chat_router as chat_router_module
    from routers import rag_router as rag_router_module

    monkeypatch.setattr(chat_router_module, "llm_service", fake_llm_service, raising=False)
    monkeypatch.setattr(rag_router_module, "llm_service", fake_llm_service, raising=False)

    fastapi_app = main_module.app
    fastapi_app.state.db_client = mongo_db
    fastapi_app.state.vectordb = fake_vectordb
    fastapi_app.state.embedding_service = fake_embedding_service
    fastapi_app.state.llm_service = fake_llm_service

    yield fastapi_app


@pytest_asyncio.fixture
async def client(app):
    """Async HTTP client talking to the app in-process (no network, no lifespan)."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest_asyncio.fixture
async def registered_user(client):
    """Registers a user and returns (auth_headers, user_id, raw_payload)."""
    payload = {
        "email": "mother1@example.com",
        "password": "supersecret123",
        "full_name": "Test Mother",
        "persona": "mother",
        "language": "ar",
    }
    resp = await client.post("/auth/register", json=payload)
    body = resp.json()
    token = body["data"]["access_token"]
    user_id = body["data"]["user_id"]
    return {"Authorization": f"Bearer {token}"}, user_id, payload


@pytest_asyncio.fixture
async def second_user(client):
    """A second, distinct user — used for IDOR / cross-tenant tests."""
    payload = {
        "email": "mother2@example.com",
        "password": "anothersecret123",
        "full_name": "Second Mother",
        "persona": "mother",
        "language": "ar",
    }
    resp = await client.post("/auth/register", json=payload)
    body = resp.json()
    token = body["data"]["access_token"]
    user_id = body["data"]["user_id"]
    return {"Authorization": f"Bearer {token}"}, user_id, payload

from contextlib import asynccontextmanager
from logging import getLogger

from core.config import get_settings
from db.qdrant_vectordb import Qdrant
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
from routers import base_router, data_router, rag_router
from routers.auth_router import auth_router
from routers.chat_router import chat_router
from routers.middleware import CorrelationIDMiddleware
from services.embedding import EmbeddingService
from services.llm_service import LLMService
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address

logger = getLogger(__name__)

settings = get_settings()
limiter = Limiter(key_func=get_remote_address, default_limits=["5/30seconds"])


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.limiter = limiter
    app.state.mongo_conn = AsyncIOMotorClient(settings.MONGODB_URI)
    app.state.db_client = app.state.mongo_conn[settings.MONGODB_DB_NAME]
    logger.info("Connected to MongoDB for Auth, Profile & Chat Storage")

    vectordb = Qdrant()
    await vectordb.connect()
    app.state.vectordb = vectordb
    logger.info("Connected to Qdrant")

    app.state.embedding_service = EmbeddingService(
        default_input_max_characters=settings.INPUT_DEFAULT_MAX_CHARACTERS,
    )
    app.state.embedding_service.set_embedding_model(
        model_id=settings.BGE_EMBEDDING_MODEL_ID,
        embedding_size=settings.BGE_EMBEDDING_MODEL_SIZE,
    )

    app.state.llm_service = LLMService()

    yield

    app.state.mongo_conn.close()
    app.state.vectordb.disconnect()
    logger.info("Database connections closed.")


def custom_rate_limiter_handler(request: Request, exc: Exception):
    from fastapi.responses import JSONResponse

    return JSONResponse(
        status_code=429,
        content={"detail": "Rate limit exceeded, please try again later"},
    )


app = FastAPI(lifespan=lifespan)
# Instrumentator.instrument(app).expose(app)

app.add_exception_handler(RateLimitExceeded, custom_rate_limiter_handler)
app.add_middleware(SlowAPIMiddleware)
app.add_middleware(CorrelationIDMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(base_router.base)
app.include_router(data_router.data)
app.include_router(auth_router)
app.include_router(chat_router)
app.include_router(rag_router.rag)

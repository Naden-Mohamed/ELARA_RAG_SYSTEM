from fastapi import FastAPI
from contextlib import asynccontextmanager
from core.config import get_settings
from motor.motor_asyncio import AsyncIOMotorClient
from logging import getLogger

# Routers
from routers import base_router, data_router, rag_router
from routers.auth_router import auth_router
from routers.chat_router import chat_router

# Vector DB & Services
from db.qdrant_vectordb import Qdrant
from services.embedding import EmbeddingService

logger = getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()

    # 1. MongoDB Connection (Auth, User Profile & Chat Storage)
    app.state.mongo_conn = AsyncIOMotorClient(settings.MONGODB_URI)
    app.state.db_client = app.state.mongo_conn[settings.MONGODB_DB_NAME]
    logger.info("Connected to MongoDB for Auth, Profile & Chat Storage")

    # 2. Qdrant Vector DB Connection
    vectordb = Qdrant()
    await vectordb.connect()
    app.state.vectordb = vectordb
    logger.info("Connected to Qdrant")

    # 3. Embedding Service Initialization
    app.state.embedding_service = EmbeddingService(
        default_input_max_characters=settings.INPUT_DEFAULT_MAX_CHARACTERS,
    )
    app.state.embedding_service.set_embedding_model(
        model_id=settings.EMBEDDING_MODEL_ID,
        embedding_size=settings.EMBEDDING_MODEL_SIZE,
    )

    yield

    # Shutdown
    app.state.mongo_conn.close()
    app.state.vectordb.disconnect()
    logger.info("Database connections closed.")


app = FastAPI(lifespan=lifespan)

# Register All Routers
app.include_router(base_router.base)
app.include_router(data_router.data)
app.include_router(auth_router)
app.include_router(chat_router)
app.include_router(rag_router.rag)
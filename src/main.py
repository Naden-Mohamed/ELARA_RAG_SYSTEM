from fastapi import FastAPI
from contextlib import asynccontextmanager
from core.config import get_settings
from motor.motor_asyncio import AsyncIOMotorClient
from logging import getLogger
from routers import base_router, data_router, rag_router
from db.qdrant_vectordb import Qdrant
from services.embedding import EmbeddingService

logger = getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    app.state.mongo_conn = AsyncIOMotorClient(settings.MONGODB_URI)
    app.state.db_client = app.state.mongo_conn[settings.MONGODB_DB_NAME]
    logger.info("Connected to Mongodb")
    print("Connected to Mongodb")

    vectordb = Qdrant()
    await vectordb.connect()
    app.state.vectordb = vectordb

    logger.info("Connected to Qdrant")

    app.state.embedding_service = EmbeddingService(
        default_input_max_characters=settings.INPUT_DEFAULT_MAX_CHARACTERS,
    )
    app.state.embedding_service.set_embedding_model(
        model_id=settings.EMBEDDING_MODEL_ID,
        embedding_size=settings.EMBEDDING_MODEL_SIZE,
    )
                                   

    yield

    app.state.mongo_conn.close()
    app.state.vectordb.disconnect()


app = FastAPI(lifespan=lifespan)

app.include_router(base_router.base)
app.include_router(data_router.data) 
app.include_router(rag_router.rag) 
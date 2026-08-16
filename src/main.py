from fastapi import FastAPI
from contextlib import asynccontextmanager
from core.config import get_settings
from motor.motor_asyncio import AsyncIOMotorClient
from logging import getLogger
from routers import base_router, data_router
from db.qdrant_vectordb import Qdrant
logger = getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    app.state.mongo_conn = AsyncIOMotorClient(settings.MONGODB_URI)
    app.state.db_client = app.state.mongo_conn[settings.MONGODB_DB_NAME]
    logger.info("Connected to Mongodb")
    app.state.vectordb = Qdrant()
    await app.state.vectordb.connect()
                                   

    yield

    app.state.mongo_conn.close()
    app.state.vectordb.disconnect()


app = FastAPI(lifespan=lifespan)

app.include_router(base_router.base)
app.include_router(data_router.data) 
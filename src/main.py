from fastapi import FastAPI
from contextlib import asynccontextmanager
from core.config import get_settings
from motor.motor_asyncio import AsyncIOMotorClient
from logging import getLogger
from routers import rag_router
from routers.auth_router import auth_router

logger = getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    # MongoDB Initialization
    app.state.mongo_conn = AsyncIOMotorClient(settings.MONGODB_URI)
    app.state.db_client = app.state.mongo_conn[settings.MONGODB_DB_NAME]
    logger.info("Connected to MongoDB for Auth & Profile Storage")
    
    yield
    
    app.state.mongo_conn.close()
    logger.info("MongoDB Connection Closed")


app = FastAPI(lifespan=lifespan)

# Routers
app.include_router(auth_router)
app.include_router(rag_router.rag)
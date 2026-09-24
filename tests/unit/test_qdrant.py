import asyncio
import logging

from qdrant_client import AsyncQdrantClient

from src.core.config import get_settings

logger = logging.getLogger(__name__)


async def main():
    settings = get_settings()
    QDRANT_URL = settings.QDRANT_URL
    QDRANT_API_KEY = settings.QDRANT_API_KEY

    client = AsyncQdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        check_compatibility=False,
    )

    try:
        collections = await client.get_collections()
        logger.info(f"existing collections {collections}")

    except Exception as e:
        logger.debug(f"failed connecting to qdrant{e}")

    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())

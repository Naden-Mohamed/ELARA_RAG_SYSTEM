import asyncio
from qdrant_client import AsyncQdrantClient
from ...src.core.config import get_settings

async def main():
    settings = get_settings()
    QDRANT_URL = settings.QDRANT_URL
    QDRANT_API_KEY = settings.QDRANT_API_KEY
    
    print("URL:", QDRANT_URL)
    print("API key exists:", bool(QDRANT_API_KEY))

    client = AsyncQdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        check_compatibility=False,
    )
    print("QDRANT URL:", repr(QDRANT_URL))
    print("API KEY EXISTS:", bool(QDRANT_API_KEY))
    try:
        collections = await client.get_collections()
        print("SUCCESS")
        print(collections)

    except Exception as e:
        print("FAILED")
        print(type(e).__name__)
        print(str(e))

    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())
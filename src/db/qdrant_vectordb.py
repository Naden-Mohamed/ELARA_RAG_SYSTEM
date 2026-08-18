from qdrant_client import AsyncQdrantClient
from qdrant_client.models import VectorParams, PointStruct, Distance
from typing import List, Optional
import logging
from enum import Enum
from core.config import get_settings
import uuid


class DistanceMetric(Enum):
    COSINE = "Cosine"
    EUCLIDEAN = "Euclid" 
    DOT_PRODUCT = "Dot"


class Qdrant:
    def __init__(self, distance_metric: str = "Cosine"):
        self.settings = get_settings()
        raw_url = self.settings.QDRANT_URL.strip() if self.settings.QDRANT_URL else "http://127.0.0.1:6333"
        
        # Ensure scheme exists to prevent client hang
        if not raw_url.startswith("http://") and not raw_url.startswith("https://"):
            raw_url = f"http://{raw_url}"
            
        self.url = raw_url
        self.api_key = self.settings.QDRANT_API_KEY.strip() if self.settings.QDRANT_API_KEY else None
        self.dense_vector_name = self.settings.DENSE_VECTOR_NAME
        self.sparse_vector_name = self.settings.SPARSE_VECTOR_NAME
        self.client = None

        if distance_metric == DistanceMetric.COSINE.value:
            self.distance_metric = Distance.COSINE
        elif distance_metric == DistanceMetric.EUCLIDEAN.value:
            self.distance_metric = Distance.EUCLID
        elif distance_metric == DistanceMetric.DOT_PRODUCT.value:
            self.distance_metric = Distance.DOT
        else:
            self.distance_metric = Distance.COSINE 

        self.logger = logging.getLogger(__name__)

    async def connect(self):
        try:
            is_https = self.url.startswith("https://")
            
            self.client = AsyncQdrantClient(
                url=self.url,
                api_key=self.api_key,
                https=is_https,
                timeout=20.0,
                check_compatibility=False,
            )

            await self.client.get_collections()

            self.logger.info("Successfully connected to Qdrant")
            print("Successfully connected to Qdrant")
            print("QDRANT URL:", repr(self.url))
            print("API KEY EXISTS:", bool(self.api_key))
        except Exception as e:
            self.logger.exception("Cannot connect to Qdrant")
            self.client = None
            raise

    def disconnect(self):
        if self.client:
            self.client = None
            self.logger.info("Qdrant client is disconnected")
        else:
            self.logger.warning("No active connection to disconnect from Qdrant database.")

    async def is_collection_exists(self, collection_name: str) -> bool:
        if not self.client:
            self.logger.error("Qdrant client is not connected. Call connect() first.")
            return False
        return await self.client.collection_exists(collection_name=collection_name)

    async def get_collection_info(self, collection_name: str):
        if self.client and await self.is_collection_exists(collection_name=collection_name):
            return await self.client.get_collection(collection_name=collection_name)
        else:
            self.logger.warning("Not connected to client or this collection doesn't exist.")

    async def get_collections(self):
        if self.client:
            return await self.client.get_collections()
        
    async def delete_collection(self, collection_name: str) -> bool:
        if not self.client:
            self.logger.error("Qdrant client is not connected. Call connect() first.")
            return False
        return await self.client.delete_collection(collection_name=collection_name)

    async def create_collection(self, collection_name: str, embedding_size: int, do_reset: int = 0) -> bool:
        if not self.client:
            self.logger.error("Qdrant client is not connected. Call connect() first.")
            return False
        
        if await self.is_collection_exists(collection_name=collection_name):
            if do_reset:
                await self.delete_collection(collection_name=collection_name)
                self.logger.info(f"do_reset is 1, collection '{collection_name}' deleted")
            else:
                self.logger.info(f"Collection '{collection_name}' already exists, skipping creation.")
                return True 
            
        try:
            await self.client.create_collection(
                collection_name=collection_name,
                vectors_config={
                    self.dense_vector_name: VectorParams(
                        size=embedding_size,
                        distance=self.distance_metric
                    )
                },
            )
            print(f"Collection '{collection_name}' created with size {embedding_size}.")
            self.logger.info(f"Collection '{collection_name}' created with size {embedding_size}.")
            return True
        except Exception as e:
            self.logger.error(f"Failed to create collection '{collection_name}': {e}")
            return False

    async def insert_one(self, collection_name: str, text: str, vector: list, record_id: Optional[str] = None, metadata: Optional[dict] = None) -> bool:
        if not self.client:
            self.logger.error("Qdrant client is not connected. Call connect() first.")
            return False
        
        if not await self.is_collection_exists(collection_name=collection_name):
            self.logger.error(f"Collection '{collection_name}' doesn't exist")
            return False

        if record_id is None:
            record_id = str(uuid.uuid4())

        try:
            await self.client.upsert(
                collection_name=collection_name,
                points=[PointStruct(
                    id=record_id,
                    vector={self.dense_vector_name: vector},
                    payload={
                        "text": text,
                        **(metadata or {})
                    }
                )]
            )
            self.logger.info(f"Inserted one point into collection '{collection_name}' successfully.")
            return True
        except Exception as e:
            self.logger.error(f"Failed to insert point into collection '{collection_name}': {e}")
            return False

    async def insert_many(self, collection_name: str, texts: list[str], vectors: list, record_ids: Optional[list[str]] = None, metadatas: Optional[list] = None, batch_size: int = 50) -> bool:
        if not self.client:
            self.logger.error("Qdrant client is not connected. Call connect() first.")
            return False
        
        if not await self.is_collection_exists(collection_name=collection_name):
            self.logger.error(f"Collection '{collection_name}' doesn't exist")
            created = await self.create_collection(collection_name, self.settings.EMBEDDING_MODEL_SIZE, do_reset=0)
            print(f"Collection '{collection_name}' created")
            if not created:
                return False

        if metadatas is None:
            metadatas = [{}] * len(texts)

        if record_ids is None:
            record_ids = [str(uuid.uuid4()) for _ in range(len(texts))]

        for i in range(0, len(record_ids), batch_size):
            batch_end = i + batch_size
            batch_texts = texts[i:batch_end]
            batch_vectors = vectors[i:batch_end]
            batch_metadatas = metadatas[i:batch_end]
            batch_record_ids = record_ids[i:batch_end]

            batch_points = [
                PointStruct(
                    id=batch_record_ids[x],
                    vector={self.dense_vector_name: batch_vectors[x]},
                    payload={
                        "text": batch_texts[x],
                        **(batch_metadatas[x] or {})
                    }
                )     
                for x in range(len(batch_texts))
            ]

            try:
                await self.client.upsert(
                    collection_name=collection_name,
                    points=batch_points
                )
                self.logger.info(f"Inserted batch of {len(batch_texts)} points into '{collection_name}'.")
            except Exception as e:
                self.logger.error(f"Failed to insert batch into '{collection_name}': {e}")
                return False

        return True

    async def search_by_vector(self, collection_name: str, vector: list, top_k: int = 5):
        if not self.client:
            self.logger.error("Qdrant client is not connected. Call connect() first.")
            return False
        
        if not await self.is_collection_exists(collection_name=collection_name):
            self.logger.error(f"Collection '{collection_name}' doesn't exist")
            return False

        try:
            search_result = await self.client.query_points(
                collection_name=collection_name,
                query=vector,
                using=self.dense_vector_name,
                limit=top_k
            )
            self.logger.info(f"Search in '{collection_name}' completed successfully.")
            return search_result
        except Exception as e:
            self.logger.error(f"Failed to search in '{collection_name}': {e}")
            return None
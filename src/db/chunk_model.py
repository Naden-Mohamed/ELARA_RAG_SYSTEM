from models.data_chunk import DataChunk
from models.enums.DataBaseEnum import DataBaseEnums
from bson.objectid import ObjectId
from pymongo import InsertOne
from motor.motor_asyncio import AsyncIOMotorDatabase

class ChunkModel:

    def __init__(self,  db_client: AsyncIOMotorDatabase):
        self.db_client = db_client
        self.collection = self.db_client[DataBaseEnums.DATA_CHUNKS_COLLECTION.value]

    @classmethod
    async def get_instance(cls,db_client:AsyncIOMotorDatabase):
        instance = cls(db_client)
        await instance.init_collection()
        return instance
    async def init_collection(self):
        all_collections = await self.db_client.list_collection_names()
        if DataBaseEnums.DATA_CHUNKS_COLLECTION.value not in all_collections:
            self.collection = self.db_client[DataBaseEnums.DATA_CHUNKS_COLLECTION.value]
            indexes = DataChunk.get_indexes()
            for index in indexes:
                await self.collection.create_index(
                    index["key"],
                    name=index["name"],
                    unique=index["unique"]
                )

    async def create_chunk(self, chunk: DataChunk):
        result = await self.collection.insert_one(chunk.dict(by_alias=True, exclude_unset=True))
        chunk.id = result.inserted_id
        return chunk

    async def get_chunk(self, chunk_id: str):
        result = await self.collection.find_one({
            "_id": ObjectId(chunk_id)
        })

        if result is None:
            return None
        
        return DataChunk(**result)

    async def insert_many_chunks(self, chunks: list, batch_size: int=100):

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i+batch_size]

            operations = [

                InsertOne(chunk.dict())
                for chunk in batch
            ]

            await self.collection.bulk_write(operations)
        
        return len(chunks)

    async def delete_chunks_by_document_id(self, document_id: str):
        result = await self.collection.delete_many({
            "chunk_document_id": ObjectId(document_id)
        })

        return result.deleted_count
    
    async def get_document_chunks(self, document_id: str, page_no: int=1, page_size: int=50):
        records = await self.collection.find({
                    "chunk_document_id": ObjectId(document_id)
                }).skip(
                    (page_no-1) * page_size
                ).limit(page_size).to_list(length=None)

        return [
            DataChunk(**record)
            for record in records
        ]

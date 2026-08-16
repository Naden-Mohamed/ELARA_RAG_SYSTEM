from models.enums.DataBaseEnum import DataBaseEnums
from models.db_schemes.document import Document
from motor.motor_asyncio import AsyncIOMotorDatabase
from pymongo.errors import ConnectionFailure
import logging
import gridfs

logger = logging.getLogger(__name__)



class DocumentModel:
    def __init__(self, db_client: AsyncIOMotorDatabase) -> None:
        self.db_client = db_client
        self.collection = self.db_client[DataBaseEnums.DOCUMENTS_COLLECTION.value]
    
    
    @classmethod
    async def get_instance(cls,db_client:AsyncIOMotorDatabase):
        instance = cls(db_client)
        await instance.init_collection()
        return instance

    async def init_collection(self):
        all_collections =  await self.db_client.list_collection_names()
        if DataBaseEnums.DOCUMENTS_COLLECTION.value not in all_collections:
            await self.db_client.create_collection(
                DataBaseEnums.DOCUMENTS_COLLECTION.value
            )
            logger.info(f"Collection '{DataBaseEnums.DOCUMENTS_COLLECTION.value}' created.")

    async def upload_document(self, doc: Document):
        result = await self.collection.insert_one(document=doc.dict(by_alias=True))
        return str(result.inserted_id)

    async def delete_document(self, doc_name: str):
         return await self.collection.find_one_and_delete({"doc_name": doc_name})

    async def get_document(self, doc_name: str):
        record = await self.collection.find_one({"doc_name": doc_name})
        if record:
            return Document(**record)





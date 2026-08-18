from models.enums.DataBaseEnum import DataBaseEnums
from models.document import Document
from motor.motor_asyncio import AsyncIOMotorDatabase
from pymongo.errors import ConnectionFailure
import logging
import gridfs
from datetime import timezone
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
    async def get_document_by_id(self, doc_id: str):
        from bson.objectid import ObjectId
        try:
            object_id = ObjectId(doc_id)
        except Exception:
            return None
        record = await self.collection.find_one({"_id": object_id})
        if record:
            return Document(**record)

    async def update_status(
        self,
        doc_id: str,
        status:str,
        chunk_count: int | None = None,
        error_message: str | None = None,
        ):
        from bson.objectid import ObjectId
        from datetime import datetime

        updated_fields: dict = {"status" : status}

        if chunk_count is not None:
            updated_fields["chunk_count"] = chunk_count
        if error_message is not None:
            updated_fields["error_message"] = error_message

        if updated_fields["status"] in (DataBaseEnums.PROCESSED.value, DataBaseEnums.FAILED.value):
            updated_fields["processed_at"] = datetime.now(timezone.utc)
        
    
        return await self.collection.update_one(
            {"_id": ObjectId(doc_id)},
            {"$set": updated_fields}
        )




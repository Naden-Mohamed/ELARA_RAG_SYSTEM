from enum import Enum

class DataBaseEnums(Enum):

    PROJECTS_COLLECTION = "AI_projects"
    DATABASE_NAME = "ai_projects"
    DATA_CHUNKS_COLLECTION = "data_chunks"
    DOCUMENTS_COLLECTION = "ELARA" # Store each collection of related assets in a separated project collection
    PROCESSED = "processed"
    FAILED ="failed"
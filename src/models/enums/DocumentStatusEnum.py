from enum import Enum


class DocumentStatusEnums(Enum):
    PENDING = "pending"        # uploaded, not yet parsed/chunked/embedded
    PROCESSING = "processing"  # ingestion in progress
    PROCESSED = "processed"    # chunks embedded and stored in Qdrant
    FAILED = "failed"          # ingestion attempted and errored


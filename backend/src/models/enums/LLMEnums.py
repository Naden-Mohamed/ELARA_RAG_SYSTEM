from enum import Enum


class LLMEnums(Enum):
    GROQ = "GROQ"
    BGE = "BGE"


class GROQEnums(Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class BGEEnums(Enum):
    SYSTEM = "SYSTEM"
    USER = "USER"
    ASSISTANT = "CHATBOT"

    DOCUMENT = "search_document"
    QUERY = "query_instruction_for_retrieval"


class DocumentTypeEnum(Enum):
    DOCUMENT = "document"
    QUERY = "query"

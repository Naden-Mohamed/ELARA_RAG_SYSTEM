# ELARA RAG System

ELARA is a modular Retrieval-Augmented Generation (RAG) backend built with FastAPI. It handles end-to-end document processing pipelines, including PDF parsing with OCR, hybrid chunking, dense vector embeddings, vector storage in Qdrant, metadata management in MongoDB, and generation workflows via LLMs.

---

## Architecture Overview

* **API Layer**: FastAPI framework with structured routing and standardized response envelopes.
* **Document Parsing**: Docling with OCR and table extraction support.
* **Chunking Engine**: Docling HybridChunker contextualized with Hugging Face tokenizers.
* **Embeddings**: SentenceTransformers with BGE models for dense retrieval.
* **Vector Store**: Qdrant Vector Database for semantic search and point indexing.
* **Metadata Store**: MongoDB (Motor async driver) for tracking document provenance and metadata.
* **LLM Engine**: Groq API integration for response synthesis.

---

## Project Structure

```text
ELARA_RAG_SYSTEM/
├── src/
│   ├── core/           # Application configuration and settings
│   ├── data/           # Uploaded files and local storage
│   ├── db/             # MongoDB and Qdrant database clients
│   ├── models/         # Pydantic schemas, database models, and enums
│   ├── routers/        # FastAPI route definitions (base, data)
│   ├── services/       # Parsing, chunking, and embedding logic
│   └── main.py         # Application entry point and lifespan management
├── tests/              # Test suites
├── .env.example        # Environment variable templates
├── pyproject.toml      # Project dependencies and packaging
├── README.md           # Project documentation
└── uv.lock             # Locked dependency tree
```

## Getting Started
1. **Clone the Repository**
```bash
git clone [https://github.com/your-username/elara-rag-system.git](https://github.com/your-username/elara-rag-system.git)
cd ELARA_RAG_SYSTEM
```
2. **Set Up Environment Variables**
Copy the example environment file and fill in the required API keys and connection strings:
```bash
cp .env.example .env
```

3. **Start Qdrant Vector Database**
Run Qdrant locally using Docker:
```bash
docker run -d -p 6333:6333 -p 6334:6334 -v $(pwd)/qdrant_storage:/qdrant/storage:z qdrant/qdrant
```

4. **Install Dependencies**
Run Qdrant locally using Docker:
```bash
uv sync
```

5. **Run the Application**
Start the FastAPI development server:
```bash
uv run uvicorn src.main:app --reload --port 8000
```
The server will be available at `http://localhost:8000.`
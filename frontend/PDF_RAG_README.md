# PDF-grounded Maternal RAG

This project now uses the two PDF files already present in `attached_assets/` as its evidence library.

## What changed

- The Express API keeps the existing `/api/rag/ask` endpoint used by the React frontend.
- The RAG engine now searches chunks extracted from the PDFs instead of the previous hard-coded sample answers.
- The answer is **abstained** when no source chunk reaches the retrieval threshold, so unrelated questions are not answered from outside knowledge.
- Citations point to the official WHO publication pages.
- The source library and evaluation dashboard use the indexed PDF sources.
- The chat suggestions were changed to topics that actually exist in the indexed PDFs.
- The original FastAPI snippets were not mixed into this TypeScript/Express workspace because this repository already has an Express API wired to the React client. The same RAG behavior was implemented in the existing backend so the frontend works without introducing a second server.

## Indexed PDFs

1. WHO recommendations on antenatal care for a positive pregnancy experience (2016)
2. WHO recommendations for care of the preterm or low birth weight infant (2022)

The generated evidence index is:
`artifacts/api-server/src/rag/evidence.ts`

## Google Drive sources supplied by the user

The seven Drive URLs are recorded in:
`attached_assets/DRIVE_SOURCES.txt`

They could not be downloaded in this environment, so they are **not** treated as indexed evidence. To keep the "answer only from the supplied sources" rule safe, the chatbot will not pretend it has read those files.

After downloading additional PDFs into `attached_assets/`, regenerate `evidence.ts` with your PDF extraction/indexing script before treating them as sources.

## Safety behavior

For an unrelated question, the API returns an abstention such as:
"I couldn't find enough information in the uploaded sources to answer that question."

The backend does not call an external LLM, so it cannot invent medical facts from general model knowledge.

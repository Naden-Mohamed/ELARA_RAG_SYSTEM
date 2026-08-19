import os
from bson import ObjectId
import logging
from fastapi import APIRouter, Request, status, HTTPException
from fastapi.responses import HTMLResponse

from models.api_responce import APIResponce
from models.enums.ResponceStatusEnum import ResponseStatusEnums
from models.enums.DocumentStatusEnum import DocumentStatusEnums
from models.enums.DataBaseEnum import DataBaseEnums
from models.enums.LLMEnums import DocumentTypeEnum
from models.data_chunk import DataChunk

from db.document_model import DocumentModel
from db.chunk_model import ChunkModel

from routers.schemas.data_requests import SearchRequest, PushRequest
from routers.schemas.rag_requests import (
    QueryRequest, 
    DirectPromptTestRequest, 
    MockChunkInput
)
from core.risk_classifier import classify_input_risk, RiskLevel
from core.safety_gate import pre_generation_gate, validate_grounded_response, build_safe_fallback_message

from services.llm_service import LLMService
from core.config import get_settings

logger = logging.getLogger(__name__)
rag = APIRouter(tags=["api/rag"], prefix="/rag")
settings = get_settings()

MOCK_BENCHMARK_CHUNKS = [
    MockChunkInput(
        chunk_id="chunk_01",
        doc_name="WHO_MNH_Care_2025.pdf",
        page_number=4,
        section="Recommendation 1. Birth Preparedness",
        text="A Birth Preparedness and Complication Readiness (BPCR) plan includes: desired birth location, identifying emergency transport, saving funds, and selecting a continuous birth companion.",
        score = 0.90
    ),
    MockChunkInput(
        chunk_id="chunk_02",
        doc_name="WHO_MNH_Care_2025.pdf",
        page_number=8,
        section="Recommendation 8. Labour Companionship",
        text="Continuous companionship during labour improves clinical outcomes and maternal satisfaction. Companions provide emotional and practical support.",
        score = 0.89
    )
]

@rag.post("/push")                    
async def index_push(                   
    request: Request,
    push_request: PushRequest
):
    try:
        db_client = request.app.state.db_client
        vectordb = request.app.state.vectordb
        embedding_service = request.app.state.embedding_service 
        
        chunk_model = await ChunkModel.get_instance(db_client=db_client)
        document_model = await DocumentModel.get_instance(db_client)
        doc = await document_model.get_document_by_id(push_request.document_id)
        
        file_chunks = await chunk_model.get_document_chunks(document_id=push_request.document_id)
        
        if not file_chunks:
            return APIResponce(
                status_code=status.HTTP_404_NOT_FOUND,
                status=ResponseStatusEnums.NO_FILES_FOUNDED_TO_PROCESS.value,
                error="No chunks found. Please run /data/ingest first."
            )

        texts = [
            c.chunk_text if hasattr(c, "chunk_text") else c.chunk_text
            for c in file_chunks
        ]
        
        batch_size = 32
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_emb = embedding_service.embed_text(batch_texts, document_type=DocumentTypeEnum.DOCUMENT.value)
            if batch_emb is not None:
                for vec in batch_emb:
                    all_embeddings.append(vec.tolist() if hasattr(vec, "tolist") else list(vec))

        if not all_embeddings or len(all_embeddings) != len(texts):
            return APIResponce(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                status=ResponseStatusEnums.RAG_ANSWER_ERROR.value,
                error="Embedding generation failed"
            )

        metadatas = [
            {
                **(c.chunk_metadata if hasattr(c, "chunk_metadata") else c.chunk_metadata),
                "document_id": str(push_request.document_id),
                "doc_name": doc.doc_name if doc and hasattr(doc, "doc_name") else (doc.get("doc_name") if isinstance(doc, dict) else "document"),
                "embedding_model": embedding_service.embedding_model_id
            }
            for c in file_chunks
        ]

        inserted = await vectordb.insert_many(
            collection_name=DataBaseEnums.DOCUMENTS_COLLECTION.value,
            texts=texts,
            vectors=all_embeddings,
            metadatas=metadatas,
            batch_size=32
        )

        if not inserted:
            return APIResponce(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                status=ResponseStatusEnums.INSERT_INTO_VECTORDB_ERROR.value,
                error="Failed to store vectors in Qdrant"

            )
        await document_model.update_status(
            doc_id=push_request.document_id,
            status=DocumentStatusEnums.PROCESSED.value,
            chunk_count=len(file_chunks),
        )

        return APIResponce(
            status_code=status.HTTP_200_OK,
            status=ResponseStatusEnums.FILE_PROCESSED_SUCCESSFULLY.value,
            data={"document_id": push_request.document_id, "chunk_count": len(file_chunks)}
        )

    except Exception as e:
        logger.exception("Error during /rag/push")
        return APIResponce(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            status=ResponseStatusEnums.INSERT_INTO_VECTORDB_ERROR.value,
            error=str(e)
        )

@rag.post("/info")                    
async def get_index_info(                   
    request: Request,
    document_id: str
):
    db_client = request.app.state.db_client
    vectordb = request.app.state.vectordb

    document_model = await DocumentModel.get_instance(db_client)
    doc = await document_model.get_document_by_id(document_id)
    info = await vectordb.get_collection_info(DataBaseEnums.DOCUMENTS_COLLECTION.value)

    if not info:
        await document_model.update_status(
            doc_id=document_id,
            status=DocumentStatusEnums.FAILED.value,
            error_message="no info retrieved",
        )
        return APIResponce(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            status=ResponseStatusEnums.INSERT_INTO_VECTORDB_ERROR.value,
            error="Failed to get_collection_info"
        )

    await document_model.update_status(
        doc_id=document_id,
        status=DocumentStatusEnums.PROCESSED.value,
    )

    return APIResponce(
        status_code=status.HTTP_200_OK,
        status=ResponseStatusEnums.FILE_PROCESSED_SUCCESSFULLY.value,
        data={"document_id": document_id, "index_info": info}
    )


@rag.post("/search")                    
async def search_by_vector(                   
    request: Request,
    search_request: SearchRequest
):
    vectordb = request.app.state.vectordb
    embedding_service = request.app.state.embedding_service 

    risk = classify_input_risk(search_request.text)

    if risk["risk_level"] == RiskLevel.UNSAFE:
        answer = build_safe_fallback_message("en")
        citations, latency = [], 0.0
    else:   
        query_embeddings = embedding_service.embed_text(search_request.text, document_type=DocumentTypeEnum.QUERY.value)
        results = await vectordb.search_by_vector(
            DataBaseEnums.DOCUMENTS_COLLECTION.value,
            query_embeddings[0],
            search_request.limit
        )

        if not results:
            return APIResponce(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                status=ResponseStatusEnums.VECTORDB_SEARCH_ERROR.value,
                error="No search results"
            )

    return APIResponce(
        status_code=status.HTTP_200_OK,
        status=ResponseStatusEnums.VECTORDB_SEARCH_SUCCESS.value,
        data={"search_results": results}
    )

@rag.post("/test-prompt", response_model=APIResponce)
async def test_llm_prompt_endpoint(request: Request, payload: DirectPromptTestRequest):
    llm_service = request.app.state.llm_service

    risk = classify_input_risk(payload.query)
    if risk["risk_level"] !=RiskLevel.SAFE and risk["risk_level"] in RiskLevel:
        answer = build_safe_fallback_message(payload.language)
        citations, latency = [], 0.0

        return APIResponce(
            status_code=status.HTTP_403_FORBIDDEN,
            status=risk["risk_level"],
            data={
                "query": payload.query,
                "persona": payload.persona.value,
                "language": payload.language.value,
                "latency_seconds": latency,
                "citations": citations,
            },
            error = risk["reason"]
        )

    try:
        chunks_to_use = payload.context_chunks

        if not chunks_to_use:
            vectordb = request.app.state.vectordb
            embedding_service = request.app.state.embedding_service
            query_embeddings = embedding_service.embed_text(payload.query, document_type=DocumentTypeEnum.QUERY.value)

            search_results = await vectordb.search_by_vector(
                DataBaseEnums.DOCUMENTS_COLLECTION.value,
                query_embeddings[0],
                5
            )

            if search_results and hasattr(search_results, "points"):
                chunks_to_use = []
                for res in search_results.points:
                    p_load = res.payload or {}
                    page_nums = p_load["page_numbers"]
                    page_num = page_nums[0] if isinstance(page_nums, list) and page_nums else 1
                    sections = p_load["section_headings"]
                    section_title = sections[0] if isinstance(sections, list) and sections else "General Recommendations"

                    chunks_to_use.append(
                        MockChunkInput(
                            chunk_id=str(res.id),
                            doc_name=p_load["original_filename"],
                            page_number=page_num,
                            section=section_title,
                            text=p_load["text"],
                            score=res.score or 0.0
                        )
                    )
            else:
                return APIResponce(
                            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            status=ResponseStatusEnums.RAG_ANSWER_ERROR.value,
                            error="No relevant chunks"
                        )


        gate_result = pre_generation_gate(payload.query, chunks_to_use)
        if not gate_result["allow"]:
            return APIResponce(
                status_code=status.HTTP_200_OK,
                status="refused",
                data={
                    "query": payload.query,
                    "persona": payload.persona.value,
                    "language": payload.language.value,
                    "answer": build_safe_fallback_message(payload.language),
                    "latency_seconds": 0.0,
                    "citations": chunks_to_use,
                    "gate_reason": gate_result["reason"],
                    "top_score": gate_result["top_score"]
                }
            )

        answer, latency, citations = await llm_service.generate_rag_response(
            query=payload.query,
            chunks=chunks_to_use,
            persona=payload.persona,
            language=payload.language
        )
        print("citations", citations)

        validation = validate_grounded_response(answer, citations, chunks_to_use)
        # if not validation["valid"]:
        #     answer = build_safe_fallback_message(payload.language)
        #     citations = []

        return APIResponce(
            status_code=status.HTTP_200_OK,
            status="success",
            data={
                "query": payload.query,
                "persona": payload.persona.value,
                "language": payload.language.value,
                "answer": answer,
                "top_similarity_score": gate_result["top_score"],
                "latency_seconds": latency,
                "citations": citations,
                "is_refusal": validation["is_refusal"],
                "validation_reason": validation["reason"],
            }
        )
    except Exception as e:
        return APIResponce(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            status="failed",
            error=str(e)
        )
# why local cooling, such as with ice packs or cold pads could be offered to woman?
#what are not recommended in the postpartum period?
@rag.get("/playground", response_class=HTMLResponse)
async def rag_playground_ui():
    html_content = """
    <!DOCTYPE html>
    <html lang="ar">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>ELARA LLM & Persona Playground</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
        <style>
            body { background: #f8fafc; font-family: system-ui, -apple-system, sans-serif; padding: 30px; }
            .card { border-radius: 12px; border: 1px solid #e2e8f0; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05); }
            .badge-persona { font-size: 13px; padding: 6px 12px; }
            .citation-tag { background: #e0f2fe; color: #0369a1; border-radius: 4px; padding: 2px 6px; font-size: 12px; font-weight: 600; display: inline-block; margin: 2px; }
            .output-box { background: #ffffff; border: 1px solid #cbd5e1; border-radius: 8px; padding: 18px; line-height: 1.8; white-space: pre-wrap; }
            .rtl { direction: rtl; text-align: right; }
            .ltr { direction: ltr; text-align: left; }
        </style>
    </head>
    <body>
        <div class="container-fluid" style="max-width: 1000px;">
            <div class="d-flex justify-content-between align-items-center mb-4">
                <div>
                    <h2 class="fw-bold mb-1">ELARA LLM Persona & Clinical Verification Playground</h2>
                    <p class="text-secondary mb-0">اختبار استجابات الذكاء الاصطناعي مع دعم اللغة العربية/الإنجليزية وفصل نبرة الخطاب</p>
                </div>
            </div>

            <div class="card p-4 mb-4">
                <form id="evalForm">
                    <div class="row g-3">
                        <div class="col-md-6">
                            <label class="form-label fw-bold">الجمهور المستهدف (Persona)</label>
                            <select id="persona" class="form-select">
                                <option value="doctor">طبيب / ممارس صحي (Doctor)</option>
                                <option value="mother">أم / أسرة (Mother / Caregiver)</option>
                                <option value="general">عام (General)</option>
                            </select>
                        </div>
                        <div class="col-md-6">
                            <label class="form-label fw-bold">اللغة (Language)</label>
                            <select id="language" class="form-select">
                                <option value="ar">العربية (Arabic)</option>
                                <option value="en">English</option>
                            </select>
                        </div>
                        <div class="col-12">
                            <label class="form-label fw-bold">السؤال الطبي (Query)</label>
                            <input type="text" id="query" class="form-control" placeholder="اكتبي السؤال هنا..." value="ما هي العناصر الأساسية لخطة الاستعداد للولادة ومضاعفاتها؟" required>
                        </div>
                        <div class="col-12 text-end">
                            <button type="submit" class="btn btn-primary px-4 fw-bold" id="btnSubmit">توليد الرد وفحص الـ Citations</button>
                        </div>
                    </div>
                </form>
            </div>

            <div id="resultCard" class="card p-4 d-none">
                <div class="d-flex justify-content-between align-items-center mb-3">
                    <h5 class="fw-bold mb-0">النتيجة المولدة</h5>
                    <div>
                        <span id="resPersona" class="badge bg-primary badge-persona me-1"></span>
                        <span id="resLang" class="badge bg-secondary badge-persona me-1"></span>
                        <span id="resLatency" class="badge bg-light text-dark badge-persona border"></span>
                    </div>
                </div>
                
                <div id="resAnswer" class="output-box mb-3"></div>

                <div id="citationsContainer">
                    <h6 class="fw-bold text-muted mb-2">الاستشهادات الموثقة (Extracted Citations):</h6>
                    <div id="citationsList"></div>
                </div>
            </div>
        </div>

        <script>
            document.getElementById('evalForm').addEventListener('submit', async (e) => {
                e.preventDefault();
                const btn = document.getElementById('btnSubmit');
                const resultCard = document.getElementById('resultCard');
                const query = document.getElementById('query').value;
                const persona = document.getElementById('persona').value;
                const language = document.getElementById('language').value;

                btn.disabled = true;
                btn.innerText = "جاري التوليد...";
                resultCard.classList.add('d-none');

                try {
                    const res = await fetch('/rag/test-prompt', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            query: query,
                            persona: persona,
                            language: language,
                            context_chunks: []
                        })
                    });
                    const resData = await res.json();
                    
                    if (resData.status_code === 200) {
                        const data = resData.data;
                        const ansEl = document.getElementById('resAnswer');
                        ansEl.innerText = data.answer;
                        ansEl.className = 'output-box mb-3 ' + (data.language === 'ar' ? 'rtl' : 'ltr');

                        document.getElementById('resPersona').innerText = data.persona.toUpperCase();
                        document.getElementById('resLang').innerText = data.language.toUpperCase();
                        document.getElementById('resLatency').innerText = data.latency_seconds + 's';

                        const citDiv = document.getElementById('citationsList');
                        citDiv.innerHTML = '';
                        if (data.citations && data.citations.length > 0) {
                            data.citations.forEach(c => {
                                const span = document.createElement('span');
                                span.className = 'citation-tag';
                                span.innerText = c;
                                citDiv.appendChild(span);
                            });
                        } else {
                            citDiv.innerHTML = '<span class="text-muted small">لا توجد استشهادات أو الإجابة خارج السياق.</span>';
                        }

                        resultCard.classList.remove('d-none');
                    } else {
                        alert("Error: " + (resData.error || "Generation failed"));
                    }
                } catch (err) {
                    alert("Network Error: " + err);
                } finally {
                    btn.disabled = false;
                    btn.innerText = "توليد الرد وفحص الـ Citations";
                }
            });
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)
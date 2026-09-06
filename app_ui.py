from typing import Literal

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

app = FastAPI(title="ELARA Clinical Chat UI - Local Mock", version="1.0")


class CitationDTO(BaseModel):
    document: str = Field(..., description="Exact document title from metadata")
    section: str = Field(..., description="Section name/number")
    page: int = Field(..., description="Page number")


class FinalResponseSchema(BaseModel):
    recommendation: str = Field(
        ..., description="The direct answer or clinical recommendation"
    )
    evidence: str = Field(
        ..., description="Direct excerpt or paraphrase from the retrieved evidence"
    )
    citations: list[CitationDTO] = Field(
        default=[], description="List of structured citations"
    )
    confidence: Literal["high", "medium", "low", "insufficient"] = Field(
        ..., description="Confidence level based on evidence quality"
    )
    confidence_score: float = Field(
        ..., description="Numerical confidence score between 0.0 and 1.0"
    )
    refusal_reason: str | None = Field(
        default=None, description="Explanation if the system refuses to answer"
    )


class ChatRequest(BaseModel):
    query: str
    persona: str = "mother"
    language: str = "ar"
    history: list[dict] = []


MOCK_KNOWLEDGE_BASE = [
    {
        "keywords": ["bpcr", "birth", "plan", "الاستعداد", "ولادة", "عناصر"],
        "doc_name": "WHO recommendations on health promotion interventions for maternal and newborn health.pdf",
        "section": "Recommendation 1",
        "page": 14,
        "text": "A Birth Preparedness and Complication Readiness (BPCR) plan contains the following elements: desired place of birth; preferred birth attendant; location of closest facility; funds for expenses; essential supplies; identified labour and birth companion; support for home; transport; and compatible blood donors.",
        "recommendation_ar": "تشمل خطة الاستعداد للولادة ومضاعفاتها: تحديد مكان الولادة المرغوب، اختيار مقدم الرعاية المفضل، معرفة أقرب منشأة صحية، تجهيز التكاليف المالية، المستلزمات الأساسية، وتحديد مرافق أثناء الولادة.",
        "recommendation_en": "A BPCR plan includes: desired place of birth, preferred birth attendant, closest facility location, funds for expenses, essential supplies, and an identified birth companion.",
    },
    {
        "keywords": ["companion", "labour", "مرافق", "مرافقة", "شخص داعم"],
        "doc_name": "WHO recommendations on health promotion interventions for maternal and newborn health.pdf",
        "section": "Recommendation 8",
        "page": 36,
        "text": "Continuous companionship during labour and birth is recommended for improving women’s satisfaction and outcomes. Women who had continuous support were less likely to have a negative experience.",
        "recommendation_ar": "يوصى بشدة بوجود مرافق أو شخص داعم بشكل مستمر أثناء المخاض والولادة لتحسين تجربة الأم ورضاها وتقليل التجارب السلبية.",
        "recommendation_en": "Continuous companionship during labour and birth is recommended to improve women's satisfaction and clinical outcomes.",
    },
    {
        "keywords": ["iron", "nutrition", "حديد", "تغذية", "فيتامينات", "مكملات"],
        "doc_name": "Maternal Nutrition and Supplementation Guidelines 2024.pdf",
        "section": "Section 3.2 - Micronutrients",
        "page": 22,
        "text": "Daily oral iron and folic acid supplementation is recommended for pregnant women to prevent maternal anaemia, puerperal sepsis, low birth weight, and preterm birth.",
        "recommendation_ar": "يوصى بتناول مكملات الحديد وحمض الفوليك يومياً للحامل للوقاية من فقر الدم ولضمان نمو صحي وتجنب الولادة المبكرة.",
        "recommendation_en": "Daily oral iron and folic acid supplementation is recommended for pregnant women to prevent maternal anaemia and preterm birth.",
    },
]


class ElaraLocalPipeline:
    def __init__(self):
        self.confidence_threshold = 0.60

    def run_pipeline(
        self, query: str, persona: str, language: str, history: list[dict]
    ) -> dict:
        query_lower = query.lower()

        if any(
            term in query_lower
            for term in [
                "ignore",
                "prescribe",
                "dosage",
                "مسكنات",
                "أدوية",
                "باراسيتامول",
                "تجاهل",
            ]
        ):
            return {
                "status": "REFUSED",
                "output": FinalResponseSchema(
                    recommendation="عذراً، لا يمكنني تقديم هذه الاستشارة بناءً على الأدلة الطبية الحالية.",
                    evidence="غير متوفر في نطاق المستندات المعتمدة.",
                    citations=[],
                    confidence="insufficient",
                    confidence_score=0.20,
                    refusal_reason="أنا أسفة جداً، مش هقدر أساعدك في النقطة دي أو أكتب أي أدوية من نفسي عشان سلامتك. دايماً استشيري دكتورك المعالج أو شخص متخصص قبل ما تاخدي أي خطوة!",
                ).model_dump(),
            }

        matched_chunks = []
        best_score = 0.0

        for item in MOCK_KNOWLEDGE_BASE:
            if any(kw in query_lower for kw in item["keywords"]):
                matched_chunks.append(item)
                best_score = 0.89

        if not matched_chunks or best_score < self.confidence_threshold:
            return {
                "status": "REFUSED",
                "output": FinalResponseSchema(
                    recommendation="عذراً، المعلومات غير متوفرة في الأدلة الطبية.",
                    evidence="مطابقة ضعيفة أو غير موجودة.",
                    citations=[],
                    confidence="insufficient",
                    confidence_score=round(best_score, 2),
                    refusal_reason="عذراً يا حبيبتي، المعلومات دي مش متوفرة عندي في الأدلة الطبية المعتمدة حالياً. عشان نكون في أمان تام وسلامتك، يفضل تسألي طبيبك المختص وهو هيفيدك أكتر بكتير!",
                ).model_dump(),
            }

        citations = [
            CitationDTO(
                document=chunk["doc_name"], section=chunk["section"], page=chunk["page"]
            )
            for chunk in matched_chunks
        ]

        evidence_text = " | ".join([c["text"] for c in matched_chunks])
        rec_text = (
            matched_chunks[0]["recommendation_ar"]
            if language == "ar"
            else matched_chunks[0]["recommendation_en"]
        )

        success_output = FinalResponseSchema(
            recommendation=rec_text,
            evidence=evidence_text,
            citations=citations,
            confidence="high",
            confidence_score=best_score,
            refusal_reason=None,
        )

        return {"status": "SUCCESS", "output": success_output.model_dump()}


pipeline = ElaraLocalPipeline()


@app.post("/api/chat")
async def chat_endpoint(payload: ChatRequest):
    try:
        result = pipeline.run_pipeline(
            query=payload.query,
            persona=payload.persona,
            language=payload.language,
            history=payload.history,
        )
        return result
    except Exception as e:
        return {
            "status": "ERROR",
            "output": {
                "recommendation": "حدث خطأ داخلي أثناء معالجة الطلب.",
                "evidence": str(e),
                "citations": [],
                "confidence": "insufficient",
                "confidence_score": 0.0,
                "refusal_reason": f"خطأ تقني: {e!s}",
            },
        }


@app.get("/", response_class=HTMLResponse)
async def chat_ui():
    return """
    <!DOCTYPE html>
    <html lang="ar" dir="rtl">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>ELARA Clinical Grounded Chat</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
        <style>
            body { background: #f1f5f9; font-family: system-ui, -apple-system, sans-serif; height: 100vh; display: flex; flex-direction: column; }
            .chat-container { max-width: 950px; margin: auto; width: 100%; height: 92vh; display: flex; flex-direction: column; background: white; border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.05); overflow: hidden; }
            .chat-header { background: #0f172a; color: white; padding: 15px 20px; display: flex; justify-content: space-between; align-items: center; }
            .chat-box { flex: 1; padding: 20px; overflow-y: auto; background: #f8fafc; display: flex; flex-direction: column; gap: 15px; }
            .message { max-width: 85%; padding: 12px 16px; border-radius: 10px; line-height: 1.6; font-size: 14px; }
            .user-msg { background: #2563eb; color: white; align-self: flex-start; }
            .bot-msg { background: white; border: 1px solid #e2e8f0; color: #1e293b; align-self: flex-end; width: 100%; }
            .chat-input-area { padding: 15px; background: white; border-top: 1px solid #e2e8f0; display: flex; gap: 10px; }
            .citation-box { background: #f0fdf4; border-right: 4px solid #22c55e; padding: 8px 12px; margin-top: 8px; font-size: 12px; color: #166534; border-radius: 4px; }
            .meta-badge { font-size: 11px; padding: 2px 8px; border-radius: 4px; background: #e0f2fe; color: #0369a1; font-weight: bold; margin-left: 5px; }
            .refusal-box { background: #f8fafc; border-right: 4px solid #3b82f6; padding: 10px 14px; margin-top: 8px; font-size: 13px; color: #1e293b; border-radius: 6px; line-height: 1.7; }
        </style>
    </head>
    <body>
        <div class="chat-container">
            <div class="chat-header">
                <h5 class="mb-0">ELARA Clinical Grounded Assistant</h5>
                <div class="d-flex gap-2">
                    <select id="personaSelect" class="form-select form-select-sm">
                        <option value="mother">أم / أسرة (Mother)</option>
                        <option value="doctor">طبيب (Doctor)</option>
                    </select>
                    <select id="langSelect" class="form-select form-select-sm">
                        <option value="ar">العربية</option>
                        <option value="en">English</option>
                    </select>
                </div>
            </div>

            <div id="chatBox" class="chat-box">
                <div class="message bot-msg">
                    <b>ELARA:</b> أهلاً بكِ. أنا مساعدك الطبي المبني على الأدلة والإرشادات المعتمدة. تفضلي بطرح سؤالكِ حول رعاية الأم والطفل.
                </div>
            </div>

            <div class="chat-input-area">
                <input type="text" id="queryInput" class="form-control" placeholder="اكتبي سؤالكِ هنا..." autofocus>
                <button id="sendBtn" class="btn btn-primary px-4" onclick="sendMessage()">إرسال</button>
            </div>
        </div>

        <script>
            let chatHistory = [];

            async function sendMessage() {
                const input = document.getElementById('queryInput');
                const chatBox = document.getElementById('chatBox');
                const persona = document.getElementById('personaSelect').value;
                const language = document.getElementById('langSelect').value;
                const query = input.value.trim();

                if (!query) return;

                chatBox.innerHTML += `<div class="message user-msg"><b>أنتِ:</b> ${query}</div>`;
                input.value = '';
                chatBox.scrollTop = chatBox.scrollHeight;

                const loadingId = 'loading-' + Date.now();
                chatBox.innerHTML += `<div id="${loadingId}" class="message bot-msg text-muted">جاري البحث والتحقق من الأدلة...</div>`;
                chatBox.scrollTop = chatBox.scrollHeight;

                try {
                    const res = await fetch('/api/chat', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ query, persona, language, history: chatHistory })
                    });
                    const data = await res.json();

                    document.getElementById(loadingId).remove();

                    if (data.status === "SUCCESS") {
                        const out = data.output;
                        let html = `<div class="message bot-msg">
                            <b>ELARA:</b><br>${out.recommendation}<br>
                            <div class="mt-2 text-secondary small"><b>الدليل المستند إليه:</b> ${out.evidence}</div>
                            <div class="mt-2">
                                <span class="meta-badge">Confidence: ${out.confidence.toUpperCase()} (${out.confidence_score * 100}%)</span>
                            </div>`;

                        if (out.citations && out.citations.length > 0) {
                            out.citations.forEach((cit, idx) => {
                                html += `<div class="citation-box"><b>المرجع الموثق (${idx + 1}):</b> [${cit.document}, Section ${cit.section}, Page ${cit.page}]</div>`;
                            });
                        }
                        html += `</div>`;
                        chatBox.innerHTML += html;

                        chatHistory.push({ role: "user", content: query });
                        chatHistory.push({ role: "assistant", content: out.recommendation });
                    } else {
                        const out = data.output;
                        let html = `<div class="message bot-msg">
                            <div class="refusal-box"><b>ELARA:</b> ${out.refusal_reason}</div>`;
                        if (out.confidence_score !== undefined) {
                            html += `<div class="mt-2"><span class="meta-badge bg-danger text-white">Confidence Score: ${out.confidence_score * 100}% (Threshold: 60%)</span></div>`;
                        }
                        html += `</div>`;
                        chatBox.innerHTML += html;
                    }
                } catch (err) {
                    document.getElementById(loadingId).remove();
                    chatBox.innerHTML += `<div class="message bot-msg text-danger">حدث خطأ في الاتصال بالشبكة.</div>`;
                }
                chatBox.scrollTop = chatBox.scrollHeight;
            }

            document.getElementById('queryInput').addEventListener('keypress', function (e) {
                if (e.key === 'Enter') {
                    sendMessage();
                }
            });
        </script>
    </body>
    </html>
    """

from dotenv import load_dotenv

load_dotenv()  # Loads .env from root directory before loading settings
import asyncio
import os
import sys

# Ensure project src is in python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(project_root, "src"))
sys.path.insert(0, project_root)

from src.routers.schemas.rag_requests import (
    LanguageEnum,
    MockChunkInput,
    UserPersonaEnum,
)
from src.services.llm_service import LLMService

MOCK_CHUNKS = [
    MockChunkInput(
        chunk_id="chunk_01",
        doc_name="WHO_MNH_Care_2025.pdf",
        page_number=4,
        score=1.0,
        section="Recommendation 1. Birth Preparedness",
        text="A Birth Preparedness and Complication Readiness (BPCR) plan includes: desired birth location, identifying emergency transport, saving funds, and selecting a continuous birth companion.",
    ),
    MockChunkInput(
        chunk_id="chunk_02",
        doc_name="WHO_MNH_Care_2025.pdf",
        page_number=8,
        score=1.0,
        section="Recommendation 8. Labour Companionship",
        text="Continuous companionship during labour improves clinical outcomes and maternal satisfaction. Companions provide emotional and practical support.",
    ),
]

TEST_CASES = [
    {
        "name": "Case 1: Doctor - English (Clinical & Detailed)",
        "query": "What are the clinical requirements and evidence regarding labour companionship?",
        "persona": UserPersonaEnum.DOCTOR,
        "language": LanguageEnum.EN,
        "chunks": MOCK_CHUNKS,
    },
    {
        "name": "Case 2: Mother - English (Simple & Supportive)",
        "query": "Can my husband stay with me while giving birth?",
        "persona": UserPersonaEnum.MOTHER,
        "language": LanguageEnum.EN,
        "chunks": MOCK_CHUNKS,
    },
    {
        "name": "Case 3: Doctor - Arabic (مصطلحات واستشهادات دقيقة)",
        "query": "ما هي العناصر الأساسية لخطة الاستعداد للولادة ومضاعفاتها؟",
        "persona": UserPersonaEnum.DOCTOR,
        "language": LanguageEnum.AR,
        "chunks": MOCK_CHUNKS,
    },
    {
        "name": "Case 4: Mother - Arabic (مبسط بدون تشخيص/جرعات)",
        "query": "إيه اللي أجهزه للولادة وهل مسموح حد يدخل معايا؟",
        "persona": UserPersonaEnum.MOTHER,
        "language": LanguageEnum.AR,
        "chunks": MOCK_CHUNKS,
    },
    {
        "name": "Case 5: Anti-Hallucination Test (Out of Context Query)",
        "query": "What is the recommended dose of amoxicillin for hypertension?",
        "persona": UserPersonaEnum.DOCTOR,
        "language": LanguageEnum.EN,
        "chunks": MOCK_CHUNKS,
    },
]


async def main():
    service = LLMService()
    print("=" * 75)
    print("           TESTING ELARA LLM SERVICE & CITATION GENERATOR")
    print("=" * 75)

    for case in TEST_CASES:
        print(f"\n>> RUNNING: {case['name']}")
        print(f'Query: "{case["query"]}"')
        print("-" * 75)

        answer, latency, citations = await service.generate_rag_response(
            query=case["query"],
            chunks=case["chunks"],
            persona=case["persona"],
            language=case["language"],
        )

        print(f"Generated Output:\n{answer}\n")
        print(f"Latency: {latency}s | Citations Detected: {len(citations)}")
        for cit in citations:
            print(f"   -> {cit}")
        print("=" * 75)


if __name__ == "__main__":
    asyncio.run(main())

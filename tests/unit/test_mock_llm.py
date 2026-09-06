import asyncio
import os

from dotenv import load_dotenv
from groq import AsyncGroq

# Load API keys from root .env
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GENERATION_MODEL_ID = os.getenv("GENERATION_MODEL_ID", "openai/gpt-oss-120b")
# Fallback models if needed: "llama-3.3-70b-versatile" or "llama3-70b-8192"

# Mock Chunks simulating retrieval output with full provenance & citations
MOCK_RETRIEVED_CHUNKS = [
    {
        "chunk_id": "chunk_doc_001_p3",
        "doc_name": "WHO_MNH_Guidelines_2025.pdf",
        "page_number": 3,
        "section": "Recommendation 1. Birth Preparedness",
        "text": "A Birth Preparedness and Complication Readiness (BPCR) plan contains: the desired place of birth, preferred birth attendant, closest emergency facility, funds for birth-related expenses, essential supplies, labour companion, transport arrangements, and identified compatible blood donors.",
    },
    {
        "chunk_id": "chunk_doc_001_p7",
        "doc_name": "WHO_MNH_Guidelines_2025.pdf",
        "page_number": 7,
        "section": "Recommendation 8. Companion of Choice",
        "text": "Continuous companionship during labour and childbirth is strongly recommended. The companion of choice may be a partner, family member, doula, or community member based on the woman's preference.",
    },
]


def build_system_prompt(persona: str) -> str:
    base = (
        "You are ELARA, an expert clinical AI assistant.\n"
        "STRICT GROUNDING & CITATION RULES:\n"
        "1. Answer ONLY using the provided Context Chunks.\n"
        "2. If the answer is not present, state: 'The provided document does not contain this information.'\n"
        "3. Every factual claim MUST include an inline citation formatted as: [Doc: <doc_name>, Page: <page>, Sec: <section>].\n"
    )
    if persona == "doctor":
        persona_rules = (
            "\nAUDIENCE: DOCTOR\n"
            "- Tone: Formal, objective, highly clinical.\n"
            "- Format: Categorized bullet points with exact criteria and source references."
        )
    elif persona == "mother":
        persona_rules = (
            "\nAUDIENCE: MOTHER / CAREGIVER\n"
            "- Tone: Empathetic, warm, and simple.\n"
            "- Format: Clear practical steps followed by a reassuring note to consult her physician."
        )
    else:
        persona_rules = "\nAUDIENCE: GENERAL"

    return f"{base}{persona_rules}"


def construct_user_prompt(query: str, chunks: list[dict]) -> str:
    context_str = ""
    for idx, c in enumerate(chunks, 1):
        context_str += (
            f"--- [SOURCE #{idx}] ---\n"
            f"Doc: {c['doc_name']} | Page: {c['page_number']} | Section: {c['section']}\n"
            f"Content: {c['text']}\n\n"
        )

    return f"Context Chunks:\n{context_str}\nQuestion: {query}\n\nAnswer:"


async def run_standalone_test():
    if not GROQ_API_KEY:
        print("[!] Error: GROQ_API_KEY is not set in environment or .env file.")
        return

    client = AsyncGroq(api_key=GROQ_API_KEY)

    test_queries = [
        {
            "persona": "doctor",
            "query": "What are the essential elements of a BPCR plan and is a birth companion formally recommended?",
        },
        {
            "persona": "mother",
            "query": "What should I prepare for my birth plan, and can my partner stay in the delivery room?",
        },
    ]

    print("=" * 70)
    print("      STANDALONE LLM GENERATION & CITATION TEST (MOCK DATA)")
    print("=" * 70)

    for item in test_queries:
        persona = item["persona"]
        query = item["query"]

        system_prompt = build_system_prompt(persona)
        user_prompt = construct_user_prompt(query, MOCK_RETRIEVED_CHUNKS)

        print(f"\n[+] Testing Query ({persona.upper()}): {query}")
        print("-" * 70)

        try:
            response = await client.chat.completions.create(
                model=GENERATION_MODEL_ID,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.1,
                max_tokens=600,
            )

            answer = response.choices[0].message.content
            print("Generated Response:\n")
            print(answer)
            print("-" * 70)

        except Exception as e:
            print(f"[!] Groq API Call failed: {e}")


if __name__ == "__main__":
    asyncio.run(run_standalone_test())

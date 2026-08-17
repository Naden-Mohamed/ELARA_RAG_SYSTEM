from groq import AsyncGroq
from core.config import get_settings
from models.schemas.rag_requests import UserPersonaEnum
import time

class LLMService:
    def __init__(self):
        settings = get_settings()
        self.client = AsyncGroq(api_key=settings.GROQ_API_KEY)
        self.model_id = settings.GENERATION_MODEL_ID
        self.temperature = settings.GENERATION_DEFAULT_TEMPERATURE

    def _get_system_prompt(self, persona: UserPersonaEnum) -> str:
        base_rules = (
            "You are ELARA, a precise and trusted medical AI assistant.\n"
            "STRICT RULES:\n"
            "1. Base your answer ONLY and STRICTLY on the provided 'Context Chunks'.\n"
            "2. If the context does not contain the answer, explicitly state: 'The provided document does not contain this information.' Do NOT fabricate or extrapolate.\n"
            "3. Do not assume facts outside the provided excerpts.\n"
        )

        if persona == UserPersonaEnum.DOCTOR:
            persona_instructions = (
                "\nTARGET AUDIENCE: DOCTOR / CLINICIAN\n"
                "- Tone: Highly clinical, professional, objective, and concise.\n"
                "- Terminology: Use exact pharmacological, anatomical, and medical terminology.\n"
                "- Format: Categorized bullet points (e.g., Indications, Contraindications, Dosages, Mechanisms, Adverse Effects) with exact numbers and metrics stated in the context."
            )
        elif persona == UserPersonaEnum.MOTHER:
            persona_instructions = (
                "\nTARGET AUDIENCE: MOTHER / CAREGIVER\n"
                "- Tone: Warm, reassuring, empathetic, and clear.\n"
                "- Terminology: Avoid dense medical jargon. Explain concepts in simple, everyday language without losing factual accuracy.\n"
                "- Structure: Direct answer first, followed by clear practical advice and warning signs that require seeing a doctor.\n"
                "- Always include a reassuring note advising consultation with a healthcare provider."
            )
        else:
            persona_instructions = (
                "\nTARGET AUDIENCE: GENERAL HEALTH CONSUMER\n"
                "- Tone: Balanced, informative, clear, and easy to read."
            )

        return f"{base_rules}{persona_instructions}"

    def build_rag_messages(self, query: str, context_chunks: list[str], persona: UserPersonaEnum) -> list[dict]:
        system_content = self._get_system_prompt(persona)
        
        context_block = "\n\n---\n\n".join(
            [f"[Source Chunk #{idx+1}]:\n{chunk}" for idx, chunk in enumerate(context_chunks)]
        )

        user_content = (
            f"Context Excerpts:\n{context_block}\n\n"
            f"Question: {query}\n\n"
            f"Provide your tailored response:"
        )

        return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content}
        ]

    async def generate_response(self, messages: list[dict]) -> tuple[str, float]:
        start_time = time.time()
        response = await self.client.chat.completions.create(
            model=self.model_id,
            messages=messages,
            temperature=self.temperature,
            max_tokens=1000
        )
        latency = round(time.time() - start_time, 3)
        return response.choices[0].message.content, latency
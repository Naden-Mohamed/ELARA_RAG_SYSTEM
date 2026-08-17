import os
import time
import re
from typing import List, Tuple
from dotenv import load_dotenv
from groq import AsyncGroq

from core.config import get_settings
from core.prompts import (
    BASE_RULES_AR,
    BASE_RULES_EN,
    PERSONA_RULES,
    USER_PROMPT_TEMPLATES,
    SOURCE_HEADER_TEMPLATES
)
from routers.schemas.rag_requests import UserPersonaEnum, LanguageEnum, MockChunkInput

load_dotenv()

class LLMService:
    def __init__(self):
        settings = get_settings()
        api_key = settings.GROQ_API_KEY or os.getenv("GROQ_API_KEY")
        self.client = AsyncGroq(api_key=api_key)
        self.model_id = settings.GENERATION_MODEL_ID
        self.temperature = settings.GENERATION_DEFAULT_TEMPERATURE

    def build_system_prompt(self, persona: UserPersonaEnum, language: LanguageEnum) -> str:
        base = BASE_RULES_AR if language == LanguageEnum.AR else BASE_RULES_EN
        persona_rule = PERSONA_RULES.get(
            (persona, language), 
            PERSONA_RULES[(UserPersonaEnum.GENERAL, language)]
        )
        return f"{base}{persona_rule}"

    def build_user_prompt(self, query: str, chunks: List[MockChunkInput], language: LanguageEnum) -> str:
        header_template = SOURCE_HEADER_TEMPLATES[language]
        context_str = ""
        
        for idx, c in enumerate(chunks, 1):
            context_str += header_template.format(
                idx=idx,
                doc_name=c.doc_name,
                page_number=c.page_number,
                section=c.section,
                text=c.text
            )

        prompt_template = USER_PROMPT_TEMPLATES[language]
        return prompt_template.format(context=context_str, query=query)

    async def generate_rag_response(
        self, 
        query: str, 
        chunks: List[MockChunkInput], 
        persona: UserPersonaEnum, 
        language: LanguageEnum
    ) -> Tuple[str, float, List[str]]:
        system_prompt = self.build_system_prompt(persona, language)
        user_prompt = self.build_user_prompt(query, chunks, language)

        start_time = time.time()
        response = await self.client.chat.completions.create(
            model=self.model_id,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=self.temperature,
            max_tokens=800
        )
        latency = round(time.time() - start_time, 3)
        answer = response.choices[0].message.content

        # Extract Citation tags
        citations = re.findall(r"\[(?:Doc|المستند):.*?,.*?,.*?\]", answer)

        return answer, latency, citations
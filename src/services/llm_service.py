import os
import time
import re
from typing import List, Tuple, cast
from dotenv import load_dotenv
from groq import AsyncGroq
from groq.types.chat import ChatCompletionMessageParam
from models.data_chunk import DataChunk
from core.config import get_settings
from core.prompts import (
    BASE_RULES_AR,
    BASE_RULES_EN,
    PERSONA_RULES,
    USER_PROMPT_TEMPLATES,
    SOURCE_HEADER_TEMPLATES
)
from routers.schemas.rag_requests import UserPersonaEnum, LanguageEnum, MockChunkInput
import logging

load_dotenv()
logger = logging.getLogger(__name__)

class LLMService:
    def __init__(self):
        settings = get_settings()
        api_key = settings.GROQ_API_KEY or os.getenv("GROQ_API_KEY")
        self.client = AsyncGroq(api_key= api_key)
        self.model_id = settings.GENERATION_MODEL_ID
        self.temperature = settings.GENERATION_DEFAULT_TEMPERATURE

    def build_system_prompt(self, persona: UserPersonaEnum, language: LanguageEnum) -> str:
        base = BASE_RULES_AR if language == LanguageEnum.AR else BASE_RULES_EN
        persona_rule = PERSONA_RULES.get(
            (persona, language), 
            PERSONA_RULES[(UserPersonaEnum.GENERAL, language)]
        )
        return f"{base}{persona_rule}"

    def build_system_prompt_with_memory(
        self, 
        persona: UserPersonaEnum, 
        language: LanguageEnum, 
        mother_profile: dict | None = None,
        dynamic_memories: list[str] | None = None
    ) -> str:
        base_prompt = self.build_system_prompt(persona, language)
        
        if persona == UserPersonaEnum.MOTHER and mother_profile:
            profile_lines = [
                f"- حالة الحمل: {mother_profile.get('pregnancy_status', 'N/A')}",
                f"- الأسبوع الحالي: {mother_profile.get('current_week', 'غير محدد')}",
                f"- عدد مرات الحمل/الولادة: {mother_profile.get('pregnancies_count', 1)}",
                f"- الأمراض المزمنة: {', '.join(mother_profile.get('chronic_conditions', [])) or 'لا يوجد'}",
                f"- الحساسية الدوائية: {', '.join(mother_profile.get('allergies', [])) or 'لا يوجد'}"
            ]
            if dynamic_memories:
                profile_lines.extend(dynamic_memories)

            context_block = "\n\n[الملف الطبي المسجل للأم]:\n" + "\n".join(profile_lines)
            return base_prompt + context_block

        return base_prompt

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

        print("user_prompt: ", user_prompt)
        start_time = time.time()
        messages: List[ChatCompletionMessageParam] = [
            cast(ChatCompletionMessageParam, {"role": "system", "content": system_prompt}),
            cast(ChatCompletionMessageParam, {"role": "user", "content": user_prompt}),
        ]
        print(system_prompt, user_prompt)
        response = await self.client.chat.completions.create(
            model=self.model_id,
            messages=messages,
            temperature=self.temperature,
            max_tokens=800
        )
        latency = round(time.time() - start_time, 3)
        answer = response.choices[0].message.content if response.choices[0].message.content else ""
        if not answer:
            logger.error("No answer returned from")
            

        citations = re.findall(r"\[(?:Doc|المستند):.*?,.*?,.*?\]", answer)
        return answer, latency, citations

    async def generate_chat_response(
        self,
        query: str,
        chunks: list,
        persona: UserPersonaEnum,
        language: LanguageEnum,
        history: list[dict],
        mother_profile: dict | None = None,
        dynamic_memories: list[str] | None = None
    ) -> tuple[str, float, list[str]]:
        system_prompt = self.build_system_prompt_with_memory(persona, language, mother_profile, dynamic_memories)
        user_prompt = self.build_user_prompt(query, chunks, language)

        messages: List[ChatCompletionMessageParam] = [
            cast(ChatCompletionMessageParam, {"role": "system", "content": system_prompt}),
            cast(ChatCompletionMessageParam, {"role": "user", "content": user_prompt})
        ]
        
        for msg in history:
            messages.append(cast(ChatCompletionMessageParam, {"role": msg["role"], "content": msg["content"]}))
            
        messages.append(cast(ChatCompletionMessageParam, {"role": "user", "content": user_prompt}))

        start_time = time.time()
        response = await self.client.chat.completions.create(
            model=self.model_id,
            messages=messages,
            temperature=self.temperature,
            max_tokens=850
        )
        latency = round(time.time() - start_time, 3)
        answer = response.choices[0].message.content if response.choices[0].message.content else ""

        citations = re.findall(r"\[(?:Doc|المستند):.*?,.*?,.*?\]", answer)

        return answer, latency, citations
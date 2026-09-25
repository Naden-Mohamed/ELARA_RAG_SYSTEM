from datetime import date
from enum import Enum

from pydantic import BaseModel, EmailStr, Field


class PersonaEnum(str, Enum):
    MOTHER = "mother"
    DOCTOR = "doctor"
    GENERAL = "general"


class PregnancyStatusEnum(str, Enum):
    PREGNANT = "pregnant"
    POSTPARTUM = "postpartum"
    PLANNING = "planning"


class MotherProfileSchema(BaseModel):
    pregnancy_status: PregnancyStatusEnum | None = PregnancyStatusEnum.PREGNANT
    current_week: int | None = Field(default=None, ge=1, le=42)
    expected_due_date: date | None = None
    pregnancies_count: int | None = Field(default=1, ge=1)
    previous_c_sections: int | None = 0
    chronic_conditions: list[str] = []
    blood_type: str | None = None
    allergies: list[str] = []
    key_interests: list[str] = []


# Request Payloads
class UserRegisterRequest(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=6)
    full_name: str
    persona: PersonaEnum = PersonaEnum.MOTHER
    language: str = "ar"
    mother_profile: MotherProfileSchema | None = None


class UserLoginRequest(BaseModel):
    email: EmailStr
    password: str


# Responses
class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user_id: str
    full_name: str
    persona: str

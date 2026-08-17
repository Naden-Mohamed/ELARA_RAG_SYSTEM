from pydantic import BaseModel, EmailStr, Field
from typing import Optional, List
from datetime import datetime, date
from enum import Enum


class PersonaEnum(str, Enum):
    MOTHER = "mother"
    DOCTOR = "doctor"
    GENERAL = "general"


class PregnancyStatusEnum(str, Enum):
    PREGNANT = "pregnant"
    POSTPARTUM = "postpartum"
    PLANNING = "planning"


class MotherProfileSchema(BaseModel):
    pregnancy_status: Optional[PregnancyStatusEnum] = PregnancyStatusEnum.PREGNANT
    current_week: Optional[int] = Field(default=None, ge=1, le=42)
    expected_due_date: Optional[date] = None
    pregnancies_count: Optional[int] = Field(default=1, ge=1)
    previous_c_sections: Optional[int] = 0
    chronic_conditions: List[str] = []
    blood_type: Optional[str] = None
    allergies: List[str] = []
    key_interests: List[str] = []


# Request Payloads
class UserRegisterRequest(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=6)
    full_name: str
    persona: PersonaEnum = PersonaEnum.MOTHER
    language: str = "ar"
    mother_profile: Optional[MotherProfileSchema] = None


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
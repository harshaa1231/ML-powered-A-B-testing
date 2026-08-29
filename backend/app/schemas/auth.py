import uuid
from typing import Literal

from pydantic import BaseModel, EmailStr

Persona = Literal["business", "learner"]


class SignupRequest(BaseModel):
    email: EmailStr
    password: str
    full_name: str | None = None
    persona: Persona = "business"


class LoginRequest(BaseModel):
    email: EmailStr
    password: str
    persona: Persona


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class UserResponse(BaseModel):
    id: uuid.UUID
    email: EmailStr
    full_name: str | None = None
    persona: Persona

    model_config = {"from_attributes": True}

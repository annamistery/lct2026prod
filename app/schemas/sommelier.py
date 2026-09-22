from typing import Any

from pydantic import BaseModel, Field

SESSION_ID_PATTERN = r"^[A-Za-z0-9_-]+$"


class SommelierAskRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    session_id: str | None = Field(None, min_length=1, max_length=64, pattern=SESSION_ID_PATTERN)


class SommelierResetRequest(BaseModel):
    session_id: str = Field(..., min_length=1, max_length=64, pattern=SESSION_ID_PATTERN)


class SommelierPick(BaseModel):
    id: str
    name: str
    score: float
    role: str
    unknown: list[str] = Field(default_factory=list)
    reasons: list[dict[str, Any]] = Field(default_factory=list)


class SommelierAskResponse(BaseModel):
    session_id: str
    kind: str
    text: str
    picks: list[SommelierPick] = Field(default_factory=list)
    types: list[dict[str, Any]] = Field(default_factory=list)
    pool: int | None = None
    context: dict[str, Any] | None = None


class SommelierWineInfoResponse(BaseModel):
    kind: str
    id: str | None = None
    text: str

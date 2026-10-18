from datetime import date
from typing import Literal
from pydantic import BaseModel, Field

EventType = Literal["fall", "medication", "behavior", "injury", "general", "unknown"]

class Extraction(BaseModel):
    resident_id: str | None = None
    event_type: EventType = "unknown"
    event_date: date | None = None
    medications_mentioned: list[str] = Field(default_factory=list)
    risk_flags: list[str] = Field(default_factory=list)
    summary: str = Field(max_length=500)
    needs_human_review: bool = True

class DocumentCreate(BaseModel):
    title: str = Field(min_length=1, max_length=200)
    content: str = Field(min_length=1, max_length=20000)

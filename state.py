from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any


class AgentState(BaseModel):
    prompt: str
    classification: List[str] = Field(default_factory=list)
    route_plan: List[str] = Field(default_factory=list)
    route_taken: List[str] = Field(default_factory=list)
    lookup_result: Optional[LookupState] = None


class LookupState(BaseModel):
    prompt: Optional[str] = None
    ticker: Optional[str] = None
    period: Optional[str] = None
    interval: Optional[str] = None
    error: Optional[str] = None

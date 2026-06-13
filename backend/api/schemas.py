"""Pydantic request/response schemas for the PawCare+ API.

The underlying workflow state (``PetCareState``) has 50+ loosely-typed fields and
the LLM/aggregator outputs are free-form JSON, so the response payloads use
permissive ``Dict[str, Any]`` containers. The *request* contract, however, is
strict — that is where validation matters and where bad input is cheap to reject.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

# Mirror the limits enforced by input_validator_agent so the API rejects junk
# before spending an LLM call.
MIN_FIELD_LEN = 3
MAX_FIELD_LEN = 5000


class AssessmentRequest(BaseModel):
    """User-supplied free-text describing their pet."""

    about_pet: str = Field(
        ...,
        description="Species, breed, age, health history, personality, etc.",
        examples=["My dog Max is a 5-year-old Labrador Retriever, neutered male."],
    )
    daily_routine: str = Field(
        ...,
        description="Exercise, diet, living environment, owner experience, etc.",
        examples=["Walks 30 min/day, eats premium kibble twice daily, lives indoors."],
    )
    health_concerns: str = Field(
        ...,
        description="Current symptoms, behavioural issues, medications, vet visits.",
        examples=["Increased thirst and lethargy over the past week."],
    )

    @field_validator("about_pet", "daily_routine", "health_concerns")
    @classmethod
    def _non_trivial(cls, value: str) -> str:
        cleaned = (value or "").strip()
        if len(cleaned) < MIN_FIELD_LEN:
            raise ValueError(f"must be at least {MIN_FIELD_LEN} characters")
        if len(cleaned) > MAX_FIELD_LEN:
            raise ValueError(f"must be at most {MAX_FIELD_LEN} characters")
        return cleaned


class AssessmentResponse(BaseModel):
    """Successful assessment payload returned to the frontend."""

    request_id: str
    path_taken: Optional[str] = None
    health_risk_score: Optional[float] = None
    care_capability_score: Optional[float] = None
    error_occurred: bool = False
    error_messages: List[str] = Field(default_factory=list)
    # Full final state from the workflow (50+ fields). Kept open on purpose.
    result: Dict[str, Any]
    # Structured, display-ready summary from get_pet_health_summary().
    summary: Dict[str, Any]


class HealthResponse(BaseModel):
    status: str = "ok"
    service: str = "pawcare-api"
    version: str = "1.0.0"
    llm_configured: bool = False


class ErrorResponse(BaseModel):
    detail: str

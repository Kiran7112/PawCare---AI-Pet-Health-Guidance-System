"""FastAPI application exposing the PawCare+ assessment workflow.

Run from the ``backend`` directory:

    uvicorn api.server:app --reload --port 8000

The frontend talks to this service; all business logic remains in ``graph.py``.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict

# ---------------------------------------------------------------------------
# Make the flat-layout backend importable (graph.py does `from state import ...`)
# ---------------------------------------------------------------------------
BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from dotenv import load_dotenv  # noqa: E402

# Load .env from repo root (one level above backend/) and from backend/ itself.
load_dotenv(BACKEND_ROOT.parent / ".env")
load_dotenv(BACKEND_ROOT / ".env")

from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.concurrency import run_in_threadpool  # noqa: E402

from api.schemas import (  # noqa: E402
    AssessmentRequest,
    AssessmentResponse,
    HealthResponse,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pawcare.api")

app = FastAPI(
    title="PawCare+ API",
    version="1.0.0",
    description="AI-powered pet health & care guidance, served over HTTP.",
)

# CORS: allow the Vite dev server and configurable production origins.
_default_origins = "http://localhost:5173,http://127.0.0.1:5173"
ALLOWED_ORIGINS = [
    o.strip() for o in os.getenv("PAWCARE_CORS_ORIGINS", _default_origins).split(",") if o.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


def _llm_configured() -> bool:
    """True if at least one OpenAI API key is discoverable."""
    try:
        from utils.openai_client import _get_all_api_keys

        return len(_get_all_api_keys()) > 0
    except Exception:  # pragma: no cover - defensive
        return bool(os.getenv("OPENAI_API_KEY"))


@app.get("/api/health", response_model=HealthResponse, tags=["meta"])
def health() -> HealthResponse:
    return HealthResponse(llm_configured=_llm_configured())


@app.post("/api/assess", response_model=AssessmentResponse, tags=["assessment"])
async def assess(payload: AssessmentRequest) -> AssessmentResponse:
    """Run the full multi-agent assessment for one pet.

    The request is validated by Pydantic before we spend an LLM call. The
    workflow itself is synchronous/CPU+IO bound, so it runs in a threadpool to
    avoid blocking the event loop.
    """
    # Imported lazily so the module (and /api/health) load even if heavy ML/LLM
    # dependencies have an issue at import time.
    from graph import assess_pet_health, get_pet_health_summary

    form_data: Dict[str, str] = {
        "about_pet": payload.about_pet,
        "daily_routine": payload.daily_routine,
        "health_concerns": payload.health_concerns,
    }

    try:
        result: Dict[str, Any] = await run_in_threadpool(assess_pet_health, form_data)
    except Exception as exc:  # pragma: no cover - unexpected hard failure
        logger.exception("Assessment crashed")
        raise HTTPException(status_code=500, detail=f"Assessment failed: {exc}") from exc

    try:
        summary = get_pet_health_summary(result)
    except Exception:  # summary is best-effort; never fail the whole request on it
        logger.exception("Summary extraction failed")
        summary = {}

    return AssessmentResponse(
        request_id=str(result.get("request_id", "unknown")),
        path_taken=result.get("path_taken"),
        health_risk_score=result.get("health_risk_score"),
        care_capability_score=result.get("care_capability_score"),
        error_occurred=bool(result.get("error_occurred", False)),
        error_messages=list(result.get("error_messages", []) or []),
        result=result,
        summary=summary,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api.server:app", host="0.0.0.0", port=8000, reload=True)

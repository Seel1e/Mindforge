"""
app/api.py
───────────
FastAPI REST API for MindForge.
Useful for integrating the model into other apps or mobile frontends.

Run with:
  uvicorn app.api:app --host 0.0.0.0 --port 8000 --reload

Endpoints:
  POST /chat          → Send a message and get a response
  POST /predict-risk  → Get risk prediction from structured data
  GET  /health        → Health check
  GET  /docs          → Swagger UI (auto-generated)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
from loguru import logger

from src.inference.pipeline import MindForgePipeline, UserProfile, ChatResponse

# ── App setup ─────────────────────────────────────────────────
app = FastAPI(
    title="MindForge API",
    description=(
        "Mental Health AI — fine-tuned LLM with RAG and risk prediction.\n\n"
        "⚠️ This API is for educational/research purposes. "
        "Not a substitute for professional mental health care."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Lazy pipeline singleton ───────────────────────────────────
_pipeline: Optional[MindForgePipeline] = None


def get_pipeline() -> MindForgePipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = MindForgePipeline()
    return _pipeline


# ── Request / Response schemas ────────────────────────────────

class ChatRequest(BaseModel):
    message: str = Field(..., description="User's message", max_length=2000)
    profile: Optional[dict] = Field(None, description="Optional user profile for risk prediction")

    class Config:
        json_schema_extra = {
            "example": {
                "message": "I've been feeling really anxious lately and can't sleep.",
                "profile": {
                    "age": 28, "gender": "Female",
                    "stress_level": 8, "sleep_hours": 5.0,
                    "depression_score": 10, "anxiety_score": 14,
                },
            }
        }


class ChatResponseSchema(BaseModel):
    answer: str
    risk_level: Optional[str]
    risk_probabilities: Optional[dict]
    retrieved_context: Optional[str]
    latency_ms: float


class RiskRequest(BaseModel):
    age: int
    gender: str = "Male"
    employment_status: str = "Employed"
    work_environment: str = "On-site"
    mental_health_history: str = "No"
    seeks_treatment: str = "No"
    stress_level: int = 5
    sleep_hours: float = 7.0
    physical_activity_days: int = 3
    depression_score: int = 5
    anxiety_score: int = 5
    social_support_score: int = 5
    productivity_score: int = 5


class RiskResponseSchema(BaseModel):
    risk: str
    probabilities: dict


# ── Endpoints ─────────────────────────────────────────────────

@app.get("/health")
def health():
    """Simple health check — returns 200 if the API is running."""
    return {"status": "ok", "service": "MindForge API"}


@app.post("/chat", response_model=ChatResponseSchema)
def chat(request: ChatRequest):
    """
    Send a message and receive a mental health AI response.
    Optionally include a user profile dict to get a risk prediction.
    """
    try:
        pipeline = get_pipeline()
        profile = UserProfile(**request.profile) if request.profile else None
        result: ChatResponse = pipeline.chat(request.message, profile=profile)
        return ChatResponseSchema(
            answer=result.answer,
            risk_level=result.risk_level,
            risk_probabilities=result.risk_probabilities,
            retrieved_context=result.retrieved_context,
            latency_ms=result.latency_ms,
        )
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=503,
            detail=f"Model not loaded: {e}. Run the training pipeline first.",
        )
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict-risk", response_model=RiskResponseSchema)
def predict_risk(request: RiskRequest):
    """
    Predict mental health risk level from structured demographic + health data.
    Returns Low / Medium / High with probabilities.
    """
    try:
        from src.training.train_risk_model import predict_single
        result = predict_single(request.dict())
        return RiskResponseSchema(risk=result["risk"], probabilities=result["probabilities"])
    except Exception as e:
        logger.error(f"Risk prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

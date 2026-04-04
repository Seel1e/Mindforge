"""
src/inference/pipeline.py
──────────────────────────
The unified inference pipeline that combines:
  1. The fine-tuned LLM (text generation)
  2. RAG (knowledge retrieval)
  3. XGBoost risk predictor (structured data prediction)

Plain English:
  This is the "brain" of the system at runtime.
  When a user sends a message, this pipeline:
    1. Checks if there is structured data (age, stress level etc) → XGBoost predicts risk.
    2. Searches the vector database for relevant psychology knowledge (RAG).
    3. Combines everything into a prompt and sends it to the fine-tuned LLM.
    4. Returns the LLM's answer together with the risk level.

Usage (from Python):
  from src.inference.pipeline import MindForgePipeline
  pipeline = MindForgePipeline()
  result = pipeline.chat("I've been feeling very anxious lately")
  print(result["answer"])
"""

from __future__ import annotations
import time
from dataclasses import dataclass, field
from typing import Optional, Any
from loguru import logger

from src.config import cfg


# ── Data classes for structured I/O ──────────────────────────

@dataclass
class UserProfile:
    """Optional structured info about the user (for risk prediction)."""
    age: Optional[int] = None
    gender: Optional[str] = None
    employment_status: Optional[str] = None
    work_environment: Optional[str] = None
    mental_health_history: Optional[str] = None
    seeks_treatment: Optional[str] = None
    stress_level: Optional[int] = None
    sleep_hours: Optional[float] = None
    physical_activity_days: Optional[int] = None
    depression_score: Optional[int] = None
    anxiety_score: Optional[int] = None
    social_support_score: Optional[int] = None
    productivity_score: Optional[int] = None


@dataclass
class ChatResponse:
    answer: str
    risk_level: Optional[str] = None
    risk_probabilities: Optional[dict] = None
    retrieved_context: Optional[str] = None
    latency_ms: float = 0.0
    model_used: str = "mindforge-lora"


class MindForgePipeline:
    """
    The main inference pipeline.
    Heavy models are loaded lazily (only when first needed) to keep
    startup time fast.
    """

    def __init__(
        self,
        model_dir: Optional[str] = None,
        use_rag: bool = True,
        use_risk_model: bool = True,
        device: str = "auto",
    ):
        self._model_dir = model_dir or str(cfg.MODELS_DIR / "mindforge-lora")
        self._use_rag = use_rag
        self._use_risk_model = use_risk_model
        self._device = device

        self._llm = None
        self._tokenizer = None
        self._retriever = None
        self._risk_model = None

        self._inf_cfg = cfg.inference

    # ─── Lazy loaders ────────────────────────────────────────

    def _load_llm(self):
        if self._llm is not None:
            return
        logger.info(f"Loading LLM from {self._model_dir} …")
        try:
            from unsloth import FastLanguageModel
            self._llm, self._tokenizer = FastLanguageModel.from_pretrained(
                self._model_dir,
                max_seq_length=cfg.model["max_seq_length"],
                dtype=None,
                load_in_4bit=cfg.model["load_in_4bit"],
            )
            FastLanguageModel.for_inference(self._llm)
        except Exception:
            logger.warning("Unsloth not available, using transformers …")
            from transformers import AutoModelForCausalLM, AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(self._model_dir)
            self._llm = AutoModelForCausalLM.from_pretrained(
                self._model_dir, device_map=self._device
            )
        logger.success("LLM loaded.")

    def _load_retriever(self):
        if self._retriever is not None:
            return
        from src.rag.retriever import MindForgeRetriever
        self._retriever = MindForgeRetriever()
        logger.success("RAG retriever loaded.")

    def _load_risk_model(self):
        if self._risk_model is not None:
            return
        import joblib
        model_path = cfg.MODELS_DIR / "risk_predictor.pkl"
        if not model_path.exists():
            logger.warning(f"Risk model not found at {model_path}. Skipping.")
            self._use_risk_model = False
            return
        self._risk_model = joblib.load(model_path)
        logger.success("Risk predictor loaded.")

    # ─── Core methods ────────────────────────────────────────

    def _build_prompt(self, user_message: str, context: str = "") -> str:
        """Construct the full prompt string."""
        system = (
            "You are MindForge, a compassionate and knowledgeable mental health AI assistant. "
            "You provide evidence-based psychological support, education, and guidance. "
            "Always remind users to seek professional help for serious concerns. "
            "Be empathetic, non-judgmental, and supportive."
        )

        if context:
            system += (
                "\n\nUse the following retrieved knowledge to inform your answer:\n"
                f"{context}\n"
                "(Only cite this if it is genuinely relevant.)"
            )

        return (
            f"<s>[INST] <<SYS>>\n{system}\n<</SYS>>\n\n"
            f"{user_message} [/INST]"
        )

    def _generate(self, prompt: str) -> str:
        """Run the LLM and return the generated text."""
        import torch
        self._load_llm()

        device = next(self._llm.parameters()).device
        inputs = self._tokenizer(
            prompt, return_tensors="pt", truncation=True,
            max_length=cfg.model["max_seq_length"] - self._inf_cfg["max_new_tokens"],
        ).to(device)

        with torch.no_grad():
            outputs = self._llm.generate(
                **inputs,
                max_new_tokens=self._inf_cfg["max_new_tokens"],
                temperature=self._inf_cfg["temperature"],
                top_p=self._inf_cfg["top_p"],
                repetition_penalty=self._inf_cfg["repetition_penalty"],
                do_sample=True,
                pad_token_id=self._tokenizer.eos_token_id,
            )

        generated = outputs[0][inputs["input_ids"].shape[1]:]
        return self._tokenizer.decode(generated, skip_special_tokens=True).strip()

    def _predict_risk(self, profile: UserProfile) -> Optional[dict]:
        """Run the XGBoost risk predictor on a UserProfile."""
        self._load_risk_model()
        if not self._risk_model:
            return None

        import pandas as pd, json
        import numpy as np

        sample = {
            "age": profile.age or 30,
            "stress_level": profile.stress_level or 5,
            "sleep_hours": profile.sleep_hours or 7.0,
            "physical_activity_days": profile.physical_activity_days or 3,
            "depression_score": profile.depression_score or 5,
            "anxiety_score": profile.anxiety_score or 5,
            "social_support_score": profile.social_support_score or 5,
            "productivity_score": profile.productivity_score or 5,
        }

        # Encode categoricals using saved mapping
        mapping_path = cfg.DATA_PROCESSED / "label_mapping.json"
        if mapping_path.exists():
            with open(mapping_path) as f:
                mapping = json.load(f)

            cat_fields = {
                "gender": profile.gender or "Male",
                "employment_status": profile.employment_status or "Employed",
                "work_environment": profile.work_environment or "On-site",
                "mental_health_history": profile.mental_health_history or "No",
                "seeks_treatment": profile.seeks_treatment or "No",
            }
            for col, val in cat_fields.items():
                enc = mapping.get(col, {})
                sample[col] = enc.get(val.title(), 0)

        # Feature engineering (mirror prepare_structured.py)
        sample["wellness_score"] = (
            (sample["sleep_hours"] / 9.0)
            + (sample["physical_activity_days"] / 7.0)
            + (sample["social_support_score"] / 10.0)
        ) / 3.0
        sample["distress_index"] = (
            (sample["depression_score"] / 30.0)
            + (sample["anxiety_score"] / 21.0)
            + (sample["stress_level"] / 10.0)
        ) / 3.0

        # Age group
        age = sample["age"]
        bins = [(0, 18, "<18"), (18, 25, "18-25"), (25, 35, "26-35"),
                (35, 50, "36-50"), (50, 65, "51-65"), (65, 200, "65+")]
        age_label = next((lbl for lo, hi, lbl in bins if lo <= age < hi), "26-35")
        if mapping_path.exists():
            age_group_enc = mapping.get("age_group", {})
            sample["age_group"] = age_group_enc.get(age_label, 0)

        sample["has_mh_history"] = 1 if (profile.mental_health_history or "No").lower() == "yes" else 0
        sample["seeks_treatment_flag"] = 1 if (profile.seeks_treatment or "No").lower() == "yes" else 0

        df = pd.DataFrame([sample])
        pred = self._risk_model.predict(df)[0]
        proba = self._risk_model.predict_proba(df)[0]

        label_map = {0: "Low", 1: "Medium", 2: "High"}
        return {
            "risk": label_map[pred],
            "probabilities": {label_map[i]: float(p) for i, p in enumerate(proba)},
        }

    def chat(
        self,
        message: str,
        profile: Optional[UserProfile] = None,
        chat_history: Optional[list] = None,
    ) -> ChatResponse:
        """
        Main entry point.

        Args:
            message:      The user's message.
            profile:      Optional structured user profile for risk prediction.
            chat_history: List of previous (user, assistant) turn dicts.

        Returns:
            ChatResponse with the answer and optional risk info.
        """
        t0 = time.time()

        # ── 1. Risk prediction ────────────────────────────────
        risk_result = None
        if self._use_risk_model and profile and profile.age is not None:
            try:
                risk_result = self._predict_risk(profile)
                logger.debug(f"Risk: {risk_result}")
            except Exception as e:
                logger.warning(f"Risk prediction failed: {e}")

        # ── 2. RAG retrieval ──────────────────────────────────
        context = ""
        if self._use_rag:
            try:
                self._load_retriever()
                context = self._retriever.get_context(message)
            except Exception as e:
                logger.warning(f"RAG retrieval failed: {e}")

        # ── 3. Add risk info to user message if available ─────
        enriched_message = message
        if risk_result:
            enriched_message += (
                f"\n\n[System note for assistant: User's predicted risk level is "
                f"{risk_result['risk']} based on their profile. "
                f"Tailor your response accordingly.]"
            )

        # ── 4. LLM generation ─────────────────────────────────
        prompt = self._build_prompt(enriched_message, context)
        answer = self._generate(prompt)

        latency = (time.time() - t0) * 1000

        return ChatResponse(
            answer=answer,
            risk_level=risk_result["risk"] if risk_result else None,
            risk_probabilities=risk_result["probabilities"] if risk_result else None,
            retrieved_context=context[:500] if context else None,
            latency_ms=round(latency, 1),
        )

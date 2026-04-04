"""
src/config.py
─────────────
Loads config.yaml and .env into a single Config object
so every other module can do: from src.config import cfg
"""

import os
from pathlib import Path
from dotenv import load_dotenv
import yaml
from loguru import logger

# ── Load .env first ───────────────────────────────────────────
load_dotenv()

ROOT = Path(__file__).resolve().parent.parent   # project root


def _load_yaml(path: Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


class _Config:
    """Dot-access wrapper around the YAML config + env vars."""

    def __init__(self):
        raw = _load_yaml(ROOT / "config.yaml")

        # ── Attach every top-level key as an attribute ────────
        for key, value in raw.items():
            setattr(self, key, value)

        # ── Resolve paths relative to project root ────────────
        self.ROOT = ROOT
        self.DATA_RAW = ROOT / self.data["raw_dir"]
        self.DATA_PROCESSED = ROOT / self.data["processed_dir"]
        self.MODELS_DIR = ROOT / "models"

        # ── Dataset absolute paths ────────────────────────────
        ds = self.data["datasets"]
        self.DS_PSYCHOLOGY_JSON = self.DATA_RAW / ds["psychology_json"]
        self.DS_CLEAN_STATEMENTS = self.DATA_RAW / ds["clean_statements"]
        self.DS_COMBINED_METRICS = self.DATA_RAW / ds["combined_metrics"]
        self.DS_THERAPY_QA = self.DATA_RAW / ds["therapy_qa"]
        self.DS_RISK_TRAIN = self.DATA_RAW / ds["risk_train"]

        # ── Env vars ──────────────────────────────────────────
        self.HF_TOKEN = os.getenv("HF_TOKEN", "")
        self.WANDB_API_KEY = os.getenv("WANDB_API_KEY", "")
        self.WANDB_PROJECT = os.getenv("WANDB_PROJECT", "mindforge")

        # Ensure processed dir exists
        self.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
        self.MODELS_DIR.mkdir(parents=True, exist_ok=True)

        logger.info(f"Config loaded — project root: {ROOT}")

    def __repr__(self):
        return f"<MindForge Config | root={self.ROOT}>"


# Singleton — import this everywhere
cfg = _Config()

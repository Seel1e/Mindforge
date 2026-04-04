"""
src/preprocessing/prepare_structured.py
─────────────────────────────────────────
Cleans and engineers features from Combined Data.csv and train.csv
for the XGBoost mental health risk predictor.

Plain English:
  "Combined Data.csv" and "train.csv" have rows like:
      age=25, gender=Female, stress_level=8 → mental_health_risk=High

  This script:
  1. Loads both files and merges them (more data = better model).
  2. Fixes missing values.
  3. Converts text columns (e.g. "Male") into numbers the model understands.
  4. Creates new features (e.g. a "wellness_score" combining sleep + activity).
  5. Saves the cleaned file to data/processed/structured_clean.csv.

Usage:
  python -m src.preprocessing.prepare_structured
"""

import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger

from src.config import cfg


# ── Column name constants ─────────────────────────────────────
TARGET = "mental_health_risk"

CATEGORICAL_COLS = [
    "gender",
    "employment_status",
    "work_environment",
    "mental_health_history",
    "seeks_treatment",
]

NUMERIC_COLS = [
    "age",
    "stress_level",
    "sleep_hours",
    "physical_activity_days",
    "depression_score",
    "anxiety_score",
    "social_support_score",
    "productivity_score",
]


def load_and_merge() -> pd.DataFrame:
    """Load both CSVs, deduplicate, and merge into one DataFrame."""
    logger.info("Loading Combined Data.csv …")
    combined = pd.read_csv(cfg.DS_COMBINED_METRICS)

    logger.info("Loading train.csv …")
    train = pd.read_csv(cfg.DS_RISK_TRAIN)

    # Normalise column names (lowercase + underscores)
    def _norm_cols(df: pd.DataFrame) -> pd.DataFrame:
        df.columns = (
            df.columns
            .str.strip()
            .str.lower()
            .str.replace(" ", "_")
            .str.replace("-", "_")
        )
        return df

    combined = _norm_cols(combined)
    train = _norm_cols(train)

    # Rename alternative target column names to the standard one
    for frame in [combined, train]:
        for alt in ["status", "risk_level", "label", "target"]:
            if alt in frame.columns and TARGET not in frame.columns:
                frame.rename(columns={alt: TARGET}, inplace=True)
                logger.info(f"Renamed '{alt}' → '{TARGET}'")

    # Stack them
    df = pd.concat([combined, train], ignore_index=True)
    before = len(df)
    df.drop_duplicates(inplace=True)
    logger.info(f"Merged {before:,} rows → {len(df):,} after dedup")

    return df


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values and fix dtypes."""

    # ── Target column ─────────────────────────────────────────
    if TARGET not in df.columns:
        raise ValueError(f"Target column '{TARGET}' not found. Columns: {df.columns.tolist()}")

    # Drop rows where target is missing (can't train without a label)
    df.dropna(subset=[TARGET], inplace=True)
    df[TARGET] = df[TARGET].str.strip().str.title()   # 'high' → 'High'

    # ── Numeric — fill with median ────────────────────────────
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col].fillna(df[col].median(), inplace=True)

    # ── Categorical — fill with mode ──────────────────────────
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.title()
            df[col].replace("Nan", np.nan, inplace=True)
            df[col].fillna(df[col].mode()[0], inplace=True)

    logger.info(f"After cleaning: {df.shape[0]:,} rows × {df.shape[1]} cols")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create new, informative features from existing ones.

    These engineered features often give the model a boost because
    they encode domain knowledge (e.g. "low sleep + high stress = bad").
    """

    # ── Wellness score: combines sleep, activity, social support ─
    if all(c in df.columns for c in ["sleep_hours", "physical_activity_days", "social_support_score"]):
        df["wellness_score"] = (
            (df["sleep_hours"] / 9.0)                   # normalise to ~0-1
            + (df["physical_activity_days"] / 7.0)
            + (df["social_support_score"] / 10.0)
        ) / 3.0

    # ── Distress index: combines depression + anxiety + stress ─
    if all(c in df.columns for c in ["depression_score", "anxiety_score", "stress_level"]):
        df["distress_index"] = (
            df["depression_score"] / 30.0               # PHQ-9 max ≈ 27
            + df["anxiety_score"] / 21.0                # GAD-7 max = 21
            + df["stress_level"] / 10.0
        ) / 3.0

    # ── Age group ─────────────────────────────────────────────
    if "age" in df.columns:
        bins = [0, 18, 25, 35, 50, 65, 200]
        labels = ["<18", "18-25", "26-35", "36-50", "51-65", "65+"]
        df["age_group"] = pd.cut(df["age"], bins=bins, labels=labels, right=False)

    # ── Binary flags ──────────────────────────────────────────
    if "mental_health_history" in df.columns:
        df["has_mh_history"] = (df["mental_health_history"].str.lower() == "yes").astype(int)
    if "seeks_treatment" in df.columns:
        df["seeks_treatment_flag"] = (df["seeks_treatment"].str.lower() == "yes").astype(int)

    logger.info(f"After feature engineering: {df.shape[1]} columns")
    return df


def encode_categoricals(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Label-encode categorical columns.
    Returns the encoded DataFrame and a mapping dict (for decoding later).
    """
    mapping: dict[str, dict] = {}

    all_cat_cols = CATEGORICAL_COLS + ["age_group"]
    for col in all_cat_cols:
        if col not in df.columns:
            continue
        df[col] = df[col].astype(str)
        unique_vals = sorted(df[col].unique())
        enc = {v: i for i, v in enumerate(unique_vals)}
        df[col] = df[col].map(enc)
        mapping[col] = enc

    # Encode target
    target_enc = {"Low": 0, "Medium": 1, "High": 2}
    df[TARGET] = df[TARGET].map(target_enc)
    df.dropna(subset=[TARGET], inplace=True)
    df[TARGET] = df[TARGET].astype(int)
    mapping[TARGET] = target_enc

    logger.info(f"Encoded {len(mapping)} categorical columns")
    return df, mapping


def run_pipeline() -> Path:
    """Full preprocessing pipeline. Returns path to saved CSV."""
    df = load_and_merge()
    df = clean_dataframe(df)
    df = engineer_features(df)
    df, mapping = encode_categoricals(df)

    out_path = cfg.DATA_PROCESSED / "structured_clean.csv"
    df.to_csv(out_path, index=False)
    logger.success(f"Saved structured data → {out_path}")

    # Save mapping for inference
    import json
    mapping_path = cfg.DATA_PROCESSED / "label_mapping.json"
    with open(mapping_path, "w") as f:
        json.dump(mapping, f, indent=2)
    logger.success(f"Saved label mapping → {mapping_path}")

    return out_path


if __name__ == "__main__":
    run_pipeline()

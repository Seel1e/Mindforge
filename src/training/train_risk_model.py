"""
src/training/train_risk_model.py
──────────────────────────────────
Trains an XGBoost classifier to predict mental health risk
(Low / Medium / High) from the structured demographic + health data.

Plain English:
  This is a completely separate model from the LLM.
  It takes numbers like age, stress level, sleep hours and predicts
  the risk category. Think of it as a "quick triage" tool.

  We also use SHAP to explain WHY the model made each prediction
  (e.g. "High risk because: stress=9/10, sleep=4h, depression_score=22").

Usage:
  python -m src.training.train_risk_model
"""

import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from loguru import logger

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    accuracy_score,
)
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
import shap
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import cfg
from src.preprocessing.prepare_structured import run_pipeline as preprocess_structured

TARGET = "mental_health_risk"
LABEL_NAMES = {0: "Low", 1: "Medium", 2: "High"}


def load_data() -> pd.DataFrame:
    """Load preprocessed structured CSV (run preprocessing if missing)."""
    clean_path = cfg.DATA_PROCESSED / "structured_clean.csv"
    if not clean_path.exists():
        logger.info("Preprocessed data not found — running preprocessing …")
        preprocess_structured()
    return pd.read_csv(clean_path)


def split_features_target(df: pd.DataFrame) -> tuple:
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    return X, y


def train(save_plots: bool = True) -> dict:
    """
    Full training pipeline.
    Returns a dict with metrics.
    """
    # ── 1. Load data ──────────────────────────────────────────
    df = load_data()
    X, y = split_features_target(df)
    logger.info(f"Feature matrix: {X.shape}  |  Class distribution:\n{y.value_counts()}")

    feature_names = X.columns.tolist()

    # ── 2. Train/test split ───────────────────────────────────
    r_cfg = cfg.risk_model
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=r_cfg["test_size"],
        stratify=y,
        random_state=42,
    )

    # ── 3. Model pipeline ─────────────────────────────────────
    xgb_params = {k: v for k, v in r_cfg["xgb_params"].items()}
    xgb_params.pop("use_label_encoder", None)   # deprecated in newer XGB

    model = XGBClassifier(**xgb_params)

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("xgb", model),
    ])

    # ── 4. Cross-validation ───────────────────────────────────
    logger.info(f"Running {r_cfg['cv_folds']}-fold cross validation …")
    cv = StratifiedKFold(n_splits=r_cfg["cv_folds"], shuffle=True, random_state=42)
    cv_scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="accuracy", n_jobs=-1)
    logger.info(f"CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # ── 5. Fit on full train set ───────────────────────────────
    logger.info("Fitting final model …")
    pipe.fit(X_train, y_train)

    # ── 6. Evaluate ───────────────────────────────────────────
    y_pred = pipe.predict(X_test)
    y_prob = pipe.predict_proba(X_test)

    acc = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob, multi_class="ovr", average="macro")

    logger.info(f"\n{classification_report(y_test, y_pred, target_names=list(LABEL_NAMES.values()))}")
    logger.info(f"Test Accuracy : {acc:.4f}")
    logger.info(f"ROC-AUC (OVR) : {roc_auc:.4f}")

    metrics = {
        "cv_accuracy_mean": float(cv_scores.mean()),
        "cv_accuracy_std": float(cv_scores.std()),
        "test_accuracy": float(acc),
        "roc_auc_ovr": float(roc_auc),
        "feature_names": feature_names,
    }

    # ── 7. Save model ─────────────────────────────────────────
    model_path = cfg.MODELS_DIR / "risk_predictor.pkl"
    joblib.dump(pipe, model_path)
    logger.success(f"Model saved → {model_path}")

    metrics_path = cfg.MODELS_DIR / "risk_model_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.success(f"Metrics saved → {metrics_path}")

    # ── 8. SHAP explainability ────────────────────────────────
    if save_plots:
        _plot_shap(pipe["xgb"], pipe["scaler"].transform(X_test), feature_names)
        _plot_confusion_matrix(y_test, y_pred)

    return metrics


def _plot_shap(xgb_model, X_test_scaled, feature_names: list):
    """Generate SHAP summary plot (feature importance)."""
    logger.info("Computing SHAP values …")
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X_test_scaled)

    plots_dir = cfg.MODELS_DIR / "plots"
    plots_dir.mkdir(exist_ok=True)

    # Summary bar plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(
        shap_values, X_test_scaled,
        feature_names=feature_names,
        plot_type="bar",
        show=False,
        class_names=list(LABEL_NAMES.values()),
    )
    plt.tight_layout()
    shap_path = plots_dir / "shap_summary.png"
    plt.savefig(shap_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.success(f"SHAP plot saved → {shap_path}")


def _plot_confusion_matrix(y_true, y_pred):
    """Plot and save the confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    labels = list(LABEL_NAMES.values())

    plots_dir = cfg.MODELS_DIR / "plots"
    plots_dir.mkdir(exist_ok=True)

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=labels, yticklabels=labels,
    )
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.title("Mental Health Risk — Confusion Matrix")
    plt.tight_layout()
    cm_path = plots_dir / "confusion_matrix.png"
    plt.savefig(cm_path, dpi=150)
    plt.close()
    logger.success(f"Confusion matrix saved → {cm_path}")


def predict_single(sample: dict, model_path: Path = None) -> dict:
    """
    Predict risk for a single sample dict.

    Args:
        sample: e.g. {"age": 28, "stress_level": 8, "sleep_hours": 5, ...}
    Returns:
        {"risk": "High", "probabilities": {"Low": 0.1, "Medium": 0.2, "High": 0.7}}
    """
    if model_path is None:
        model_path = cfg.MODELS_DIR / "risk_predictor.pkl"

    pipe = joblib.load(model_path)
    df = pd.DataFrame([sample])

    # Align columns to training feature order
    metrics_path = cfg.MODELS_DIR / "risk_model_metrics.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            feature_names = json.load(f)["feature_names"]
        for col in feature_names:
            if col not in df.columns:
                df[col] = 0
        df = df[feature_names]

    pred = pipe.predict(df)[0]
    proba = pipe.predict_proba(df)[0]

    return {
        "risk": LABEL_NAMES[pred],
        "probabilities": {LABEL_NAMES[i]: float(p) for i, p in enumerate(proba)},
    }


if __name__ == "__main__":
    train()

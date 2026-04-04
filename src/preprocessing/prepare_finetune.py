"""
src/preprocessing/prepare_finetune.py
──────────────────────────────────────
Converts ALL text-based datasets into a single JSONL file that the
SFTTrainer (fine-tuning script) can consume.

Plain English:
  The LLM learns by reading thousands of examples in the format:
      <|system|>You are a compassionate mental health assistant.
      <|user|>I feel really anxious today.
      <|assistant|>I hear you. Anxiety can feel overwhelming…

  This script reads every dataset and converts them into that format,
  then saves a combined train.jsonl and val.jsonl file.

Usage:
  python -m src.preprocessing.prepare_finetune
"""

import json
import random
from pathlib import Path
from typing import Iterator

import pandas as pd
from loguru import logger
from tqdm import tqdm

from src.config import cfg
from src.preprocessing.clean_text import clean, truncate

# ── System prompt used for ALL examples ──────────────────────
SYSTEM_PROMPT = (
    "You are MindForge, a compassionate and knowledgeable mental health AI assistant. "
    "You provide evidence-based psychological support, education, and guidance. "
    "Always remind users to seek professional help for serious concerns. "
    "Be empathetic, non-judgmental, and supportive."
)

# ── Mistral chat template ─────────────────────────────────────
def format_mistral(system: str, user: str, assistant: str) -> str:
    """
    Format a single (system, user, assistant) triple into the
    Mistral/Llama-3 chat format that the tokenizer expects.
    """
    return (
        f"<s>[INST] <<SYS>>\n{system}\n<</SYS>>\n\n"
        f"{user} [/INST] {assistant} </s>"
    )


def _iter_psychology_json(path: Path) -> Iterator[dict]:
    """
    Yields formatted examples from Alpie-core_medical_psychology_dataset.json.
    Each record has: prompt | complex_cot | response

    We create TWO types of examples per record:
      1. Standard Q&A  (prompt → response)
      2. Chain-of-Thought Q&A  (prompt → think step by step + response)
    """
    logger.info(f"Loading psychology JSON from {path} …")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Handle both list-of-dicts and dict-of-lists formats
    if isinstance(data, list):
        records = data
    elif isinstance(data, dict):
        # Try to find the list inside
        records = next((v for v in data.values() if isinstance(v, list)), [])
    else:
        records = []

    logger.info(f"  → {len(records):,} psychology records")

    for rec in records:
        prompt = clean(str(rec.get("prompt", "")))
        response = clean(str(rec.get("response", "")))
        cot = clean(str(rec.get("complex_cot", "")))

        if not prompt or not response:
            continue

        # Example type 1 — direct answer
        yield {
            "text": format_mistral(SYSTEM_PROMPT, prompt, truncate(response, 1500)),
            "source": "psychology_qa",
        }

        # Example type 2 — chain-of-thought answer (if cot exists)
        if cot and len(cot) > 50:
            cot_answer = f"[Thinking step by step]\n{truncate(cot, 600)}\n\n[Answer]\n{truncate(response, 900)}"
            yield {
                "text": format_mistral(SYSTEM_PROMPT, prompt, cot_answer),
                "source": "psychology_cot",
            }


def _iter_therapy_qa(path: Path) -> Iterator[dict]:
    """
    Yields examples from mental_health_dataset.csv.
    Columns: Context | Response
    """
    logger.info(f"Loading therapy Q&A from {path} …")
    df = pd.read_csv(path)
    logger.info(f"  → {len(df):,} therapy rows")

    for _, row in df.iterrows():
        context = clean(str(row.get("Context", row.iloc[0])))
        response = clean(str(row.get("Response", row.iloc[1])))

        if not context or not response or len(context) < 10:
            continue

        yield {
            "text": format_mistral(SYSTEM_PROMPT, context, truncate(response, 1500)),
            "source": "therapy_qa",
        }


def _iter_clean_statements(path: Path) -> Iterator[dict]:
    """
    Yields examples from cleanData.csv.
    Columns: statement | status (e.g., Anxiety, Depression …)

    We turn each statement into an instruction to classify it,
    teaching the model to identify mental health signals.
    """
    logger.info(f"Loading clean statements from {path} …")
    df = pd.read_csv(path)
    logger.info(f"  → {len(df):,} statement rows")

    # Map raw status labels to friendly names
    label_map = {
        "anxiety": "Anxiety",
        "depression": "Depression",
        "normal": "Normal / No concern",
        "suicidal": "Suicidal ideation — immediate professional help is needed",
        "stress": "Stress",
        "bipolar": "Bipolar tendencies",
        "personality disorder": "Personality disorder indicators",
    }

    for _, row in df.iterrows():
        statement = clean(str(row.get("statement", row.iloc[1])))
        raw_status = str(row.get("status", row.iloc[2])).strip().lower()
        status = label_map.get(raw_status, raw_status.title())

        if not statement or len(statement) < 8:
            continue

        user_msg = (
            f"Please analyse this statement and identify any mental health signals: "
            f'"{statement}"'
        )
        assistant_msg = (
            f"Based on the language and emotional tone of this statement, "
            f"it shows signs consistent with: **{status}**.\n\n"
            f"If you or someone you know is struggling, please reach out to a "
            f"mental health professional or a crisis helpline."
        )

        yield {
            "text": format_mistral(SYSTEM_PROMPT, user_msg, assistant_msg),
            "source": "classification",
        }


def build_dataset(
    val_split: float = 0.05,
    max_psychology: int = 20_000,   # cap to avoid VRAM blowout
    max_therapy: int = 10_000,
    max_statements: int = 15_000,
    seed: int = 42,
) -> tuple[Path, Path]:
    """
    Combines all iterators, shuffles, splits train/val, and saves JSONL files.

    Returns:
        (train_path, val_path)
    """
    random.seed(seed)
    all_examples: list[dict] = []

    # ── 1. Psychology JSON ────────────────────────────────────
    psych_examples = list(_iter_psychology_json(cfg.DS_PSYCHOLOGY_JSON))
    if max_psychology and len(psych_examples) > max_psychology:
        psych_examples = random.sample(psych_examples, max_psychology)
    all_examples.extend(psych_examples)
    logger.info(f"Psychology examples collected: {len(psych_examples):,}")

    # ── 2. Therapy Q&A ────────────────────────────────────────
    therapy_examples = list(_iter_therapy_qa(cfg.DS_THERAPY_QA))
    if max_therapy and len(therapy_examples) > max_therapy:
        therapy_examples = random.sample(therapy_examples, max_therapy)
    all_examples.extend(therapy_examples)
    logger.info(f"Therapy Q&A examples collected: {len(therapy_examples):,}")

    # ── 3. Clean Statements ───────────────────────────────────
    statement_examples = list(_iter_clean_statements(cfg.DS_CLEAN_STATEMENTS))
    if max_statements and len(statement_examples) > max_statements:
        statement_examples = random.sample(statement_examples, max_statements)
    all_examples.extend(statement_examples)
    logger.info(f"Statement examples collected: {len(statement_examples):,}")

    # ── Shuffle & split ───────────────────────────────────────
    random.shuffle(all_examples)
    split_idx = int(len(all_examples) * (1 - val_split))
    train_data = all_examples[:split_idx]
    val_data = all_examples[split_idx:]

    logger.info(f"Total examples: {len(all_examples):,}  |  Train: {len(train_data):,}  |  Val: {len(val_data):,}")

    # ── Save JSONL ────────────────────────────────────────────
    out_dir = cfg.DATA_PROCESSED
    train_path = out_dir / "train.jsonl"
    val_path = out_dir / "val.jsonl"

    def _write_jsonl(records: list, path: Path):
        with open(path, "w", encoding="utf-8") as f:
            for rec in tqdm(records, desc=f"Writing {path.name}"):
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    _write_jsonl(train_data, train_path)
    _write_jsonl(val_data, val_path)

    logger.success(f"Saved → {train_path}")
    logger.success(f"Saved → {val_path}")

    # ── Source distribution stats ─────────────────────────────
    from collections import Counter
    counts = Counter(ex["source"] for ex in all_examples)
    logger.info("Source distribution:")
    for src, n in counts.most_common():
        logger.info(f"  {src:<25} {n:>7,}")

    return train_path, val_path


if __name__ == "__main__":
    build_dataset()

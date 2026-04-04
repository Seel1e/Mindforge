"""
src/training/finetune_llm.py
─────────────────────────────
Fine-tunes a 7B-parameter LLM on our mental health datasets using
QLoRA (Quantised LoRA) via Unsloth — the fastest open-source fine-tuning library.

Plain English:
  "Fine-tuning" means taking a pre-trained model (like Mistral-7B,
   which already knows English) and teaching it new things from our
   datasets using a method called LoRA.

   LoRA is clever: instead of retraining ALL 7 billion parameters
   (would need 80+ GB of VRAM), it freezes the original model and
   trains a tiny set of "adapter" matrices that modify its behaviour.
   With 4-bit quantisation (QLoRA) it fits in 8–16 GB of VRAM.

Usage:
  python -m src.training.finetune_llm

  Or with custom overrides:
  python -m src.training.finetune_llm --epochs 5 --lr 1e-4
"""

import json
import typer
from pathlib import Path
from loguru import logger

# ── Delay heavy imports so the module is importable even without GPU ──
def _import_training_libs():
    try:
        from unsloth import FastLanguageModel
        from trl import SFTTrainer
        from transformers import TrainingArguments
        from datasets import Dataset
        return FastLanguageModel, SFTTrainer, TrainingArguments, Dataset
    except ImportError as e:
        raise ImportError(
            f"Training libraries not installed: {e}\n"
            "Run: pip install unsloth trl transformers datasets"
        )


app = typer.Typer()


def load_jsonl(path: Path) -> list[dict]:
    """Load a JSONL file into a list of dicts."""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def prepare_datasets(train_path: Path, val_path: Path):
    """Load JSONL files and convert to HuggingFace Dataset objects."""
    from datasets import Dataset

    train_records = load_jsonl(train_path)
    val_records = load_jsonl(val_path)

    # The SFTTrainer just needs a "text" column
    train_ds = Dataset.from_list(train_records)
    val_ds = Dataset.from_list(val_records)

    logger.info(f"Train set: {len(train_ds):,} examples")
    logger.info(f"Val set:   {len(val_ds):,} examples")
    return train_ds, val_ds


@app.command()
def train(
    epochs: int = typer.Option(None, help="Override config num_train_epochs"),
    lr: float = typer.Option(None, help="Override learning rate"),
    batch_size: int = typer.Option(None, help="Override per_device_train_batch_size"),
    dry_run: bool = typer.Option(False, help="Load model only, don't train (sanity check)"),
):
    """
    Main fine-tuning entry point.
    Reads config.yaml for all hyper-parameters (override with CLI flags).
    """
    from src.config import cfg
    from src.preprocessing.prepare_finetune import build_dataset

    FastLanguageModel, SFTTrainer, TrainingArguments, Dataset = _import_training_libs()

    # ── 1. Prepare dataset (if not already done) ─────────────
    train_jsonl = cfg.DATA_PROCESSED / "train.jsonl"
    val_jsonl = cfg.DATA_PROCESSED / "val.jsonl"

    if not train_jsonl.exists() or not val_jsonl.exists():
        logger.info("JSONL files not found — running preprocessing first …")
        build_dataset()

    train_ds, val_ds = prepare_datasets(train_jsonl, val_jsonl)

    # ── 2. Load base model (4-bit quantised) ─────────────────
    m_cfg = cfg.model
    l_cfg = cfg.lora
    t_cfg = cfg.training

    logger.info(f"Loading base model: {m_cfg['base_model']} …")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=m_cfg["base_model"],
        max_seq_length=m_cfg["max_seq_length"],
        dtype=None,                             # auto-detect
        load_in_4bit=m_cfg["load_in_4bit"],
        token=cfg.HF_TOKEN or None,
    )

    # ── 3. Add LoRA adapters ──────────────────────────────────
    logger.info("Attaching LoRA adapters …")
    model = FastLanguageModel.get_peft_model(
        model,
        r=l_cfg["r"],
        target_modules=l_cfg["target_modules"],
        lora_alpha=l_cfg["lora_alpha"],
        lora_dropout=l_cfg["lora_dropout"],
        bias=l_cfg["bias"],
        use_gradient_checkpointing=l_cfg["use_gradient_checkpointing"],
        random_state=t_cfg.get("seed", 42),
    )

    # Print trainable parameter count
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Trainable params: {trainable:,} / {total:,}  ({100*trainable/total:.2f}%)")

    if dry_run:
        logger.warning("Dry run mode — skipping training.")
        return

    # ── 4. Configure training ─────────────────────────────────
    output_dir = str(cfg.MODELS_DIR / "mindforge-lora")

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs or t_cfg["num_train_epochs"],
        per_device_train_batch_size=batch_size or t_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=t_cfg["gradient_accumulation_steps"],
        warmup_steps=t_cfg["warmup_steps"],
        learning_rate=lr or t_cfg["learning_rate"],
        weight_decay=t_cfg["weight_decay"],
        lr_scheduler_type=t_cfg["lr_scheduler_type"],
        fp16=t_cfg["fp16"],
        logging_steps=t_cfg["logging_steps"],
        save_steps=t_cfg["save_steps"],
        eval_steps=t_cfg["eval_steps"],
        save_total_limit=t_cfg["save_total_limit"],
        evaluation_strategy=t_cfg["evaluation_strategy"],
        load_best_model_at_end=t_cfg["load_best_model_at_end"],
        report_to=t_cfg["report_to"],
        run_name=t_cfg["run_name"],
        seed=t_cfg.get("seed", 42),
    )

    # ── 5. SFTTrainer ─────────────────────────────────────────
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        dataset_text_field="text",              # the column our JSONL has
        max_seq_length=m_cfg["max_seq_length"],
        args=training_args,
        packing=False,                          # set True to pack short sequences (faster)
    )

    logger.info("Starting training …")
    trainer_stats = trainer.train()

    # ── 6. Save ───────────────────────────────────────────────
    logger.info(f"Saving model to {output_dir} …")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Save in merged 16-bit format for easier inference
    merged_dir = str(cfg.MODELS_DIR / "mindforge-merged")
    logger.info(f"Saving merged 16-bit model to {merged_dir} …")
    model.save_pretrained_merged(merged_dir, tokenizer, save_method="merged_16bit")

    logger.success("Training complete!")
    logger.info(f"Training time: {trainer_stats.metrics['train_runtime']:.1f}s")
    logger.info(f"Final train loss: {trainer_stats.metrics.get('train_loss', 'N/A')}")


if __name__ == "__main__":
    app()

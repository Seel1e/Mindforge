"""
run.py
───────
One-file command-line runner for MindForge.

Usage examples:
  python run.py preprocess          # Step 1: prepare datasets
  python run.py build-index         # Step 2: build RAG vector store
  python run.py train-risk          # Step 3: train XGBoost risk model
  python run.py train-llm           # Step 4: fine-tune the LLM (needs GPU)
  python run.py evaluate            # Step 5: evaluate the fine-tuned LLM
  python run.py app                 # Step 6: launch the Streamlit chat app
  python run.py api                 # Launch the FastAPI REST API
  python run.py all                 # Run steps 1-4 in sequence
"""

import sys
import subprocess
import typer

app = typer.Typer(help="MindForge CLI")


@app.command()
def preprocess(
    max_psych: int = typer.Option(20_000, help="Max psychology examples"),
    max_therapy: int = typer.Option(10_000, help="Max therapy Q&A examples"),
    max_statements: int = typer.Option(15_000, help="Max classification examples"),
):
    """Step 1: Clean and prepare all datasets for training."""
    from src.preprocessing.prepare_finetune import build_dataset
    from src.preprocessing.prepare_structured import run_pipeline

    typer.echo("→ Building fine-tuning JSONL dataset …")
    build_dataset(max_psychology=max_psych, max_therapy=max_therapy, max_statements=max_statements)

    typer.echo("→ Building structured ML dataset …")
    run_pipeline()

    typer.secho("✓ Preprocessing complete!", fg=typer.colors.GREEN)


@app.command()
def build_index(
    max_psych: int = typer.Option(5_000, help="Max psychology chunks to index"),
    max_therapy: int = typer.Option(3_000, help="Max therapy chunks to index"),
):
    """Step 2: Build the ChromaDB RAG vector store."""
    from src.rag.build_index import build_vector_store
    typer.echo("→ Building RAG vector store …")
    build_vector_store(max_psych=max_psych, max_therapy=max_therapy)
    typer.secho("✓ Vector store built!", fg=typer.colors.GREEN)


@app.command()
def train_risk():
    """Step 3: Train the XGBoost mental health risk predictor."""
    from src.training.train_risk_model import train
    typer.echo("→ Training XGBoost risk model …")
    metrics = train()
    typer.secho(f"✓ Risk model trained! Test accuracy: {metrics['test_accuracy']:.4f}", fg=typer.colors.GREEN)


@app.command()
def train_llm(
    epochs: int = typer.Option(None),
    lr: float = typer.Option(None),
    dry_run: bool = typer.Option(False),
):
    """Step 4: Fine-tune the LLM with QLoRA (requires GPU with 8+ GB VRAM)."""
    typer.echo("→ Launching LLM fine-tuning …")
    from src.training.finetune_llm import train
    train(epochs=epochs, lr=lr, dry_run=dry_run)
    typer.secho("✓ LLM fine-tuning complete!", fg=typer.colors.GREEN)


@app.command()
def evaluate(
    max_samples: int = typer.Option(200),
    skip_bertscore: bool = typer.Option(False),
):
    """Step 5: Evaluate the fine-tuned LLM."""
    from src.evaluation.evaluate import run_full_evaluation
    metrics = run_full_evaluation(max_samples=max_samples, skip_bertscore=skip_bertscore)
    typer.secho(f"✓ Evaluation complete! ROUGE-L: {metrics.get('rougeL', 'N/A'):.4f}", fg=typer.colors.GREEN)


@app.command()
def chat_app():
    """Step 6: Launch the Streamlit chat interface."""
    typer.echo("→ Starting Streamlit app …")
    subprocess.run([sys.executable, "-m", "streamlit", "run", "app/app.py"], check=True)


@app.command()
def api():
    """Launch the FastAPI REST API."""
    typer.echo("→ Starting FastAPI server on http://localhost:8000 …")
    subprocess.run([
        sys.executable, "-m", "uvicorn",
        "app.api:app", "--host", "0.0.0.0", "--port", "8000", "--reload",
    ], check=True)


@app.command()
def all_steps():
    """Run the full pipeline: preprocess → build-index → train-risk → train-llm."""
    typer.echo("Running full MindForge pipeline …\n")
    preprocess()
    build_index()
    train_risk()
    train_llm()
    typer.secho("\n✓ Full pipeline complete!", fg=typer.colors.GREEN, bold=True)


if __name__ == "__main__":
    app()

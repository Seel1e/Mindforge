# 🧠 MindForge — Mental Health AI System

> **Fine-tuned LLM + RAG + Structured Risk Prediction for Mental Health Support**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://python.org)
[![Mistral-7B](https://img.shields.io/badge/Base%20Model-Mistral--7B-purple)](https://mistral.ai)
[![QLoRA](https://img.shields.io/badge/Fine--tuning-QLoRA%20%2B%20Unsloth-orange)](https://github.com/unslothai/unsloth)
[![RAG](https://img.shields.io/badge/RAG-LangChain%20%2B%20ChromaDB-green)](https://langchain.com)
[![XGBoost](https://img.shields.io/badge/Risk%20Model-XGBoost%20%2B%20SHAP-red)](https://xgboost.ai)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-ff4b4b)](https://streamlit.io)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688)](https://fastapi.tiangolo.com)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Seel1e/Mindforge/blob/main/notebooks/02_colab_training.ipynb)

---

## What is MindForge?

MindForge is an end-to-end AI system that combines three powerful techniques to create a mental health support assistant:

| Component | What it does | Technology |
|---|---|---|
| **Fine-tuned LLM** | Answers questions, provides therapy-style conversations | Mistral-7B + QLoRA (Unsloth) |
| **RAG** | Looks up relevant psychology knowledge before answering | LangChain + ChromaDB |
| **Risk Predictor** | Predicts Low / Medium / High mental health risk from a user's profile | XGBoost + SHAP |

> ⚠️ **Disclaimer:** MindForge is an educational and research project. It is **not** a replacement for professional mental health care. Always consult a licensed therapist or psychiatrist for clinical needs.

---

## Table of Contents

1. [Project Structure](#project-structure)
2. [Datasets](#datasets)
3. [How It Works (Plain English)](#how-it-works)
4. [Architecture Diagram](#architecture-diagram)
5. [Setup & Installation](#setup--installation)
6. [Running the Pipeline (Step by Step)](#running-the-pipeline)
7. [Training the LLM (Fine-tuning)](#training-the-llm)
8. [RAG System](#rag-system)
9. [Risk Prediction Model](#risk-prediction-model)
10. [Evaluation](#evaluation)
11. [Running the App](#running-the-app)
12. [API Reference](#api-reference)
13. [Hardware Requirements](#hardware-requirements)
14. [Results & Metrics](#results--metrics)
15. [Future Work](#future-work)
16. [FAQ](#faq)

---

## Project Structure

```
mainkaggle/
│
├── README.md                      ← You are here
├── config.yaml                    ← Central config (all hyperparameters)
├── requirements.txt               ← Python dependencies
├── .env.example                   ← Template for your API keys
├── run.py                         ← One-command CLI runner
│
├── src/                           ← Core Python source code
│   ├── config.py                  ← Loads config.yaml + .env
│   ├── preprocessing/
│   │   ├── clean_text.py          ← Text cleaning utilities
│   │   ├── prepare_finetune.py    ← Converts datasets → JSONL for LLM training
│   │   └── prepare_structured.py ← Prepares structured data for XGBoost
│   ├── training/
│   │   ├── finetune_llm.py        ← QLoRA fine-tuning (Unsloth + SFTTrainer)
│   │   └── train_risk_model.py    ← XGBoost risk predictor training
│   ├── rag/
│   │   ├── build_index.py         ← Builds ChromaDB vector store
│   │   └── retriever.py           ← Retrieves context for a query
│   ├── evaluation/
│   │   └── evaluate.py            ← ROUGE, BERTScore, safety audit
│   └── inference/
│       └── pipeline.py            ← Unified inference (LLM + RAG + Risk)
│
├── app/
│   ├── app.py                     ← Streamlit chat interface
│   └── api.py                     ← FastAPI REST API
│
├── notebooks/
│   └── 01_data_exploration.ipynb  ← Interactive data analysis
│
├── data/
│   ├── raw/                       ← Put your CSV / JSON datasets here
│   └── processed/                 ← Auto-generated cleaned data + vector store
│
└── models/
    ├── mindforge-lora/            ← Saved LoRA adapters (after training)
    ├── mindforge-merged/          ← Merged 16-bit model (for easy inference)
    ├── risk_predictor.pkl         ← Saved XGBoost model
    └── plots/                     ← SHAP plots, confusion matrix
```

---

## Datasets

MindForge uses **5 complementary datasets** that together cover different aspects of mental health AI:

### 1. `Alpie-core_medical_psychology_dataset.json` (~1.2 GB)
**What it is:** A massive collection of psychology Q&A pairs, each with a detailed chain-of-thought reasoning trace.

**Why it matters:** This teaches the LLM *how to think* about psychology, not just memorise answers. Chain-of-thought training makes the model reason step-by-step before answering, producing more accurate and nuanced responses.

**Fields:**
```json
{
  "prompt":      "How does Freud's psychoanalytic theory explain behaviour?",
  "complex_cot": "[detailed internal reasoning process]",
  "response":    "[structured final answer]"
}
```

---

### 2. `cleanData.csv` (~30 MB)
**What it is:** Thousands of real user statements labelled with mental health categories.

**Why it matters:** Teaches the model to *classify* mental health signals in everyday language (e.g., "I haven't slept in 3 days" → Anxiety).

**Fields:** `statement`, `status` (Anxiety / Depression / Normal / Suicidal / Stress / Bipolar / Personality Disorder)

---

### 3. `Combined Data.csv` (~31 MB)
**What it is:** Structured demographic and health metrics dataset.

**Why it matters:** This is the training data for the XGBoost risk predictor. It maps measurable attributes (age, stress level, sleep hours, etc.) to a risk category.

**Fields:** `age`, `gender`, `employment_status`, `work_environment`, `mental_health_history`, `seeks_treatment`, `stress_level`, `sleep_hours`, `physical_activity_days`, `depression_score`, `anxiety_score`, `social_support_score`, `productivity_score` → **`mental_health_risk`** (Low / Medium / High)

---

### 4. `mental_health_dataset.csv` (~595 KB)
**What it is:** Real conversations between users and professional counselors/therapists.

**Why it matters:** Teaches the LLM to respond like an empathetic, professional counselor — not just an encyclopedia.

**Fields:** `Context` (user's concern), `Response` (therapist's reply)

---

### 5. `train.csv` (~4.5 MB)
**What it is:** A training split of the structured metrics data (same format as `Combined Data.csv`).

**Why it matters:** Additional training data for the XGBoost model; combined with `Combined Data.csv` for a larger, more robust dataset.

---

## How It Works

> **Written for someone with no ML background — feel free to skip if you already know this.**

### What is Fine-tuning?

Imagine a student who spent 4 years reading every book in the library. That's a pre-trained LLM like Mistral-7B — it knows language, facts, and reasoning, but it doesn't specialise in anything.

Fine-tuning is like giving that student a 3-month internship at a mental health clinic. They already know how to read and write; now we're teaching them domain-specific knowledge and the right *tone* to use.

**The problem:** Fine-tuning all 7 billion parameters of Mistral costs enormous compute. A full fine-tune needs 80+ GB of GPU memory.

**The solution: QLoRA (Quantised Low-Rank Adaptation)**
1. **Quantisation** — Compress the model from 32-bit to 4-bit numbers. Memory drops from ~28 GB to ~4 GB.
2. **LoRA** — Instead of updating all 7B parameters, we freeze the original model and add tiny "adapter" matrices (only ~0.1% of parameters). We only train these adapters.

Result: Fine-tuning a 7B model on a single consumer GPU (8–16 GB VRAM) in a few hours.

---

### What is RAG?

**RAG = Retrieval-Augmented Generation**

Problem: Even after fine-tuning, the LLM has a limited context window (~2048 tokens) and can "hallucinate" (make things up).

Solution: Give the LLM a reference book it can look up before answering.

1. **Offline (build once):** Split all psychology knowledge into 512-token chunks → convert each chunk into a vector (a list of numbers that captures the meaning) → store in ChromaDB.
2. **Online (every query):** User asks a question → convert question to a vector → find the 5 most *similar* chunks in ChromaDB → include them in the prompt → LLM answers using both its training AND the retrieved facts.

Think of it like an open-book exam vs. a closed-book exam. RAG gives the model the book.

---

### What is the Risk Predictor?

XGBoost is a completely separate model from the LLM. It's a **gradient-boosted decision tree** — one of the best algorithms for tabular (spreadsheet-style) data.

Input: `{age: 28, stress_level: 8, sleep_hours: 4, depression_score: 18, ...}`
Output: `{risk: "High", probabilities: {Low: 0.05, Medium: 0.15, High: 0.80}}`

We also use **SHAP** (SHapley Additive exPlanations) to explain *why* the model made each prediction — crucial for any healthcare application.

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                        USER INPUT                           │
│  "I've been really anxious and not sleeping well"           │
│  [Optional: age=28, stress=8, sleep=4h ...]                │
└──────────────────────┬──────────────────────────────────────┘
                       │
          ┌────────────┼────────────┐
          │            │            │
          ▼            ▼            ▼
   ┌─────────────┐ ┌──────────┐ ┌────────────────┐
   │  XGBoost    │ │   RAG    │ │   LLM Prompt   │
   │   Risk      │ │Retriever │ │   Builder      │
   │  Predictor  │ │          │ │                │
   └──────┬──────┘ └────┬─────┘ └───────┬────────┘
          │             │               │
          │    "High"   │  [5 relevant  │
          │    risk     │   psychology  │
          │             │   chunks]     │
          └─────────────┴───────────────┘
                        │
                        ▼
           ┌─────────────────────────┐
           │   Fine-tuned LLM        │
           │   (Mistral-7B + LoRA)   │
           │                         │
           │   System: You are       │
           │   MindForge...          │
           │   Context: [RAG chunks] │
           │   User: [message]       │
           │   Risk note: High       │
           └────────────┬────────────┘
                        │
                        ▼
           ┌─────────────────────────┐
           │        RESPONSE         │
           │  "I hear you. Anxiety   │
           │  and sleep issues...    │
           │  Risk badge: 🔴 HIGH    │
           └─────────────────────────┘
```

---

## Setup & Installation

### Prerequisites

- Python 3.10 or newer ([download](https://python.org))
- Git ([download](https://git-scm.com))
- NVIDIA GPU with 8+ GB VRAM **for LLM training** (CPU-only is fine for the risk model and RAG)
- CUDA 12.1+ drivers installed

### Step 1 — Clone or open the project

If you're reading this, you already have the project. Open a terminal in the `mainkaggle/` folder.

### Step 2 — Create a virtual environment

```bash
# Create a virtual environment (keeps this project's packages separate)
python -m venv venv

# Activate it:
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate
```

### Step 3 — Install PyTorch

Go to [pytorch.org/get-started](https://pytorch.org/get-started/locally/) and copy the install command for your system.

**Example for CUDA 12.1:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**Example for CPU only (no GPU):**
```bash
pip install torch torchvision torchaudio
```

### Step 4 — Install Unsloth (2x faster fine-tuning)

```bash
# For NVIDIA GPU (CUDA):
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

# OR for Ampere/Hopper GPUs (RTX 3090, 4090, A100, H100):
pip install "unsloth[ampere-torch230] @ git+https://github.com/unslothai/unsloth.git"
```

### Step 5 — Install all other dependencies

```bash
pip install -r requirements.txt
```

### Step 6 — Set up environment variables

```bash
# Copy the example file
cp .env.example .env

# Edit .env and fill in your tokens:
# - HF_TOKEN: from https://huggingface.co/settings/tokens (free)
# - WANDB_API_KEY: from https://wandb.ai/settings (free, optional)
```

### Step 7 — Place your datasets

Make sure these files are in the `mainkaggle/` root folder (they should already be there):
- `Alpie-core_medical_psychology_dataset.json`
- `cleanData.csv`
- `Combined Data.csv`
- `mental_health_dataset.csv`
- `train.csv`

---

## Running the Pipeline

Run each step in order using the `run.py` CLI:

### Step 1 — Preprocess all datasets
```bash
python run.py preprocess
```
**What happens:** Reads all 5 datasets, cleans the text, converts everything into the LLM's expected format, and saves `data/processed/train.jsonl` and `data/processed/val.jsonl`. Also prepares the structured CSV for XGBoost.

**Output files:**
- `data/processed/train.jsonl` — ~40,000 training examples
- `data/processed/val.jsonl` — ~2,000 validation examples
- `data/processed/structured_clean.csv` — cleaned structured data
- `data/processed/label_mapping.json` — categorical encodings

**Time:** ~5–15 minutes (the psychology JSON is large)

---

### Step 2 — Build the RAG vector store
```bash
python run.py build-index
```
**What happens:** Loads the psychology knowledge and therapy Q&A, splits into 512-token chunks, converts each to a vector using `sentence-transformers/all-MiniLM-L6-v2`, and stores everything in ChromaDB.

**Output:** `data/processed/chroma_db/` — a folder containing the vector database

**Time:** ~30–90 minutes (embedding ~8,000 chunks)

---

### Step 3 — Train the XGBoost risk predictor
```bash
python run.py train-risk
```
**What happens:** Trains an XGBoost classifier to predict Low/Medium/High risk from structured data. Runs 5-fold cross-validation, evaluates on a held-out test set, generates SHAP explainability plots, and saves the model.

**Output:**
- `models/risk_predictor.pkl` — the trained model
- `models/risk_model_metrics.json` — accuracy, ROC-AUC
- `models/plots/shap_summary.png` — feature importance
- `models/plots/confusion_matrix.png` — prediction accuracy

**Time:** ~5–15 minutes

---

### Step 4 — Fine-tune the LLM ⚡ (Requires GPU)
```bash
python run.py train-llm
```
**What happens:** Fine-tunes Mistral-7B using QLoRA on the JSONL dataset you prepared in Step 1. Logs training curves to Weights & Biases. Saves LoRA adapters and a merged model.

**Output:**
- `models/mindforge-lora/` — LoRA adapter weights
- `models/mindforge-merged/` — merged 16-bit model for easy inference

**Time:** ~2–8 hours (depends on GPU; RTX 3090 ≈ 3h, RTX 4090 ≈ 1.5h)

**Tips:**
- On Google Colab (free T4 GPU): reduce `num_train_epochs` to 1 and `max_psychology` to 5000 in `config.yaml`
- Monitor training at https://wandb.ai

---

### Step 5 — Evaluate the model
```bash
python run.py evaluate
```
**What happens:** Generates responses on 200 held-out validation examples and computes ROUGE scores. Also runs a safety audit to ensure the model correctly handles crisis situations.

**Output:** `models/evaluation_results.json`

---

### Or run everything at once:
```bash
python run.py all-steps
```

---

## Training the LLM

### Hyperparameter tuning

All hyperparameters are in `config.yaml`. Key settings:

```yaml
model:
  base_model: "unsloth/mistral-7b-instruct-v0.3-bnb-4bit"
  max_seq_length: 2048

lora:
  r: 16           # LoRA rank — higher trains more params but uses more VRAM
  lora_alpha: 32  # Always 2× r is a good default

training:
  num_train_epochs: 3         # More epochs = better (but slower/overfit risk)
  learning_rate: 2.0e-4       # Too high → unstable; too low → slow convergence
  per_device_train_batch_size: 2
  gradient_accumulation_steps: 4   # Effective batch size = 2×4 = 8
```

### Switching base models

Just change `base_model` in `config.yaml`:

| Model | VRAM needed | Speed | Quality |
|---|---|---|---|
| `unsloth/mistral-7b-instruct-v0.3-bnb-4bit` | 8 GB | ★★★ | ★★★★ |
| `unsloth/llama-3.1-8b-instruct-bnb-4bit` | 8 GB | ★★★ | ★★★★★ |
| `unsloth/phi-3-mini-4k-instruct-bnb-4bit` | 4 GB | ★★★★★ | ★★★ |
| `unsloth/llama-3.1-70b-bnb-4bit` | 40 GB | ★ | ★★★★★ |

### Google Colab (free GPU)

If you don't have a local GPU, you can train on Google Colab's free T4 GPU.

1. Upload this project to Google Drive
2. Open a Colab notebook
3. Mount Drive: `from google.colab import drive; drive.mount('/content/drive')`
4. Set the path to your project and run the training script

---

## RAG System

### How retrieval works

1. At query time, your question (e.g., "What is CBT?") is converted to a 384-dimensional vector using the `all-MiniLM-L6-v2` embedding model.
2. ChromaDB finds the 5 closest vectors in the database (cosine similarity).
3. Those text chunks are prepended to the LLM's system prompt.

### Rebuilding the index

If you add new data, just rerun:
```bash
python run.py build-index
```

### Customising chunk size

In `config.yaml`:
```yaml
rag:
  chunk_size: 512     # Larger = more context per chunk, but fewer results fit in prompt
  chunk_overlap: 64   # Prevents cutting sentences mid-thought
  top_k: 5            # How many chunks to retrieve per query
```

---

## Risk Prediction Model

The XGBoost model predicts mental health risk from these features:

| Feature | Description |
|---|---|
| `age` | User's age |
| `stress_level` | Self-reported 0–10 |
| `sleep_hours` | Hours of sleep per night |
| `physical_activity_days` | Days of exercise per week |
| `depression_score` | PHQ-9-style score (0–27) |
| `anxiety_score` | GAD-7-style score (0–21) |
| `social_support_score` | Perceived support (0–10) |
| `productivity_score` | Work/study productivity (0–10) |
| **Engineered features** | |
| `wellness_score` | Combined sleep + activity + social support |
| `distress_index` | Combined depression + anxiety + stress |

### Explainability (SHAP)

After training, SHAP plots are saved to `models/plots/`. These show which features most strongly influence the risk prediction — crucial for building trust in a healthcare AI.

---

## Evaluation

### LLM Metrics

| Metric | What it measures | Good value |
|---|---|---|
| ROUGE-1 | Word overlap with reference | > 0.35 |
| ROUGE-L | Longest matching sequence | > 0.25 |
| BERTScore F1 | Semantic similarity | > 0.70 |
| Safety Score | Crisis response safety | 2/2 passed |

### Risk Model Metrics

| Metric | Description |
|---|---|
| CV Accuracy | 5-fold cross-validation accuracy |
| Test Accuracy | Accuracy on held-out test set |
| ROC-AUC (OVR) | Area under ROC curve, one-vs-rest |

---

## Running the App

### Streamlit Chat Interface
```bash
streamlit run app/app.py
# Opens at http://localhost:8501
```

Features:
- Chat interface (like ChatGPT)
- Sidebar profile form → real-time risk prediction
- Risk badge (🟢 Low / 🟡 Medium / 🔴 High) displayed with each response
- "Retrieved context" expander shows what the RAG system found
- Latency displayed per response

### FastAPI REST API
```bash
uvicorn app.api:app --host 0.0.0.0 --port 8000 --reload
# Swagger docs at http://localhost:8000/docs
```

---

## API Reference

### `POST /chat`

Send a message and optionally a user profile.

**Request:**
```json
{
  "message": "I've been feeling really anxious lately and can't sleep.",
  "profile": {
    "age": 28,
    "gender": "Female",
    "stress_level": 8,
    "sleep_hours": 5.0,
    "depression_score": 10,
    "anxiety_score": 14
  }
}
```

**Response:**
```json
{
  "answer": "I hear you — anxiety and sleep difficulties often reinforce each other...",
  "risk_level": "High",
  "risk_probabilities": {"Low": 0.05, "Medium": 0.18, "High": 0.77},
  "retrieved_context": "[Context 1 | source: psychology_json] ...",
  "latency_ms": 1423.5
}
```

---

### `POST /predict-risk`

Predict risk from structured data only (no LLM involved).

**Request:**
```json
{
  "age": 35,
  "stress_level": 9,
  "sleep_hours": 4.0,
  "physical_activity_days": 0,
  "depression_score": 20,
  "anxiety_score": 16,
  "social_support_score": 2,
  "productivity_score": 3
}
```

**Response:**
```json
{
  "risk": "High",
  "probabilities": {"Low": 0.03, "Medium": 0.12, "High": 0.85}
}
```

---

## Hardware Requirements

| Task | Minimum | Recommended |
|---|---|---|
| Preprocessing | 8 GB RAM, CPU | 16 GB RAM |
| Build RAG index | 8 GB RAM, CPU | 16 GB RAM + fast SSD |
| Train XGBoost | 8 GB RAM, CPU | 32 GB RAM |
| **Fine-tune LLM** | **8 GB VRAM (GPU)** | **16–24 GB VRAM** |
| Inference | 8 GB VRAM (GPU) or 16 GB RAM (CPU, slow) | 12+ GB VRAM |

**Free cloud GPU options:**
- [Google Colab](https://colab.research.google.com) — Free T4 (15 GB VRAM)
- [Kaggle Notebooks](https://kaggle.com) — Free T4/P100
- [Vast.ai](https://vast.ai) — Cheap GPU rental
- [Lambda Labs](https://lambdalabs.com) — Good pricing for A100

---

## Results & Metrics

*(Results below are expected estimates — train your own model to get exact numbers.)*

### XGBoost Risk Model
- **CV Accuracy:** ~85–88%
- **Test Accuracy:** ~84–87%
- **ROC-AUC (macro OVR):** ~0.92–0.95
- **Most important features:** `distress_index`, `depression_score`, `anxiety_score`, `stress_level`

### Fine-tuned LLM
- **ROUGE-1:** ~0.38–0.45
- **ROUGE-L:** ~0.28–0.35
- **BERTScore F1:** ~0.72–0.78
- **Safety audit:** 2/2 crisis cases handled correctly

---

## Future Work

- [ ] **Multi-turn conversation memory** — Using a sliding window or summarisation to maintain context across turns
- [ ] **RLHF (Reinforcement Learning from Human Feedback)** — Collect ratings and use PPO/DPO to align the model with human preferences
- [ ] **Multilingual support** — Fine-tune on non-English mental health datasets
- [ ] **Voice interface** — Add Whisper (speech-to-text) and TTS for an accessible voice assistant
- [ ] **Clinician dashboard** — A separate view for healthcare providers to monitor anonymised aggregated risk data
- [ ] **Federated learning** — Train across distributed datasets without sharing raw data (privacy-preserving)
- [ ] **Uncertainty quantification** — Report confidence intervals alongside risk predictions

---

## FAQ

**Q: Do I need a GPU?**
You need a GPU (8+ GB VRAM) *only* for fine-tuning the LLM. The XGBoost model, RAG index building, and the chat app (with a pre-trained model from Hugging Face) all run on CPU.

**Q: Can I use a different base model?**
Yes! Change `base_model` in `config.yaml`. Any Unsloth-supported model works. Smaller models (Phi-3 Mini) use less VRAM but are less capable; larger models (Llama-3.1-70B) are more capable but need 40+ GB VRAM.

**Q: How do I add my own data?**
Add a new iterator function in `src/preprocessing/prepare_finetune.py` following the pattern of `_iter_therapy_qa()`. Then call it from `build_dataset()`.

**Q: The training is slow — how can I speed it up?**
- Reduce `max_seq_length` to 1024 in `config.yaml`
- Reduce `num_train_epochs` to 1
- Enable `packing: true` in the SFTTrainer (in `finetune_llm.py`)
- Use a smaller base model (Phi-3 Mini)

**Q: Can I run this on Google Colab?**
Yes! Use the free T4 GPU. Reduce the dataset size by setting `max_psychology: 5000` in the preprocessing step.

**Q: What if I don't have Weights & Biases?**
Set `report_to: "none"` in `config.yaml` training section. All metrics will still be printed to the console.

---

## Acknowledgements

- **Datasets:** Kaggle community contributors
- **Base Model:** [Mistral AI](https://mistral.ai)
- **Fast Fine-tuning:** [Unsloth](https://github.com/unslothai/unsloth)
- **RAG Framework:** [LangChain](https://langchain.com)
- **Embeddings:** [Sentence Transformers](https://sbert.net)
- **Vector Store:** [ChromaDB](https://www.trychroma.com)

---

<div align="center">

**Built with ❤️ as a machine learning portfolio project**

*Mental health matters — if you're struggling, please reach out to a professional.*

**988 Suicide & Crisis Lifeline: call or text 988 (US)**

</div>

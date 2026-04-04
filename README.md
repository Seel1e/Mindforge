# MindForge — Mental Health AI System

> Fine-tuned LLM + RAG + Text Classification for Mental Health Support

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://python.org)
[![Mistral-7B](https://img.shields.io/badge/Base%20Model-Mistral--7B-purple)](https://mistral.ai)
[![QLoRA](https://img.shields.io/badge/Fine--tuning-QLoRA%20%2B%20Unsloth-orange)](https://github.com/unslothai/unsloth)
[![RAG](https://img.shields.io/badge/RAG-LangChain%20%2B%20ChromaDB-green)](https://langchain.com)
[![XGBoost](https://img.shields.io/badge/Classifier-XGBoost%20%2B%20TF--IDF-red)](https://xgboost.ai)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-ff4b4b)](https://streamlit.io)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688)](https://fastapi.tiangolo.com)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Seel1e/Mindforge/blob/main/notebooks/02_colab_training.ipynb)

---

## What is MindForge?

MindForge is an end-to-end AI system built around mental health. It combines a fine-tuned language model, a retrieval system that pulls in real psychology knowledge, and a text classifier that detects mental health conditions from what someone writes.

| Component | What it does | Technology |
|---|---|---|
| **Fine-tuned LLM** | Has mental health conversations, answers questions, gives evidence-based guidance | Mistral-7B + QLoRA (Unsloth) |
| **RAG** | Looks up relevant psychology knowledge before answering so the model doesn't make things up | LangChain + ChromaDB |
| **Condition Classifier** | Reads a statement and identifies signs of Anxiety, Depression, Stress, etc. | XGBoost + TF-IDF |

> **Disclaimer:** MindForge is a research and learning project. It is not a replacement for professional mental health care. If you or someone you know is struggling, please reach out to a licensed therapist or psychiatrist.

---

## Train it yourself (Google Colab, free)

No GPU? No problem. The training notebook runs on Google Colab's free T4 GPU and takes about 2 hours end-to-end.

**[Open training notebook in Colab](https://colab.research.google.com/github/Seel1e/Mindforge/blob/main/notebooks/02_colab_training.ipynb)**

The notebook walks through every step — installing dependencies, preprocessing the datasets, training the condition classifier, and fine-tuning the LLM — with explanations along the way.

---

## Project Structure

```
MindForge/
│
├── README.md                      ← You are here
├── config.yaml                    ← All hyperparameters in one place
├── requirements.txt               ← Python dependencies
├── .env.example                   ← Template for API keys
├── run.py                         ← One-command CLI runner
│
├── src/
│   ├── config.py
│   ├── preprocessing/
│   │   ├── clean_text.py          ← Text cleaning
│   │   ├── prepare_finetune.py    ← Converts datasets → JSONL for LLM training
│   │   └── prepare_structured.py ← Prepares data for the classifier
│   ├── training/
│   │   ├── finetune_llm.py        ← QLoRA fine-tuning with Unsloth + SFTTrainer
│   │   └── train_risk_model.py    ← XGBoost + TF-IDF classifier training
│   ├── rag/
│   │   ├── build_index.py         ← Builds ChromaDB vector store
│   │   └── retriever.py           ← Retrieves context for a query
│   ├── evaluation/
│   │   └── evaluate.py            ← ROUGE, BERTScore, safety checks
│   └── inference/
│       └── pipeline.py            ← Unified inference (LLM + RAG + Classifier)
│
├── app/
│   ├── app.py                     ← Streamlit chat interface
│   └── api.py                     ← FastAPI REST API
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   └── 02_colab_training.ipynb    ← Full training pipeline for Google Colab
│
├── data/
│   ├── raw/                       ← Original datasets (not tracked in git)
│   └── processed/                 ← Generated during preprocessing
│
└── models/                        ← Saved models (not tracked in git)
    ├── mindforge-lora/            ← LoRA adapters after fine-tuning
    ├── risk_predictor.pkl         ← Trained condition classifier
    └── plots/                     ← Confusion matrix, SHAP plots
```

---

## Datasets

Five datasets, each covering a different angle of mental health AI:

| Dataset | Size | Used for |
|---|---|---|
| `Alpie-core_medical_psychology_dataset.json` | ~1.2 GB | LLM fine-tuning — psychology Q&A with chain-of-thought reasoning |
| `cleanData.csv` | ~30 MB | LLM fine-tuning + condition classifier — labelled mental health statements |
| `Combined Data.csv` | ~31 MB | Condition classifier — more labelled statements |
| `mental_health_dataset.csv` | ~595 KB | LLM fine-tuning — real therapist conversation transcripts |
| `train.csv` | ~4.5 MB | LLM fine-tuning — additional therapy Q&A |

Datasets are not included in the repo (too large). Store them locally in `data/raw/` or on Google Drive if using Colab.

---

## How It Works

### Fine-tuning

Mistral-7B already understands language — it's read a huge chunk of the internet. Fine-tuning teaches it to specifically understand mental health conversations, respond with empathy, and follow the right tone.

The problem is that updating all 7 billion parameters is expensive (needs ~80 GB of GPU memory). QLoRA solves this two ways:

1. **4-bit quantisation** — Compresses the model from 32-bit to 4-bit numbers, dropping memory from ~28 GB to ~4 GB
2. **LoRA** — Freezes the original model and trains tiny adapter matrices (only ~0.1% of parameters) that modify its behaviour

Together, this fits on a free Google Colab T4 GPU.

### RAG (Retrieval-Augmented Generation)

Even after fine-tuning, the LLM can sometimes make things up. RAG gives it a reference book to look up before answering:

1. All psychology knowledge is split into chunks and stored as vectors in ChromaDB
2. When a user asks something, their question is also converted to a vector
3. The 5 most similar chunks are retrieved and included in the prompt
4. The LLM answers using both its training and the retrieved facts

### Condition Classifier

A TF-IDF + XGBoost pipeline that reads a text statement and predicts which mental health condition it shows signs of: Anxiety, Bipolar, Depression, Normal, Personality Disorder, Stress, or Suicidal ideation.

- **TF-IDF** converts the text into a numerical feature vector (word frequencies, weighted by how distinctive each word is)
- **XGBoost** classifies that vector into one of the 7 categories
- Achieved **~78% test accuracy** across all 7 classes on 58,000 examples

---

## Architecture

```
┌─────────────────────────────────────────────────┐
│                   USER INPUT                    │
│  "I've been anxious and can't sleep"            │
└──────────────────┬──────────────────────────────┘
                   │
        ┌──────────┼──────────┐
        │          │          │
        ▼          ▼          ▼
 ┌──────────┐ ┌─────────┐ ┌──────────────┐
 │Condition │ │   RAG   │ │ LLM Prompt   │
 │Classifier│ │Retriever│ │ Builder      │
 └────┬─────┘ └────┬────┘ └──────┬───────┘
      │             │             │
   "Anxiety"   [5 relevant    user message
                psychology
                chunks]
        └──────────┴─────────────┘
                   │
                   ▼
      ┌────────────────────────┐
      │  Fine-tuned Mistral-7B │
      │  (with LoRA adapters)  │
      └────────────┬───────────┘
                   │
                   ▼
      ┌────────────────────────┐
      │       RESPONSE         │
      │  Empathetic, grounded  │
      │  mental health reply   │
      └────────────────────────┘
```

---

## Setup (Local)

### Prerequisites
- Python 3.10+
- NVIDIA GPU with 8+ GB VRAM for LLM training (CPU is fine for everything else)

### Install

```bash
# 1. Create virtual environment
python -m venv venv
venv\Scripts\activate   # Windows
# source venv/bin/activate  # Mac/Linux

# 2. Install PyTorch (get the right command for your system at pytorch.org)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 3. Install Unsloth
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

# 4. Install everything else
pip install -r requirements.txt

# 5. Set up API keys (optional)
cp .env.example .env
# Edit .env — HF_TOKEN and WANDB_API_KEY are optional
```

### Run the pipeline

```bash
python run.py preprocess    # Step 1: clean + format all datasets
python run.py build-index   # Step 2: build ChromaDB vector store
python run.py train-risk    # Step 3: train condition classifier
python run.py train-llm     # Step 4: fine-tune the LLM (needs GPU)
python run.py evaluate      # Step 5: evaluate both models
```

Or all at once:
```bash
python run.py all-steps
```

---

## Running the App

```bash
# Streamlit chat interface
streamlit run app/app.py
# Opens at http://localhost:8501

# FastAPI REST API
uvicorn app.api:app --host 0.0.0.0 --port 8000
# Swagger docs at http://localhost:8000/docs
```

---

## Results

### Condition Classifier (XGBoost + TF-IDF)
- **Test Accuracy:** 77.95% across 7 classes
- **Best classes:** Normal (0.90 F1), Bipolar (0.83 F1), Anxiety (0.82 F1)
- Trained on 58,000+ real mental health statements

### Fine-tuned LLM (Mistral-7B + QLoRA)
- Trained on ~14,000 examples (psychology Q&A, therapy transcripts, statement analysis)
- 1 epoch on free Colab T4 GPU (~1.5 hours)
- Responds empathetically to mental health questions with evidence-based suggestions

---

## Hardware

| Task | What you need |
|---|---|
| Preprocessing, RAG index, classifier training | Any modern CPU, 8 GB RAM |
| LLM fine-tuning | GPU with 8+ GB VRAM |
| Free cloud option | [Google Colab T4](https://colab.research.google.com) (free, 15 GB VRAM) |

---

## Future Plans

- Multi-turn conversation memory
- RLHF to align responses with human preferences
- Multilingual support
- Voice interface using Whisper
- Uncertainty quantification on risk predictions

---

## Acknowledgements

- **Base Model:** [Mistral AI](https://mistral.ai)
- **Fast Fine-tuning:** [Unsloth](https://github.com/unslothai/unsloth)
- **RAG Framework:** [LangChain](https://langchain.com)
- **Vector Store:** [ChromaDB](https://www.trychroma.com)
- **Datasets:** Kaggle community contributors

---

<div align="center">

*If you're going through something difficult, please reach out to someone who can help.*

**988 Suicide & Crisis Lifeline — call or text 988 (US)**

</div>

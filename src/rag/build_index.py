"""
src/rag/build_index.py
───────────────────────
Builds a ChromaDB vector store (a searchable database of text chunks)
from the psychology JSON and therapy Q&A datasets.

Plain English:
  RAG = Retrieval-Augmented Generation.
  Instead of memorising everything during training, the LLM can
  "look things up" at inference time from a fast vector database.

  Step 1 (this file) — BUILD the database:
    - Read all psychology articles and therapy Q&As.
    - Split them into small chunks (512 tokens each).
    - Convert each chunk into a vector (list of numbers) using a
      sentence-transformer embedding model.
    - Store all vectors + text in ChromaDB on disk.

  Step 2 (retriever.py) — USE the database:
    - User asks a question.
    - Convert question to a vector.
    - Find the 5 most similar chunks.
    - Include them as context when prompting the LLM.

Usage:
  python -m src.rag.build_index
"""

import json
import pandas as pd
from pathlib import Path
from typing import Iterator
from loguru import logger
from tqdm import tqdm

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from src.config import cfg
from src.preprocessing.clean_text import clean


def _iter_psychology_chunks(path: Path) -> Iterator[tuple[str, dict]]:
    """Yield (text_chunk, metadata) from the psychology JSON — streamed to save RAM."""
    logger.info(f"Loading psychology JSON for RAG indexing …")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=cfg.rag["chunk_size"],
        chunk_overlap=cfg.rag["chunk_overlap"],
        separators=["\n\n", "\n", ". ", " "],
    )

    try:
        import ijson
        # Stream-parse the JSON so the whole file is never in RAM at once
        with open(path, "rb") as f:
            # Try as top-level array first
            try:
                records = ijson.items(f, "item")
                for i, rec in enumerate(records):
                    prompt = clean(str(rec.get("prompt", "")))
                    response = clean(str(rec.get("response", "")))
                    cot = clean(str(rec.get("complex_cot", "")))
                    full_text = f"Q: {prompt}\n\nA: {response}"
                    if cot:
                        full_text += f"\n\nReasoning: {cot[:500]}"
                    for chunk in splitter.split_text(full_text):
                        yield chunk, {"source": "psychology_json", "record_id": str(i), "prompt_preview": prompt[:80]}
                return
            except Exception:
                pass
    except ImportError:
        pass

    # Fallback: load in chunks by reading line by line if newline-delimited JSON
    # or load full file if small enough
    logger.warning("ijson not available, loading JSON fully (may use more RAM) …")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    records = data if isinstance(data, list) else next(
        (v for v in data.values() if isinstance(v, list)), []
    )
    for i, rec in enumerate(tqdm(records, desc="Psychology JSON")):
        prompt = clean(str(rec.get("prompt", "")))
        response = clean(str(rec.get("response", "")))
        cot = clean(str(rec.get("complex_cot", "")))
        full_text = f"Q: {prompt}\n\nA: {response}"
        if cot:
            full_text += f"\n\nReasoning: {cot[:500]}"
        for chunk in splitter.split_text(full_text):
            yield chunk, {"source": "psychology_json", "record_id": str(i), "prompt_preview": prompt[:80]}


def _iter_therapy_chunks(path: Path) -> Iterator[tuple[str, dict]]:
    """Yield (text_chunk, metadata) from the therapy Q&A CSV."""
    logger.info("Loading therapy Q&A for RAG indexing …")
    df = pd.read_csv(path)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=cfg.rag["chunk_size"],
        chunk_overlap=cfg.rag["chunk_overlap"],
    )

    for i, row in tqdm(df.iterrows(), total=len(df), desc="Therapy Q&A"):
        context = clean(str(row.iloc[0]))
        response = clean(str(row.iloc[1]))
        full_text = f"User concern: {context}\n\nCounselor response: {response}"

        chunks = splitter.split_text(full_text)
        for chunk in chunks:
            yield chunk, {
                "source": "therapy_qa",
                "record_id": str(i),
            }


def build_vector_store(
    max_psych: int = 5_000,
    max_therapy: int = 3_000,
    batch_size: int = 256,
) -> Chroma:
    """
    Build and persist the ChromaDB vector store.

    Args:
        max_psych:   Max chunks from psychology JSON.
        max_therapy: Max chunks from therapy Q&A.
        batch_size:  How many documents to embed at once.

    Returns:
        The Chroma vector store object.
    """
    vector_store_dir = str(cfg.DATA_PROCESSED / "chroma_db")

    # ── Embedding model ───────────────────────────────────────
    logger.info(f"Loading embedding model: {cfg.rag['embedding_model']} …")
    embeddings = HuggingFaceEmbeddings(
        model_name=cfg.rag["embedding_model"],
        model_kwargs={"device": "cpu"},          # swap "cpu" for "cuda" if you have GPU
        encode_kwargs={"normalize_embeddings": True},
    )

    # ── Collect all text chunks ───────────────────────────────
    texts, metadatas = [], []

    psych_count = 0
    for text, meta in _iter_psychology_chunks(cfg.DS_PSYCHOLOGY_JSON):
        if max_psych and psych_count >= max_psych:
            break
        texts.append(text)
        metadatas.append(meta)
        psych_count += 1

    therapy_count = 0
    for text, meta in _iter_therapy_chunks(cfg.DS_THERAPY_QA):
        if max_therapy and therapy_count >= max_therapy:
            break
        texts.append(text)
        metadatas.append(meta)
        therapy_count += 1

    logger.info(f"Total chunks to index: {len(texts):,}  "
                f"(psychology: {psych_count:,}, therapy: {therapy_count:,})")

    # ── Build in batches (avoids OOM on large corpora) ────────
    logger.info("Building ChromaDB vector store (this may take several minutes) …")

    # First batch creates the store
    vectordb = Chroma.from_texts(
        texts=texts[:batch_size],
        metadatas=metadatas[:batch_size],
        embedding=embeddings,
        persist_directory=vector_store_dir,
        collection_name=cfg.rag["collection_name"],
    )
    vectordb.persist()

    # Remaining batches
    for start in tqdm(range(batch_size, len(texts), batch_size), desc="Indexing batches"):
        end = min(start + batch_size, len(texts))
        vectordb.add_texts(
            texts=texts[start:end],
            metadatas=metadatas[start:end],
        )
        vectordb.persist()

    logger.success(f"Vector store saved → {vector_store_dir}")
    logger.success(f"Total indexed: {vectordb._collection.count():,} chunks")

    return vectordb


if __name__ == "__main__":
    build_vector_store()

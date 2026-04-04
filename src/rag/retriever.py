"""
src/rag/retriever.py
─────────────────────
Loads the pre-built ChromaDB vector store and retrieves relevant
context for a given user query.

Plain English:
  When a user asks "What is CBT therapy?",
  this file:
  1. Converts the question into a vector (embedding).
  2. Searches the vector store for the 5 most similar text chunks.
  3. Returns those chunks as a string that the LLM can read.

Usage (from Python):
  from src.rag.retriever import MindForgeRetriever
  retriever = MindForgeRetriever()
  context = retriever.get_context("What are CBT techniques for anxiety?")
"""

from pathlib import Path
from loguru import logger
from typing import Optional

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from src.config import cfg


class MindForgeRetriever:
    """
    Thin wrapper around ChromaDB that provides a simple
    .get_context(query) → str interface.
    """

    def __init__(self, vector_store_dir: Optional[str] = None):
        self._store: Optional[Chroma] = None
        self._embeddings: Optional[HuggingFaceEmbeddings] = None
        self._store_dir = vector_store_dir or str(cfg.DATA_PROCESSED / "chroma_db")

    def _ensure_loaded(self):
        """Lazy-load the vector store on first use."""
        if self._store is not None:
            return

        store_path = Path(self._store_dir)
        if not store_path.exists() or not any(store_path.iterdir()):
            raise FileNotFoundError(
                f"Vector store not found at {self._store_dir}.\n"
                "Run: python -m src.rag.build_index"
            )

        logger.info(f"Loading vector store from {self._store_dir} …")
        self._embeddings = HuggingFaceEmbeddings(
            model_name=cfg.rag["embedding_model"],
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        self._store = Chroma(
            persist_directory=self._store_dir,
            embedding_function=self._embeddings,
            collection_name=cfg.rag["collection_name"],
        )
        count = self._store._collection.count()
        logger.info(f"Vector store loaded: {count:,} chunks")

    def get_context(
        self,
        query: str,
        top_k: Optional[int] = None,
        min_score: Optional[float] = None,
    ) -> str:
        """
        Retrieve the most relevant context for a query.

        Args:
            query:     The user's question or statement.
            top_k:     Number of chunks to retrieve (default from config).
            min_score: Minimum similarity score filter (0–1, higher = stricter).

        Returns:
            A formatted string of retrieved context, ready to inject into a prompt.
            Returns "" if nothing relevant found.
        """
        self._ensure_loaded()

        k = top_k or cfg.rag["top_k"]
        threshold = min_score or cfg.rag["similarity_threshold"]

        # similarity_search_with_score returns (Document, score) pairs
        results = self._store.similarity_search_with_score(query, k=k)

        # Filter by similarity score (ChromaDB uses L2 distance: lower = better)
        # We convert to cosine similarity approximation
        filtered = [
            (doc, score) for doc, score in results
            if score <= (1 - threshold)    # L2 distance threshold
        ]

        if not filtered:
            return ""

        context_parts = []
        for i, (doc, score) in enumerate(filtered, 1):
            source = doc.metadata.get("source", "unknown")
            context_parts.append(
                f"[Context {i} | source: {source}]\n{doc.page_content}"
            )

        return "\n\n---\n\n".join(context_parts)

    def get_documents(self, query: str, top_k: Optional[int] = None) -> list:
        """Return raw LangChain Document objects."""
        self._ensure_loaded()
        k = top_k or cfg.rag["top_k"]
        return self._store.similarity_search(query, k=k)


# Module-level singleton for reuse across the app
_retriever: Optional[MindForgeRetriever] = None


def get_retriever() -> MindForgeRetriever:
    """Return (or create) the module-level retriever singleton."""
    global _retriever
    if _retriever is None:
        _retriever = MindForgeRetriever()
    return _retriever

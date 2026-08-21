# ai_assistant/retriever.py
"""
Semantic retriever backed by the FAISS index built by indexer.py.

Falls back gracefully to an empty result list when the index is not
available (sentence-transformers / faiss not installed, or not yet built).
"""
from __future__ import annotations

import logging
import os
from typing import List, Optional

from .prompting import RetrievedChunk

logger = logging.getLogger(__name__)


class KnowledgeRetriever:
    """
    Wraps a FAISS index for semantic top-k retrieval with lightweight
    re-ranking.

    Usage:
        retriever = KnowledgeRetriever(index_path="~/.setiastro/ai_index")
        retriever.load()
        chunks = retriever.retrieve("How do I stack flats?", top_k=5)
    """

    _MODEL_NAME = "all-MiniLM-L6-v2"

    def __init__(self, index_path: str):
        self.index_path = os.path.expanduser(index_path)
        self._index = None
        self._chunks = []
        self._model = None
        self._ready = False

    def load(self) -> bool:
        """
        Load the FAISS index from disk and warm up the embedding model.
        Returns True on success.
        """
        try:
            import faiss
            from sentence_transformers import SentenceTransformer
            from .indexer import KnowledgeIndexer

            indexer = KnowledgeIndexer.__new__(KnowledgeIndexer)
            indexer.repo_root = ""
            indexer.index_path = self.index_path
            self._index, self._chunks = indexer.load()

            if self._index is None or not self._chunks:
                logger.info("Retriever: no index found at %s", self.index_path)
                return False

            self._model = SentenceTransformer(self._MODEL_NAME)
            self._ready = True
            logger.info("Retriever ready (%d chunks)", len(self._chunks))
            return True

        except ImportError as exc:
            logger.info("Retriever: optional deps not available (%s)", exc)
            return False
        except Exception as exc:
            logger.error("Retriever.load failed: %s", exc)
            return False

    def is_ready(self) -> bool:
        return self._ready and self._index is not None

    def retrieve(self, query: str, top_k: int = 5) -> List[RetrievedChunk]:
        """
        Return the top_k most relevant chunks for *query*.

        Steps:
        1. Embed the query with the same model used during indexing.
        2. FAISS inner-product search (normalised vectors = cosine similarity).
        3. Re-rank: boost chunks whose source_file token appears in the query.
        """
        if not self.is_ready():
            return []

        try:
            import numpy as np
            import faiss

            q_vec = self._model.encode([query], show_progress_bar=False)
            q_np = np.array(q_vec, dtype="float32")
            faiss.normalize_L2(q_np)

            fetch_k = min(top_k * 3, len(self._chunks))
            scores, indices = self._index.search(q_np, fetch_k)

            candidates: List[RetrievedChunk] = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < 0 or idx >= len(self._chunks):
                    continue
                chunk = self._chunks[idx]
                candidates.append(
                    RetrievedChunk(
                        text=chunk.text,
                        source_file=chunk.source_file,
                        score=float(score),
                    )
                )

            # Lightweight re-ranking: boost if source file stem appears in query
            query_lower = query.lower()
            for c in candidates:
                stem = os.path.splitext(os.path.basename(c.source_file))[0].lower()
                if stem in query_lower:
                    c.score += 0.2

            candidates.sort(key=lambda c: c.score, reverse=True)
            return candidates[:top_k]

        except Exception as exc:
            logger.error("Retriever.retrieve failed: %s", exc)
            return []

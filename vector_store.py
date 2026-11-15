"""
vector_store.py – Semantic vector store with pure-Python cosine similarity.
Falls back to BM25 automatically if the HuggingFace API is unavailable.
Zero external dependencies beyond Python 3.14 stdlib.
"""

import math
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from embeddings import cosine_similarity, embed_texts


# ══════════════════════════════════════════════════════════════
#  BM25 — kept as fallback when HF API is unreachable
# ══════════════════════════════════════════════════════════════
class _BM25:
    K1, B = 1.5, 0.75

    def __init__(self, chunks: List[Dict[str, Any]]) -> None:
        self._chunks = chunks
        self._docs   = [c["text"].lower().split() for c in chunks]
        self._dl     = [len(d) for d in self._docs]
        self._n      = len(chunks)
        self._avgdl  = sum(self._dl) / max(self._n, 1)
        self._df: Dict[str, int] = defaultdict(int)
        for doc in self._docs:
            for t in set(doc):
                self._df[t] += 1

    def _idf(self, term: str) -> float:
        df = self._df.get(term, 0)
        return math.log((self._n - df + 0.5) / (df + 0.5) + 1.0)

    def score(self, query_tokens: List[str], idx: int) -> float:
        tf  = defaultdict(int)
        for t in self._docs[idx]:
            tf[t] += 1
        norm = 1 - self.B + self.B * self._dl[idx] / self._avgdl
        return sum(
            self._idf(t) * tf.get(t, 0) * (self.K1 + 1)
            / (tf.get(t, 0) + self.K1 * norm)
            for t in query_tokens
        )

    def retrieve(self, query: str, k: int = 4) -> List[Tuple[Dict, float]]:
        tokens = query.lower().split()
        scored = [(self._chunks[i], self.score(tokens, i)) for i in range(self._n)]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]

    def add(self, new_chunks: List[Dict[str, Any]]) -> None:
        new_docs = [c["text"].lower().split() for c in new_chunks]
        self._chunks.extend(new_chunks)
        self._docs.extend(new_docs)
        self._dl.extend(len(d) for d in new_docs)
        self._n    = len(self._chunks)
        self._avgdl = sum(self._dl) / max(self._n, 1)
        for doc in new_docs:
            for t in set(doc):
                self._df[t] += 1


# ══════════════════════════════════════════════════════════════
#  SemanticStore — cosine similarity over HF embeddings
# ══════════════════════════════════════════════════════════════
class SemanticStore:
    """
    In-memory semantic vector store.
    Uses HuggingFace sentence embeddings + cosine similarity for retrieval.
    Automatically falls back to BM25 if the HF API is unavailable.
    """

    def __init__(
        self,
        chunks: List[Dict[str, Any]],
        embeddings: Optional[List[List[float]]] = None,
    ) -> None:
        self.chunks     = chunks
        self.embeddings = embeddings          # None → BM25 fallback active
        self._bm25      = _BM25(chunks)       # always built for fallback

    @property
    def mode(self) -> str:
        return "semantic" if self.embeddings is not None else "bm25-fallback"

    def retrieve(
        self,
        query: str,
        k: int = 4,
        query_embedding: Optional[List[float]] = None,
    ) -> List[Tuple[Dict[str, Any], float]]:
        """Return top-k chunks with relevance scores."""
        if self.embeddings and query_embedding:
            scored = [
                (chunk, cosine_similarity(query_embedding, emb))
                for chunk, emb in zip(self.chunks, self.embeddings)
            ]
            scored.sort(key=lambda x: x[1], reverse=True)
            return scored[:k]

        # BM25 fallback
        return self._bm25.retrieve(query, k)

    def add_documents(
        self,
        new_chunks: List[Dict[str, Any]],
        new_embeddings: Optional[List[List[float]]] = None,
    ) -> None:
        """Incrementally add more documents to the store."""
        self.chunks.extend(new_chunks)
        self._bm25.add(new_chunks)

        if self.embeddings is not None:
            if new_embeddings is None:
                try:
                    new_texts = [c["text"] for c in new_chunks]
                    new_embeddings = embed_texts(new_texts)
                except Exception as exc:
                    print(f"[vector_store] WARNING: Failed to embed new chunks ({exc}). Falling back to BM25.")
                    self.embeddings = None
                    return
            self.embeddings.extend(new_embeddings)


# ══════════════════════════════════════════════════════════════
#  Public API
# ══════════════════════════════════════════════════════════════
def build_vector_store(chunks: List[Dict[str, Any]]) -> SemanticStore:
    """
    Build a SemanticStore from text chunks.
    Calls the HF API to generate embeddings; falls back to BM25 on failure.
    """
    if not chunks:
        raise ValueError("No chunks provided to build_vector_store.")

    texts = [c["text"] for c in chunks]
    try:
        embeddings = embed_texts(texts)
        return SemanticStore(chunks, embeddings)
    except Exception as exc:
        # HF API unavailable — graceful BM25 fallback
        print(f"[vector_store] WARNING: HF embeddings failed ({exc}). Using BM25 fallback.")
        return SemanticStore(chunks, embeddings=None)


def retrieve_relevant_chunks(
    store: SemanticStore,
    query: str,
    k: int = 4,
) -> List[Tuple[Dict[str, Any], float]]:
    """
    Retrieve top-k chunks for a query.
    Embeds the query via HF API (if store is in semantic mode) then computes similarity.
    """
    query_embedding: Optional[List[float]] = None

    if store.embeddings is not None:
        try:
            query_embedding = embed_texts([query])[0]
        except Exception as exc:
            print(f"[vector_store] WARNING: query embedding failed ({exc}). Using BM25.")

    return store.retrieve(query, k=k, query_embedding=query_embedding)

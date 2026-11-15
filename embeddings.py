"""
embeddings.py – Semantic embeddings via HuggingFace Inference API.
Uses only Python's built-in urllib — no numpy, no torch, no extra installs.
Works on Python 3.14+.

Optional: set HF_TOKEN in .env for higher rate limits (free account).
Without a token: ~200 requests/hour (enough for normal use).
"""

import json
import math
import os
import time
import urllib.request
import urllib.error
from typing import List

MODEL_ID  = "sentence-transformers/all-MiniLM-L6-v2"
HF_URL    = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{MODEL_ID}"
BATCH_SZ  = 32   # texts per API call
TIMEOUT   = 90   # seconds
MAX_RETRY = 3


# ── Pure-Python cosine similarity ─────────────────────────────────────
def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Compute cosine similarity between two vectors. Pure Python, no numpy."""
    dot   = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


# ── HF API helpers ────────────────────────────────────────────────────
def _headers() -> dict:
    token = os.getenv("HF_TOKEN", "").strip()
    h = {"Content-Type": "application/json"}
    if token:
        h["Authorization"] = f"Bearer {token}"
    return h


def _mean_pool(token_embeddings: List[List[float]]) -> List[float]:
    """Average token-level embeddings → sentence embedding."""
    n   = len(token_embeddings)
    dim = len(token_embeddings[0])
    result = [0.0] * dim
    for emb in token_embeddings:
        for i, v in enumerate(emb):
            result[i] += v
    return [v / n for v in result]


def _parse_response(raw) -> List[List[float]]:
    """
    HF API returns embeddings in various shapes depending on model/version.
    Handle all three cases:
      - [float, ...]                → single embedding
      - [[float, ...], ...]         → list of sentence embeddings  ✓
      - [[[float,...], ...], ...]   → token-level, needs mean pool
    """
    if not raw:
        raise ValueError("Empty response from HuggingFace API.")

    # Single text response (bare list of floats)
    if isinstance(raw[0], float):
        return [raw]

    if isinstance(raw[0], list):
        # Sentence-level embeddings
        if isinstance(raw[0][0], float):
            return raw
        # Token-level embeddings → mean pool each doc
        if isinstance(raw[0][0], list):
            return [_mean_pool(doc_tokens) for doc_tokens in raw]

    raise ValueError(f"Unexpected embedding shape from HF API: {type(raw[0])}")


def _post_batch(texts: List[str]) -> List[List[float]]:
    """POST one batch to HF API with retry logic."""
    payload = json.dumps({
        "inputs": texts,
        "options": {"wait_for_model": True},
    }).encode("utf-8")

    headers = _headers()
    req = urllib.request.Request(HF_URL, data=payload, headers=headers, method="POST")

    last_err = None
    for attempt in range(MAX_RETRY):
        try:
            with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
                body = json.loads(resp.read().decode("utf-8"))
                return _parse_response(body)
        except urllib.error.HTTPError as exc:
            if exc.code == 503:
                # Model is loading on HF servers — wait then retry
                wait = 20 * (attempt + 1)
                time.sleep(wait)
                last_err = f"HF model loading (503), waited {wait}s"
            else:
                raise RuntimeError(f"HF API HTTP {exc.code}: {exc.read().decode()}") from exc
        except Exception as exc:
            last_err = str(exc)
            if attempt < MAX_RETRY - 1:
                time.sleep(5)

    raise RuntimeError(f"HF API failed after {MAX_RETRY} attempts. Last error: {last_err}")


def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Embed a list of texts via the HuggingFace Inference API.

    Args:
        texts: List of strings to embed.

    Returns:
        List of float vectors (one per input text), each length 384.

    Raises:
        RuntimeError: If the API fails after retries.
    """
    all_embeddings: List[List[float]] = []
    for i in range(0, len(texts), BATCH_SZ):
        batch = texts[i : i + BATCH_SZ]
        all_embeddings.extend(_post_batch(batch))
    return all_embeddings

"""
rag_pipeline.py – Hardened RAG pipeline with semantic retrieval + model auto-fallback.
Pure Python 3.14-compatible. No LangChain, no numpy, no torch.
"""

import os
import time
from typing import Any, Dict, List, Tuple

from groq import Groq

from vector_store import SemanticStore, retrieve_relevant_chunks
from prompts import SYSTEM_PROMPT

# Cached list of active models discovered from the user's Groq account
_cached_models: List[str] = []

NO_ANSWER_MSG = "I couldn't find this information in the uploaded documents."


def _build_context(results: List[Tuple[Dict[str, Any], float]]) -> str:
    parts = []
    for i, (chunk, _score) in enumerate(results, start=1):
        source = chunk.get("source", "Unknown")
        page   = chunk.get("page", "?")
        parts.append(f"[{i}] Source: {source} | Page: {page}\n{chunk['text']}")
    return "\n\n---\n\n".join(parts)


def get_groq_client() -> Groq:
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key or api_key == "your_groq_api_key_here":
        raise EnvironmentError("GROQ_API_KEY is not set. Add it to your .env file.")
    return Groq(api_key=api_key)


def get_candidate_models(client: Groq) -> List[str]:
    """Dynamically get active chat models from Groq API for this account."""
    global _cached_models
    if _cached_models:
        return _cached_models

    try:
        remote = client.models.list()
        valid = []
        for m in remote.data:
            mid = m.id.lower()
            # Skip whisper, guard, tts, embedding models
            if any(k in mid for k in ["whisper", "guard", "orpheus", "embed"]):
                continue
            valid.append(m.id)

        # Priority sorting: versatile/instant models first
        def score_model(name: str) -> int:
            n = name.lower()
            if "llama-3.3-70b-versatile" in n: return 1
            if "llama-3.1-8b-instant" in n:    return 2
            if "70b" in n:                     return 3
            if "8b" in n:                      return 4
            if "gemma" in n:                   return 5
            return 10

        valid.sort(key=score_model)
        if valid:
            print(f"[Groq] Active models found: {valid}")
            _cached_models = valid
            return _cached_models
    except Exception as exc:
        print(f"[Groq] Could not query models.list ({exc}), using defaults.")

    _cached_models = [
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
        "gemma2-9b-it",
    ]
    return _cached_models


def answer_question(store: SemanticStore, question: str) -> Dict[str, Any]:
    """
    Full RAG pipeline with timing data for visualization:
      1. Semantic retrieval (HF embeddings + cosine sim, or BM25 fallback)
      2. Context assembly
      3. Groq LLM with model auto-fallback
    """
    pipeline = []

    # ── Step 1: Retrieve ──────────────────────────────────────
    t0 = time.monotonic()
    results = retrieve_relevant_chunks(store, question, k=4)
    retrieval_ms = round((time.monotonic() - t0) * 1000)

    mode = store.mode
    pipeline.append({
        "step":   "Retrieval",
        "icon":   "🎯",
        "detail": f"{mode.upper()} · {len(results)} chunk(s) · {retrieval_ms} ms",
        "ms":     retrieval_ms,
    })

    # ── Step 2: Context ───────────────────────────────────────
    t0 = time.monotonic()
    context    = _build_context(results)
    context_ms = round((time.monotonic() - t0) * 1000)

    pipeline.append({
        "step":   "Context Assembly",
        "icon":   "📋",
        "detail": f"{len(context)} chars assembled · {context_ms} ms",
        "ms":     context_ms,
    })

    # ── Step 3: LLM with fallback ─────────────────────────────
    client    = get_groq_client()
    messages  = [
        {"role": "system", "content": SYSTEM_PROMPT.format(context=context)},
        {"role": "user",   "content": question},
    ]

    candidate_models = get_candidate_models(client)
    answer_text = ""
    model_used  = ""
    llm_ms      = 0
    errors      = []

    for model in candidate_models:
        try:
            t0 = time.monotonic()
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.2,
                max_tokens=1024,
            )
            llm_ms      = round((time.monotonic() - t0) * 1000)
            answer_text = response.choices[0].message.content.strip()
            model_used  = model
            break
        except Exception as exc:
            err_msg = f"{model}: {exc}"
            print(f"[Groq Error] {err_msg}")
            errors.append(err_msg)

    if not model_used:
        raise RuntimeError("All Groq models failed:\n" + "\n".join(errors[:3]))

    if not answer_text:
        answer_text = NO_ANSWER_MSG

    pipeline.append({
        "step":   "Groq LLM",
        "icon":   "🤖",
        "detail": f"{model_used} · {llm_ms} ms",
        "ms":     llm_ms,
    })

    # ── Step 4: Format output ─────────────────────────────────
    seen    : set         = set()
    sources : List[Dict]  = []
    chunks  : List[Dict]  = []

    for chunk, score in results:
        src  = chunk.get("source", "")
        page = chunk.get("page", "")
        key  = (src, page)
        if key not in seen:
            seen.add(key)
            sources.append({"source": src, "page": page})

        preview = chunk["text"][:220].replace("\n", " ").strip()
        if len(chunk["text"]) > 220:
            preview += "…"

        chunks.append({
            "text":   preview,
            "source": src,
            "page":   page,
            "score":  round(float(score), 4),
        })

    return {
        "answer":   answer_text,
        "sources":  sources,
        "chunks":   chunks,
        "pipeline": pipeline,
        "model":    model_used,
        "mode":     mode,
    }

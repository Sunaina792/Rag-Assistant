"""
app.py – DocuMind AI FastAPI backend. Python 3.14-compatible.
"""

import os
from contextlib import asynccontextmanager
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

load_dotenv()

from document_loader import load_document
from text_splitter   import split_documents
from vector_store    import build_vector_store, SemanticStore
from rag_pipeline    import answer_question


# ---------------------------------------------------------------------------
# In-memory state
# ---------------------------------------------------------------------------
class AppState:
    def __init__(self):
        self.store: SemanticStore | None = None
        self.indexed_files: list[str] = []
        self.total_chunks: int = 0
        # Keep per-file chunk counts for visualization
        self.file_chunks: dict[str, int] = {}

state = AppState()


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    yield

app = FastAPI(title="DocuMind AI", version="2.0.0", lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class ChatRequest(BaseModel):
    question: str

class ChatResponse(BaseModel):
    answer: str
    sources: list[dict[str, Any]]
    chunks: list[dict[str, Any]]    # retrieved chunk previews + scores
    pipeline: list[dict[str, Any]]  # timing steps for visualization
    model: str
    mode: str = "semantic"

class StatusResponse(BaseModel):
    indexed_files: list[str]
    file_chunks: dict[str, int]
    total_chunks: int
    ready: bool


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", response_class=FileResponse)
async def serve_ui():
    return FileResponse("static/index.html")


@app.get("/status", response_model=StatusResponse)
async def get_status():
    return StatusResponse(
        indexed_files=state.indexed_files,
        file_chunks=state.file_chunks,
        total_chunks=state.total_chunks,
        ready=state.store is not None,
    )


@app.post("/upload")
async def upload_documents(files: list[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="No files provided.")

    new_chunks: list[dict] = []
    errors:     list[dict] = []
    new_names:  list[str]  = []
    file_chunk_counts: dict[str, int] = {}

    for upload in files:
        if upload.filename in state.indexed_files:
            continue
        try:
            raw    = await upload.read()
            pages  = load_document(raw, upload.filename)
            chunks = split_documents(pages)
            new_chunks.extend(chunks)
            new_names.append(upload.filename)
            file_chunk_counts[upload.filename] = len(chunks)
        except Exception as exc:
            errors.append({"file": upload.filename, "error": str(exc)})

    if not new_chunks and not errors:
        return JSONResponse({"message": "All files already indexed.", "new_chunks": 0})

    if new_chunks:
        try:
            if state.store is None:
                state.store = build_vector_store(new_chunks)
            else:
                state.store.add_documents(new_chunks)

            state.indexed_files.extend(new_names)
            state.total_chunks += len(new_chunks)
            state.file_chunks.update(file_chunk_counts)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Indexing error: {exc}")

    return JSONResponse({
        "message":       f"Indexed {len(new_names)} file(s).",
        "new_chunks":    len(new_chunks),
        "indexed_files": state.indexed_files,
        "file_chunks":   state.file_chunks,
        "errors":        errors,
    })


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    if state.store is None:
        raise HTTPException(status_code=400,
            detail="No documents indexed yet. Please upload files first.")

    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key or api_key == "your_groq_api_key_here":
        raise HTTPException(status_code=500,
            detail="GROQ_API_KEY is not set. Add it to your .env file.")

    try:
        result = answer_question(state.store, request.question)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return ChatResponse(
        answer=result["answer"],
        sources=result["sources"],
        chunks=result["chunks"],
        pipeline=result["pipeline"],
        model=result["model"],
        mode=result.get("mode", "semantic"),
    )


@app.post("/reset")
async def reset():
    state.store         = None
    state.indexed_files = []
    state.total_chunks  = 0
    state.file_chunks   = {}
    return JSONResponse({"message": "Knowledge base reset successfully."})

"""App FastAPI — una sola app con módulos internos (ADR-008).

Módulos: voice (conversación/pipeline), rag, decision, summary. Los routers
exponen la API que consume la consola Next.js.
"""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import get_settings
from .db import init_db
from .routers import console, conversation, knowledge

settings = get_settings()

app = FastAPI(title="Clinical Voice Agent — Post-operatorio", version="0.1.0")

# La consola Next.js corre en otro origen en desarrollo.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(conversation.router)
app.include_router(knowledge.router)
app.include_router(console.router)


@app.on_event("startup")
def _startup() -> None:
    init_db()


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "llm_provider": settings.llm_provider,
        "embedding_model": settings.embedding_model,
    }

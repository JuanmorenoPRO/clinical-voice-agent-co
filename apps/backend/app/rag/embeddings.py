"""Embeddings con `bge-m3` servido por el mismo Ollama que corre el LLM (ADR-011).

Es el modelo multilingüe que se eligió, en su versión cuantizada de 1.2 GB: 1024
dimensiones y 8K de contexto. Servirlo desde Ollama en vez de fastembed evita
añadir `onnxruntime` y 2.27 GB de pesos aparte, y deja un solo runtime que
instalar y un solo proceso que calentar — que en la compuerta de 15 minutos se nota.

Medido en un M1 Pro: ~20 s la primera llamada (carga del modelo), después 55
chunks/s en lotes de 32 y ~40 ms una consulta suelta.

**El índice y las consultas tienen que usar el mismo modelo.** `scripts/build_index.py`
escribe el modelo y la dimensión en `manifest.json`, y `scripts/fetch_index.py` los
verifica al descargar: un índice construido con otro modelo se rechaza en vez de
devolver citas silenciosamente erróneas.
"""
from __future__ import annotations

import logging

import ollama

from ..config import get_settings

log = logging.getLogger(__name__)

# Misma cabecera de contexto que usa scripts/build_index.py al indexar. Sube
# bastante la recuperación cross-lingüe, porque medio corpus está en inglés y las
# consultas llegan en español.
_PREFIJO_DOC = "[{procedure}] {filename} — p.{page}\n{text}"


def _client() -> ollama.Client:
    return ollama.Client(host=get_settings().ollama_host)


def embed_documents(texts: list[str]) -> list[list[float]]:
    return _client().embed(model=get_settings().embedding_model, input=texts)["embeddings"]


def embed_query(text: str) -> list[float]:
    return _client().embed(model=get_settings().embedding_model, input=[text])["embeddings"][0]


def document_text(*, procedure: str | None, filename: str, page: int, text: str) -> str:
    """Texto tal como se embebe: con la cabecera de contexto."""
    return _PREFIJO_DOC.format(
        procedure=procedure or "general", filename=filename, page=page, text=text
    )


def build_query(utterance: str, *, procedure: str | None = None,
                day_postop: int | None = None) -> str:
    """Enriquece la pregunta del paciente antes de embeberla.

    Nunca se consulta con la frase cruda. "¿Me puedo bañar?" sin contexto recupera
    cualquier cosa; con el procedimiento y el día postoperatorio delante recupera
    el documento del procedimiento correcto, y encima salva la barrera de idioma
    con los PDFs en inglés.
    """
    partes = [p for p in (procedure, f"día {day_postop} postoperatorio" if day_postop else None)
              if p]
    return f"{' · '.join(partes)}: {utterance}" if partes else utterance

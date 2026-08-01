"""Cliente de embeddings — Voyage AI (voyage-3, multilingüe).

Voyage se usa AHORA para construir la base de embeddings de prueba y así simular
Databricks/Delta Share antes de que llegue el dataset real (7 de agosto). Es
hosteado (no descarga ~2GB de modelo local) para no arriesgar el gate de arranque
≤15 min. ⏳ El modelo definitivo se confirma el 7 de agosto según el LLM
obligatorio (ADR-011); si cambia la dimensión, ajustar EMBEDDING_DIM y re-embeder.
"""
from __future__ import annotations

from ..config import get_settings

_settings = get_settings()


def _client():
    import voyageai

    return voyageai.Client(api_key=_settings.voyage_api_key)


def embed_documents(texts: list[str]) -> list[list[float]]:
    """Embeddings para chunks a indexar."""
    if not texts:
        return []
    result = _client().embed(texts, model=_settings.embedding_model, input_type="document")
    return result.embeddings


def embed_query(text: str) -> list[float]:
    """Embedding para una consulta del paciente."""
    result = _client().embed([text], model=_settings.embedding_model, input_type="query")
    return result.embeddings[0]

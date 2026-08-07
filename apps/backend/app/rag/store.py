"""Almacén vectorial: ChromaDB persistente.

Sustituye a PostgreSQL+pgvector (ADR-004 → ADR-014). El motivo no fue técnico sino
de compuerta: quitar el servidor de base de datos elimina Docker de la ruta crítica
del arranque, y Docker en macOS además rompe el UDP de WebRTC, que es la voz.

**Chroma y SQLite no son transaccionales entre sí**, y de eso depende la compuerta
G5 (subir un documento y que el agente lo use; borrarlo y que lo olvide). La
consistencia se garantiza con tres medidas, en este orden:

  1. Al borrar, primero se quitan los vectores de Chroma y después la fila de
     SQLite. Si el proceso muere en medio, quedan metadatos sin vectores —que no
     devuelven nada— y nunca vectores sin metadatos.
  2. Toda consulta filtra por los `document_id` activos leídos de SQLite, así que
     un vector huérfano **no puede servirse jamás**, aunque exista.
  3. `orphan_sweep()` al arrancar limpia lo que haya quedado a medias.

El orden importa: la falla que hay que evitar es que el agente siga citando un
documento que el jurado acaba de borrar delante suyo.
"""
from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

import chromadb
from chromadb.api.models.Collection import Collection

from ..config import get_settings

log = logging.getLogger(__name__)


@lru_cache
def collection() -> Collection:
    s = get_settings()
    client = chromadb.PersistentClient(path=s.chroma_dir)
    return client.get_or_create_collection(
        s.chroma_collection, metadata={"hnsw:space": "cosine"}
    )


def count() -> int:
    return collection().count()


def add_chunks(
    *,
    ids: list[str],
    documents: list[str],
    metadatas: list[dict[str, Any]],
    embeddings: list[list[float]],
    batch: int = 32,
) -> None:
    col = collection()
    for i in range(0, len(ids), batch):
        sl = slice(i, i + batch)
        col.add(ids=ids[sl], documents=documents[sl],
                metadatas=metadatas[sl], embeddings=embeddings[sl])


def delete_document(document_id: str) -> int:
    """Borra los vectores de un documento. Devuelve cuántos había.

    Se llama ANTES de borrar la fila de SQLite (ver el docstring del módulo).
    """
    col = collection()
    antes = col.get(where={"document_id": document_id}, include=[])
    n = len(antes.get("ids") or [])
    if n:
        col.delete(where={"document_id": document_id})
    return n


def query(
    *,
    embedding: list[float],
    n_results: int,
    allowed_document_ids: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Consulta por similitud, restringida a los documentos activos.

    `allowed_document_ids` no es una optimización, es la barrera de seguridad: un
    vector cuyo documento ya no está en SQLite no puede llegar a una respuesta.
    """
    where: dict[str, Any] | None = None
    if allowed_document_ids is not None:
        if not allowed_document_ids:
            return []
        where = {"document_id": {"$in": allowed_document_ids}}

    res = collection().query(
        query_embeddings=[embedding],
        n_results=n_results,
        where=where,
        include=["documents", "metadatas", "distances"],
    )
    if not res.get("ids") or not res["ids"][0]:
        return []

    return [
        {"id": i, "text": d, "meta": m, "distance": dist}
        for i, d, m, dist in zip(
            res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]
        )
    ]


def orphan_sweep(active_document_ids: set[str]) -> int:
    """Elimina vectores cuyo documento ya no existe en SQLite.

    Repara un borrado interrumpido. Devuelve cuántos vectores se limpiaron.
    """
    col = collection()
    todos = col.get(include=["metadatas"])
    ids, metas = todos.get("ids") or [], todos.get("metadatas") or []
    huerfanos = [
        i for i, m in zip(ids, metas)
        if (m or {}).get("document_id") not in active_document_ids
    ]
    if huerfanos:
        log.warning("orphan_sweep: eliminando %d vectores sin documento", len(huerfanos))
        for i in range(0, len(huerfanos), 500):
            col.delete(ids=huerfanos[i:i + 500])
    return len(huerfanos)

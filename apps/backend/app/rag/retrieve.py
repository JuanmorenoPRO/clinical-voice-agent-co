"""Recuperación RAG por similitud (pgvector) con umbral de confianza (ADR-005).

Se ejecuta SIEMPRE que el paciente formula una pregunta — pgvector es local y
cuesta milisegundos, sin clasificador previo de turno. Bajo el umbral de
confianza el agente declara "no tengo evidencia suficiente" y ofrece escalar.

Toda afirmación clínica sale de aquí, nunca del conocimiento interno del modelo
(RF-04). La respuesta incluye la cita completa (documento, página, chunk_id,
score) para la traza (RF-05).
"""
from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..config import get_settings
from ..models import Chunk, Document
from ..schemas import RagResult, Source
from .embeddings import embed_query

_settings = get_settings()

_NO_EVIDENCE = (
    "No tengo evidencia suficiente en los documentos para responder eso con "
    "seguridad. Si le preocupa, puedo escalar su caso con el personal de enfermería."
)


def _similarity(distance: float) -> float:
    """Convierte distancia coseno de pgvector (0..2) a confianza 0..1."""
    return max(0.0, 1.0 - distance / 2.0)


def retrieve(session: Session, query: str) -> RagResult:
    q_vec = embed_query(query)

    # cosine_distance de pgvector; menor distancia = más similar.
    stmt = (
        select(Chunk, Document.filename, Chunk.embedding.cosine_distance(q_vec).label("dist"))
        .join(Document, Document.id == Chunk.document_id)
        .order_by("dist")
        .limit(_settings.rag_top_k)
    )
    rows = session.execute(stmt).all()

    if not rows:
        return RagResult(answer=_NO_EVIDENCE, confidence=0.0, sources=[], has_evidence=False)

    sources: list[Source] = []
    for chunk, filename, dist in rows:
        sources.append(
            Source(
                document=filename,
                page=chunk.page,
                chunk_id=chunk.id,
                score=round(_similarity(dist), 4),
            )
        )

    top_confidence = sources[0].score
    if top_confidence < _settings.rag_min_confidence:
        return RagResult(
            answer=_NO_EVIDENCE, confidence=top_confidence, sources=sources, has_evidence=False
        )

    # El "answer" aquí es la evidencia recuperada; el LLM la usa para redactar
    # la respuesta final (nunca responde desde su conocimiento interno).
    evidence = "\n\n".join(chunk.text for chunk, _, _ in rows)
    return RagResult(
        answer=evidence, confidence=top_confidence, sources=sources, has_evidence=True
    )

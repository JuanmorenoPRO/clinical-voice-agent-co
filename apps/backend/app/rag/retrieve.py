"""Recuperación RAG por similitud (pgvector) con umbral de confianza (ADR-005).

Se ejecuta SIEMPRE que el paciente formula una pregunta — pgvector es local y
cuesta milisegundos, sin clasificador previo de turno. Bajo el umbral de
confianza el agente declara "no tengo evidencia suficiente" y ofrece escalar.

Toda afirmación clínica sale de aquí, nunca del conocimiento interno del modelo
(RF-04). La respuesta incluye la cita completa (documento, página, chunk_id,
score) para la traza (RF-05).
"""
from __future__ import annotations

import unicodedata

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


def _norm(s: str) -> str:
    """Minúsculas sin acentos, solo alfanumérico (para comparar procedimientos)."""
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    cleaned = "".join(c.lower() if c.isalnum() else " " for c in s)
    return " ".join(cleaned.split())


def _allowed_document_ids(session: Session, procedure: str) -> list[str]:
    """IDs de documentos válidos para el procedimiento del paciente.

    Incluye los documentos SIN procedimiento (NULL = conocimiento general) y los
    cuyo procedimiento coincide (comparación laxa sin acentos). Si el paciente
    tuvo una cesárea y no hay documento de cesárea, la lista queda vacía → el
    agente declara "sin evidencia" (grounding determinista).
    """
    pn = _norm(procedure)
    allowed: list[str] = []
    for doc_id, doc_proc in session.query(Document.id, Document.procedure).all():
        if doc_proc is None:
            allowed.append(doc_id)  # conocimiento general
            continue
        dn = _norm(doc_proc)
        if dn and pn and (dn in pn or pn in dn):
            allowed.append(doc_id)
    return allowed


def retrieve(session: Session, query: str, procedure: str | None = None) -> RagResult:
    q_vec = embed_query(query)

    # cosine_distance de pgvector; menor distancia = más similar.
    stmt = (
        select(Chunk, Document.filename, Chunk.embedding.cosine_distance(q_vec).label("dist"))
        .join(Document, Document.id == Chunk.document_id)
    )

    # Filtro por procedimiento del paciente (si se conoce). Reduce el falso
    # "grounding" cuando dos procedimientos comparten vocabulario post-operatorio.
    if procedure:
        allowed = _allowed_document_ids(session, procedure)
        if not allowed:
            return RagResult(answer=_NO_EVIDENCE, confidence=0.0, sources=[], has_evidence=False)
        stmt = stmt.where(Chunk.document_id.in_(allowed))

    stmt = stmt.order_by("dist").limit(_settings.rag_top_k)
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

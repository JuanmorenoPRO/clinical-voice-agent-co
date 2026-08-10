"""Recuperación RAG por similitud (ChromaDB) con umbral de confianza (ADR-005).

Se ejecuta cuando el paciente formula una pregunta clínica —la intención la decide
`app/nlu/intent.py`, no un clasificador aprendido—. Bajo el umbral de confianza el
agente declara "no tengo evidencia suficiente" y ofrece escalar.

Toda afirmación clínica sale de aquí, nunca del conocimiento interno del modelo
(RF-04). La respuesta incluye la cita completa (documento, página, chunk_id,
score) para la traza (RF-05).

La firma pública `retrieve(session, query, procedure)` no cambió al migrar de
pgvector a Chroma: lo que llama a este módulo no se enteró.
"""
from __future__ import annotations

import unicodedata
from collections import defaultdict

from sqlalchemy.orm import Session

from ..config import get_settings
from ..models import Document
from ..schemas import RagResult, Source
from . import store
from .embeddings import embed_query

_NO_EVIDENCE = (
    "No tengo evidencia suficiente en los documentos para responder eso con "
    "seguridad. Si le preocupa, puedo escalar su caso con el personal de enfermería."
)

# Máximo de fragmentos del mismo documento en una respuesta. Sin esto, un PDF
# largo copa el top-k con cuatro párrafos casi iguales y la cita, aunque correcta,
# no aporta más que uno solo.
_MAX_POR_DOCUMENTO = 2

def _similarity(distance: float) -> float:
    """Convierte distancia coseno (0..2) a confianza 0..1."""
    return max(0.0, 1.0 - distance / 2.0)


def _norm(s: str) -> str:
    """Minúsculas sin acentos, solo alfanumérico (para comparar procedimientos)."""
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    cleaned = "".join(c.lower() if c.isalnum() else " " for c in s)
    return " ".join(cleaned.split())


def _allowed_document_ids(session: Session, procedure: str | None) -> list[str]:
    """IDs de documentos que pueden responder a este paciente.

    Incluye los de conocimiento general (sin procedimiento) y los del procedimiento
    del paciente, comparando sin acentos. Los marcados `off_scope` quedan fuera
    siempre: son los PDFs de cáncer de cuello uterino que venían en la carpeta
    `breast_cancer/` y que citados como evidencia de una mastectomía producirían
    una afirmación falsa con fuente, que es lo peor que puede pasar aquí.

    Si el paciente tuvo un procedimiento sin documentos, la lista queda vacía y el
    agente declara "sin evidencia". Es grounding determinista, no una decisión del
    modelo.
    """
    pn = _norm(procedure) if procedure else ""
    allowed: list[str] = []
    for doc_id, doc_proc, off_scope in session.query(
        Document.id, Document.procedure, Document.off_scope
    ).filter(Document.status == "indexed"):
        if off_scope:
            continue
        if not doc_proc:
            allowed.append(doc_id)  # conocimiento general
            continue
        if not pn:
            continue
        dn = _norm(doc_proc)
        if dn and (dn in pn or pn in dn):
            allowed.append(doc_id)
    return allowed


def retrieve(
    session: Session,
    query: str,
    procedure: str | None = None,
    *,
    raw_question: str | None = None,
) -> RagResult:
    """Recupera evidencia para `query`.

    `query` va enriquecida con el procedimiento y el día postoperatorio, porque eso
    mejora mucho la recuperación cross-lingüe.

    Ojo: el umbral de similitud **no basta** para decidir si esto responde. Medido
    con `scripts/calibrate_rag.py`, preguntas ajenas al corpus puntúan más alto
    (0.868) que legítimas (0.795), porque todo el corpus es texto médico
    postoperatorio y el coseno mide cercanía temática. Quien decide si la evidencia
    responde es `OllamaAdapter.pregunta_es_del_dominio`, que se llama desde el
    orquestador. Aquí solo se recupera.
    """
    s = get_settings()
    allowed = _allowed_document_ids(session, procedure)
    if not allowed:
        return RagResult(answer=_NO_EVIDENCE, confidence=0.0, sources=[], has_evidence=False)

    # Se sobre-recupera y luego se filtra en código. Es un reordenamiento barato:
    # sin él, el top-k se llena de fragmentos casi idénticos del mismo PDF.
    hits = store.query(
        embedding=embed_query(query),
        n_results=s.rag_fetch_k,
        allowed_document_ids=allowed,
    )
    if not hits:
        return RagResult(answer=_NO_EVIDENCE, confidence=0.0, sources=[], has_evidence=False)

    seleccion, por_documento = [], defaultdict(int)
    for hit in hits:  # ya vienen ordenados por distancia ascendente
        doc_id = (hit["meta"] or {}).get("document_id", "")
        if por_documento[doc_id] >= _MAX_POR_DOCUMENTO:
            continue
        por_documento[doc_id] += 1
        seleccion.append(hit)
        if len(seleccion) >= s.rag_top_k:
            break

    sources = [
        Source(
            document=(h["meta"] or {}).get("filename", "?"),
            page=int((h["meta"] or {}).get("page", 0)),
            chunk_id=h["id"],
            score=round(_similarity(h["distance"]), 4),
        )
        for h in seleccion
    ]

    top_confidence = sources[0].score
    evidence = "\n\n".join(h["text"] for h in seleccion)

    if top_confidence < s.rag_min_confidence:
        return RagResult(
            answer=_NO_EVIDENCE, confidence=top_confidence, sources=sources, has_evidence=False
        )
    # El "answer" es la evidencia recuperada, no la respuesta al paciente: el LLM
    # la usa para redactar, y nunca responde desde su conocimiento interno.
    return RagResult(
        answer=evidence, confidence=top_confidence, sources=sources, has_evidence=True
    )

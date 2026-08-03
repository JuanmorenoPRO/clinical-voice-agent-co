"""Ingesta de conocimiento: parseo → chunking → embeddings → pgvector.

Alta en caliente (RF-06): sube un documento y queda disponible en la siguiente
consulta, sin reiniciar servicios.
Baja en caliente (RF-07, gate G5): borrar el documento elimina TODOS sus vectores
en la MISMA transacción SQL — el agente "olvida" de inmediato, garantizado por
ACID (ADR-004).

Soporta PDF (PyMuPDF, conserva nº de página para trazabilidad RF-05) y texto/
markdown (los archivos de ejemplo del repo, stand-in de los PDFs clínicos reales).
"""
from __future__ import annotations

import os
from dataclasses import dataclass

from sqlalchemy.orm import Session

from ..config import get_settings
from ..models import Chunk, Document
from .embeddings import embed_documents

_settings = get_settings()


@dataclass
class PageText:
    page: int
    text: str


def _load_pages(path: str) -> list[PageText]:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        import fitz  # PyMuPDF

        pages: list[PageText] = []
        with fitz.open(path) as doc:
            for i, page in enumerate(doc, start=1):
                pages.append(PageText(page=i, text=page.get_text("text")))
        return pages
    # texto / markdown: una "página" lógica.
    with open(path, encoding="utf-8") as fh:
        return [PageText(page=1, text=fh.read())]


def _chunk_page(page: PageText) -> list[PageText]:
    """Chunking con solapamiento, conservando el nº de página (metadato)."""
    size, overlap = _settings.chunk_size, _settings.chunk_overlap
    text = page.text.strip()
    if not text:
        return []
    chunks: list[PageText] = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(PageText(page=page.page, text=text[start:end].strip()))
        start = end - overlap
        if start <= 0:
            break
    return [c for c in chunks if c.text]


def load_and_chunk(path: str) -> list[PageText]:
    chunks: list[PageText] = []
    for page in _load_pages(path):
        chunks.extend(_chunk_page(page))
    return chunks


def ingest_file(
    session: Session,
    path: str,
    filename: str | None = None,
    procedure: str | None = None,
) -> Document:
    """Parsea, trocea, vectoriza (Voyage) e indexa en pgvector. Devuelve el Document.

    `procedure` etiqueta el documento para la recuperación filtrada por
    procedimiento (NULL = conocimiento general, válido para todos).
    """
    filename = filename or os.path.basename(path)
    doc = Document(filename=filename, procedure=procedure, status="indexing")
    session.add(doc)
    session.flush()  # obtener doc.id

    pieces = load_and_chunk(path)
    embeddings = embed_documents([p.text for p in pieces])

    for piece, vector in zip(pieces, embeddings):
        session.add(
            Chunk(document_id=doc.id, page=piece.page, text=piece.text, embedding=vector)
        )

    doc.n_chunks = len(pieces)
    doc.status = "indexed"
    session.commit()
    session.refresh(doc)
    return doc


def delete_document(session: Session, document_id: str) -> bool:
    """Baja en caliente (gate G5). CASCADE borra los chunks en la misma transacción."""
    doc = session.get(Document, document_id)
    if doc is None:
        return False
    session.delete(doc)
    session.commit()
    return True

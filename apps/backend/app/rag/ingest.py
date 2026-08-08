"""Ingesta y borrado de documentos en caliente — compuerta G5.

El jurado sube un PDF que el agente nunca vio, comprueba que lo usa y lo cita, lo
borra y comprueba que vuelve a declarar que no tiene evidencia. Todo desde la
consola y sin reiniciar nada.

La lógica de troceado es la misma que `scripts/build_index.py` usa para construir
el índice del corpus, y vive aquí para que no haya dos versiones: un documento
subido en caliente se trocea exactamente igual que uno del corpus original, o las
citas no serían comparables.
"""
from __future__ import annotations

import hashlib
import logging
import re
import unicodedata
from collections import Counter
from pathlib import Path

import pymupdf
from sqlalchemy.orm import Session

from ..config import get_settings
from ..models import Document
from . import store
from .embeddings import document_text, embed_documents

log = logging.getLogger(__name__)

MIN_CHUNK_CHARS = 120
# Por debajo de esto el documento entero se considera escaneado sin capa de texto.
MIN_DOC_CHARS = 200


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s.lower())
    return re.sub(r"[^a-z0-9 ]", "", "".join(c for c in s if not unicodedata.combining(c)))


def load_pages(path: Path | str) -> list[tuple[int, str]]:
    """[(número de página real, texto)]. La página es la unidad de trazabilidad."""
    path = Path(path)
    if path.suffix.lower() != ".pdf":
        return [(1, path.read_text(encoding="utf-8", errors="replace"))]
    with pymupdf.open(path) as doc:
        return [(i, page.get_text("text")) for i, page in enumerate(doc, start=1)]


def strip_running_heads(pages: list[tuple[int, str]]) -> list[tuple[int, str]]:
    """Quita encabezados y pies repetidos en la mayoría de las páginas."""
    if len(pages) < 4:
        return pages
    firsts = Counter(p[1].strip().split("\n")[0].strip() for p in pages if p[1].strip())
    lasts = Counter(p[1].strip().split("\n")[-1].strip() for p in pages if p[1].strip())
    umbral = max(3, len(pages) // 2)
    repes = {ln for ln, n in (firsts + lasts).items() if n >= umbral and 3 < len(ln) < 120}
    if not repes:
        return pages
    return [(n, "\n".join(l for l in t.split("\n") if l.strip() not in repes)) for n, t in pages]


def chunk_page(text: str) -> list[str]:
    """Trocea respetando párrafos y frontera de frase. Nunca cruza de página."""
    s = get_settings()
    text = re.sub(r"\n{3,}", "\n\n", re.sub(r"[ \t]+", " ", text)).strip()
    if not text:
        return []

    chunks: list[str] = []
    for para in text.split("\n\n"):
        para = para.strip()
        if not para:
            continue
        if len(para) <= s.chunk_size:
            chunks.append(para)
            continue
        start = 0
        while start < len(para):
            end = min(start + s.chunk_size, len(para))
            if end < len(para):
                corte = max(para.rfind(". ", start, end), para.rfind("\n", start, end))
                if corte > start + MIN_CHUNK_CHARS:
                    end = corte + 1
            chunks.append(para[start:end].strip())
            if end >= len(para):
                break
            start = max(start + 1, end - s.chunk_overlap)

    out: list[str] = []
    for c in chunks:
        if len(c) < MIN_CHUNK_CHARS and out:
            out[-1] = f"{out[-1]} {c}"      # se fusiona, no se tira
        elif len(c) >= MIN_CHUNK_CHARS:
            out.append(c)
    return out


def ingest_file(
    session: Session, path: Path | str, filename: str, procedure: str | None = None
) -> Document:
    """Indexa un documento y lo deja consultable. Devuelve la fila creada.

    El orden es el inverso al del borrado: primero los vectores, después marcar el
    documento como `indexed`. Así, si el proceso muere en medio, queda un documento
    en estado `indexing` que las consultas ignoran, en vez de uno consultable al
    que le faltan fragmentos.
    """
    doc = Document(filename=filename, procedure=procedure, status="indexing")
    session.add(doc)
    session.flush()

    try:
        pages = strip_running_heads(load_pages(path))
    except Exception as exc:  # noqa: BLE001
        log.warning("ingesta fallida de %s: %s", filename, exc)
        doc.status = "error"
        session.commit()
        return doc

    doc.n_pages = len(pages)
    if len("\n".join(t for _, t in pages).strip()) < MIN_DOC_CHARS:
        # PDF escaneado sin capa de texto. Se registra con ese estado y visible en
        # la consola: una limitación declarada es mejor que un fallo silencioso.
        doc.status = "sin_capa_texto"
        doc.n_chunks = 0
        session.commit()
        return doc

    ids, textos, metas = [], [], []
    vistos: set[str] = set()
    for page_no, texto in pages:
        for ci, chunk in enumerate(chunk_page(texto)):
            h = hashlib.sha256(_norm(chunk).encode()).hexdigest()[:16]
            if h in vistos:
                continue
            vistos.add(h)
            ids.append(f"{doc.id}:{page_no}:{ci}")
            textos.append(document_text(procedure=procedure, filename=Path(filename).stem,
                                        page=page_no, text=chunk))
            metas.append({
                "document_id": doc.id, "filename": filename, "page": page_no,
                "procedure": procedure or "", "source_folder": "upload",
                "topic": "upload", "off_scope": False, "chunk_index": ci, "doc_hash": doc.id,
            })

    if ids:
        store.add_chunks(ids=ids, documents=textos, metadatas=metas,
                         embeddings=embed_documents(textos))

    doc.n_chunks = len(ids)
    doc.status = "indexed"
    session.commit()
    log.info("indexado %s: %d páginas, %d chunks", filename, len(pages), len(ids))
    return doc


def delete_document(session: Session, document_id: str) -> bool:
    """Borra un documento y lo olvida. False si no existía.

    **Primero Chroma, después SQLite.** Si el proceso muere entre las dos, quedan
    metadatos sin vectores —que no devuelven nada— y nunca vectores servibles sin
    metadatos. Es la garantía de que "borrado" signifique borrado durante G5.
    """
    doc = session.get(Document, document_id)
    if doc is None:
        return False

    n = store.delete_document(document_id)
    session.delete(doc)
    session.commit()
    log.info("borrado %s: %d vectores eliminados", doc.filename, n)
    return True

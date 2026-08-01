"""Consola de conocimiento (gate G5, RF-06/RF-07).

- GET    /knowledge/documents        : listar (nº chunks, estado, timestamps).
- POST   /knowledge/documents        : subir PDF/texto -> parsear, trocear,
                                        vectorizar (Voyage), indexar en caliente.
- DELETE /knowledge/documents/{id}   : eliminar documento + sus vectores en una
                                        transacción (el agente "olvida" ya).
"""
from __future__ import annotations

import os
import tempfile

from fastapi import APIRouter, Depends, HTTPException, UploadFile
from sqlalchemy.orm import Session

from ..db import get_session
from ..models import Document
from ..rag import ingest

router = APIRouter(prefix="/knowledge", tags=["knowledge"])


@router.get("/documents")
def list_documents(session: Session = Depends(get_session)) -> list[dict]:
    docs = session.query(Document).order_by(Document.created_at.desc()).all()
    return [
        {
            "id": d.id,
            "filename": d.filename,
            "status": d.status,
            "n_chunks": d.n_chunks,
            "created_at": d.created_at.isoformat(),
            "updated_at": d.updated_at.isoformat(),
        }
        for d in docs
    ]


@router.post("/documents")
async def upload_document(
    file: UploadFile, session: Session = Depends(get_session)
) -> dict:
    suffix = os.path.splitext(file.filename or "doc")[1] or ".txt"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name
    try:
        doc = ingest.ingest_file(session, tmp_path, filename=file.filename)
    finally:
        os.unlink(tmp_path)
    return {"id": doc.id, "filename": doc.filename, "n_chunks": doc.n_chunks, "status": doc.status}


@router.delete("/documents/{document_id}")
def delete_document(document_id: str, session: Session = Depends(get_session)) -> dict:
    if not ingest.delete_document(session, document_id):
        raise HTTPException(404, "Documento no encontrado")
    return {"deleted": document_id}

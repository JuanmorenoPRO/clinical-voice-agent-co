"""Prueba de ida y vuelta del RAG: ingesta -> recuperación.

Requiere VOYAGE_API_KEY y una BD PostgreSQL+pgvector accesible (DATABASE_URL).
Si faltan, la prueba se salta (skip) — así el resto de la suite corre en cualquier
máquina sin credenciales (gate: los tests deterministas siempre pasan).
"""
from __future__ import annotations

import os
import tempfile

import pytest
from sqlalchemy import text

from app.config import get_settings

_settings = get_settings()


def _db_available() -> bool:
    try:
        from app.db import engine

        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _settings.voyage_api_key or not _db_available(),
    reason="Requiere VOYAGE_API_KEY y PostgreSQL+pgvector (DATABASE_URL).",
)


def test_ingest_then_retrieve():
    from app.db import SessionLocal, init_db
    from app.rag import ingest, retrieve

    init_db()
    session = SessionLocal()
    try:
        content = (
            "Una fiebre mayor a 38.5 grados sostenida puede indicar infección "
            "y requiere valoración médica tras la cirugía."
        )
        with tempfile.NamedTemporaryFile(
            "w", suffix=".md", delete=False, encoding="utf-8"
        ) as tmp:
            tmp.write(content)
            path = tmp.name

        try:
            doc = ingest.ingest_file(session, path, filename="prueba_fiebre.md")
            assert doc.n_chunks >= 1

            result = retrieve.retrieve(session, "¿Qué hago si tengo fiebre alta?")
            assert result.has_evidence
            assert result.sources
            assert result.sources[0].document == "prueba_fiebre.md"
        finally:
            os.unlink(path)
            ingest.delete_document(session, doc.id)
    finally:
        session.close()

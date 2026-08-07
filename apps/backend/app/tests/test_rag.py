"""Prueba de ida y vuelta del RAG: ingesta -> recuperación.

⚠️ Escrita para el stack anterior (Voyage AI + PostgreSQL/pgvector), que ya no
existe: los embeddings los sirve Ollama con bge-m3 y los vectores viven en
ChromaDB. Queda desactivada hasta que la migración del RAG la reescriba, junto
con el test de conocimiento vivo (alta/baja en caliente) que exige la compuerta G5.
"""
from __future__ import annotations

import os
import tempfile

import pytest

pytestmark = pytest.mark.skip(
    reason="Pendiente de reescritura para ChromaDB + bge-m3 (migración del RAG)."
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

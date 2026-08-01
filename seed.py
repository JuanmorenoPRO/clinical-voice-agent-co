"""Semilla de datos para una demo funcional inmediata (RNF-01).

Carga pacientes de ejemplo e ingesta los protocolos clínicos por el pipeline
real (Voyage -> pgvector), construyendo la base de embeddings de prueba que
SIMULA Databricks/Delta Share hasta que llegue el dataset real (7 de agosto).

Uso:
    python seed.py

Requiere:
    - DATABASE_URL apuntando a PostgreSQL+pgvector (docker compose lo levanta).
    - VOYAGE_API_KEY para generar los embeddings.
"""
from __future__ import annotations

import glob
import json
import os
import sys

# Permitir `python seed.py` desde la raíz del repo importando el paquete backend.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "apps", "backend"))

from app.config import get_settings  # noqa: E402
from app.db import SessionLocal, init_db  # noqa: E402
from app.models import Document, Patient  # noqa: E402
from app.rag import ingest  # noqa: E402

settings = get_settings()


def seed_patients(session) -> int:
    path = os.path.join(settings.samples_dir, "patients.json")
    with open(path, encoding="utf-8") as fh:
        rows = json.load(fh)

    created = 0
    for row in rows:
        exists = session.query(Patient).filter(Patient.name == row["name"]).first()
        if exists:
            continue
        session.add(
            Patient(
                name=row["name"],
                surgery=row["surgery"],
                surgery_date=row.get("surgery_date"),
                phone=row.get("phone"),
                extra=row.get("extra", {}),
            )
        )
        created += 1
    session.commit()
    return created


def seed_documents(session) -> int:
    # ------------------------------------------------------------------
    # TODO Aug 7 (Delta Share): reemplazar esta carga de archivos locales
    # por la ingesta desde Databricks Delta Share con el cliente
    # `delta-sharing`. El pipeline de ingesta (ingest.ingest_file) NO cambia;
    # solo cambia la FUENTE de los documentos/pacientes.
    #     import delta_sharing
    #     df = delta_sharing.load_as_pandas(settings.databricks_share_profile + "#share.schema.table")
    #     ...
    # ------------------------------------------------------------------
    paths = sorted(glob.glob(os.path.join(settings.samples_dir, "*.md")))
    paths = [p for p in paths if os.path.basename(p) != "README.md"]

    created = 0
    for path in paths:
        filename = os.path.basename(path)
        exists = session.query(Document).filter(Document.filename == filename).first()
        if exists:
            continue
        doc = ingest.ingest_file(session, path, filename=filename)
        print(f"  ingestado: {filename} -> {doc.n_chunks} chunks")
        created += 1
    return created


def main() -> None:
    print("Inicializando BD (pgvector + tablas)...")
    init_db()
    session = SessionLocal()
    try:
        n_patients = seed_patients(session)
        print(f"Pacientes cargados: {n_patients}")

        if not settings.voyage_api_key:
            print(
                "\n⚠️  VOYAGE_API_KEY no está configurada: se omite la ingesta de "
                "documentos (embeddings). Configúrala en .env para construir la "
                "base de embeddings de prueba."
            )
            return

        print("Ingestando documentos clínicos (Voyage -> pgvector)...")
        n_docs = seed_documents(session)
        print(f"Documentos ingestados: {n_docs}")
        print("\n✅ Semilla completa. La demo está lista.")
    finally:
        session.close()


if __name__ == "__main__":
    main()

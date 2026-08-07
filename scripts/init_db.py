"""Prepara la base de datos: tablas, pacientes y registro del corpus indexado.

    .venv/bin/python scripts/init_db.py

Los vectores viven en ChromaDB y los metadatos en SQLite, y **las consultas
filtran por los documentos vivos de SQLite** (ver `app/rag/store.py`). Eso protege
contra servir un documento borrado, pero tiene una consecuencia que hay que
atender aquí: un índice descargado sin sus filas en SQLite no devuelve nada en
absoluto. Este script reconcilia las dos mitades leyendo `manifest.json`.

Es idempotente: se puede volver a correr sin duplicar nada.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "apps" / "backend"))

from app.config import get_settings  # noqa: E402
from app.db import Base, SessionLocal, engine  # noqa: E402
from app.models import Document, Patient  # noqa: E402


def crear_tablas() -> None:
    Base.metadata.create_all(engine)
    print("✓ tablas creadas")


def sembrar_pacientes(session) -> int:
    ruta = Path(get_settings().seed_dir) / "patients.json"
    if not ruta.exists():
        print(f"  (sin {ruta}; corre scripts/load_dataset.py para generarlo)")
        return 0

    datos = json.loads(ruta.read_text(encoding="utf-8"))
    existentes = {p.id for p in session.query(Patient.id).all()}
    nuevos = 0
    for p in datos:
        if p["paciente_id"] in existentes:
            continue
        session.add(Patient(
            id=p["paciente_id"],
            name=p["nombre"],
            surgery=p["procedimiento"],
            surgery_date=p.get("fecha_cirugia"),
            extra={
                "edad": p.get("edad"), "genero": p.get("genero"),
                "comorbilidades": p.get("comorbilidades", []),
                "ciudad": p.get("ciudad"), "departamento": p.get("departamento"),
                "eps": p.get("eps"), "documento_cc": p.get("documento_cc"),
                "modulo": p.get("modulo"),
            },
        ))
        nuevos += 1
    session.commit()
    return nuevos


def registrar_corpus(session) -> tuple[int, int]:
    """Crea una fila `Document` por cada documento del índice.

    El `document_id` tiene que ser el mismo `doc_hash` que `build_index.py` grabó
    en la metadata de cada vector: es la clave con la que se emparejan las dos
    mitades.
    """
    manifest = Path(get_settings().chroma_dir) / "manifest.json"
    if not manifest.exists():
        print(f"  (sin {manifest}; corre scripts/fetch_index.py o build_index.py)")
        return 0, 0

    m = json.loads(manifest.read_text(encoding="utf-8"))
    existentes = {d.id for d in session.query(Document.id).all()}
    nuevos = 0
    for d in m["documents"]:
        if d["doc_hash"] in existentes:
            continue
        session.add(Document(
            id=d["doc_hash"],
            filename=d["file"],
            procedure=d.get("procedure") or None,
            status="indexed",
            n_chunks=d["chunks"],
            n_pages=d["pages"],
            off_scope=bool(d.get("off_scope")),
            topic=d.get("topic"),
        ))
        nuevos += 1

    # Los omitidos también se registran: un PDF escaneado sin capa de texto tiene
    # que verse en la consola con su estado, no desaparecer sin explicación.
    omitidos = 0
    for s in m.get("skipped", []):
        if s.get("reason") != "sin_capa_texto":
            continue  # los duplicados no aportan nada al operador
        doc_id = f"skip:{s['file']}"
        if doc_id in existentes:
            continue
        session.add(Document(
            id=doc_id, filename=s["file"], procedure=None,
            status="sin_capa_texto", n_chunks=0, n_pages=s.get("pages", 0),
            off_scope=True, topic=s.get("folder"),
        ))
        omitidos += 1

    session.commit()
    return nuevos, omitidos


def main() -> int:
    s = get_settings()
    print(f"Base de datos: {s.database_url}")
    print(f"Índice:        {s.chroma_dir}\n")

    crear_tablas()
    with SessionLocal() as session:
        pacientes = sembrar_pacientes(session)
        docs, omitidos = registrar_corpus(session)
        total_p = session.query(Patient).count()
        total_d = session.query(Document).count()

    print(f"✓ pacientes: {total_p} en total ({pacientes} nuevos)")
    print(f"✓ documentos: {total_d} en total ({docs} nuevos"
          + (f", {omitidos} sin capa de texto" if omitidos else "") + ")")

    # Comprobación de coherencia: si el índice tiene vectores pero la base no
    # tiene documentos, ninguna consulta devolvería nada y el fallo sería mudo.
    try:
        from app.rag import store

        vectores = store.count()
        print(f"✓ vectores en el índice: {vectores}")
        if vectores and not total_d:
            print("\n⚠️  Hay vectores pero ningún documento registrado: el RAG no "
                  "devolvería nada.\n    Falta data/chroma/manifest.json.", file=sys.stderr)
            return 1
    except Exception as exc:  # noqa: BLE001
        print(f"  (no se pudo leer el índice: {exc})")

    return 0


if __name__ == "__main__":
    sys.exit(main())

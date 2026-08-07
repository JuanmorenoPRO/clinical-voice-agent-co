"""Motor SQLAlchemy y sesión. SQLite para el estado de las llamadas (ADR-014).

Se abandonó PostgreSQL+pgvector: los vectores viven en ChromaDB y el resto —
pacientes, conversaciones, turnos, alertas, resúmenes— cabe de sobra en SQLite.
El motivo no fue técnico sino de compuerta: quitar el servidor de base de datos
elimina Docker de la ruta crítica del arranque de 15 minutos, y Docker en macOS
además rompe el UDP de WebRTC, que es justamente la voz que evalúa G4.
"""
from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path

from sqlalchemy import create_engine, event
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from .config import get_settings

log = logging.getLogger(__name__)
settings = get_settings()

_is_sqlite = settings.database_url.startswith("sqlite")
if _is_sqlite:
    # sqlite:///./data/clinical.db -> data/clinical.db
    ruta = Path(settings.database_url.split("///", 1)[-1])
    ruta.parent.mkdir(parents=True, exist_ok=True)

engine = create_engine(
    settings.database_url,
    pool_pre_ping=True,
    future=True,
    # El pipeline de voz toca la sesión desde el hilo del worker de Pipecat.
    connect_args={"check_same_thread": False} if _is_sqlite else {},
)

if _is_sqlite:
    @event.listens_for(engine, "connect")
    def _sqlite_pragmas(dbapi_conn, _record):  # noqa: ANN001
        cur = dbapi_conn.cursor()
        # WAL permite leer mientras se escribe: la consola hace polling de alertas
        # mientras la llamada está escribiendo turnos.
        cur.execute("PRAGMA journal_mode=WAL")
        cur.execute("PRAGMA synchronous=NORMAL")
        cur.execute("PRAGMA foreign_keys=ON")
        cur.close()

SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False, future=True)


class Base(DeclarativeBase):
    pass


def init_db() -> None:
    """Crea las tablas y repara el almacén vectorial si quedó a medias. Idempotente."""
    from . import models  # noqa: F401  — registra las tablas en Base.metadata

    Base.metadata.create_all(engine)

    # Reconciliación Chroma <-> SQLite: si un borrado se cortó por la mitad pueden
    # quedar vectores sin documento. Aquí se limpian, para que ninguna consulta
    # pueda devolver un fragmento de algo que ya se borró (compuerta G5).
    try:
        from .models import Document
        from .rag import store

        with SessionLocal() as session:
            activos = {d for (d,) in session.query(Document.id).all()}
        if store.count():
            huerfanos = store.orphan_sweep(activos)
            if huerfanos:
                log.warning("init_db: %d vectores huérfanos eliminados", huerfanos)
    except Exception as exc:  # noqa: BLE001
        # Un índice ausente no debe impedir arrancar: la consola tiene que poder
        # abrirse para subir documentos aunque no haya corpus todavía.
        log.warning("init_db: no se pudo reconciliar el índice vectorial: %s", exc)


def get_session() -> Iterator[Session]:
    """Dependencia FastAPI: una sesión por request."""
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()

"""Motor SQLAlchemy y sesión. Una sola instancia PostgreSQL + pgvector (ADR-004)."""
from __future__ import annotations

from collections.abc import Iterator

from sqlalchemy import create_engine, text
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from .config import get_settings

settings = get_settings()

engine = create_engine(settings.database_url, pool_pre_ping=True, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False, future=True)


class Base(DeclarativeBase):
    pass


def init_db() -> None:
    """Crea la extensión pgvector y todas las tablas. Idempotente."""
    # Importar modelos registra las tablas en Base.metadata.
    from . import models  # noqa: F401

    with engine.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    Base.metadata.create_all(engine)
    # Migración ligera idempotente: añade columnas nuevas a tablas ya existentes
    # (create_all no altera tablas creadas antes). Postgres soporta IF NOT EXISTS.
    with engine.begin() as conn:
        conn.execute(text("ALTER TABLE documents ADD COLUMN IF NOT EXISTS procedure VARCHAR"))


def get_session() -> Iterator[Session]:
    """Dependencia FastAPI: una sesión por request."""
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()

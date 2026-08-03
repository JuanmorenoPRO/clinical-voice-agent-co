"""Modelo de datos (architecture.md §5).

Las 7 tablas del sistema. La columna de embedding en `chunks` usa pgvector con
dimensión configurable (EMBEDDING_DIM) porque el modelo de embeddings puede
cambiar el 7 de agosto — re-embeder implica recrear esa columna.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    JSON,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .config import get_settings
from .db import Base

EMBEDDING_DIM = get_settings().embedding_dim


def _uuid() -> str:
    return str(uuid.uuid4())


def _now() -> datetime:
    return datetime.now(timezone.utc)


class Patient(Base):
    __tablename__ = "patients"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    name: Mapped[str] = mapped_column(String, nullable=False)
    surgery: Mapped[str] = mapped_column(String, nullable=False)
    surgery_date: Mapped[str | None] = mapped_column(String, nullable=True)
    phone: Mapped[str | None] = mapped_column(String, nullable=True)
    # Campos libres del dataset real (⏳ esquema definitivo llega el 7 de agosto).
    extra: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    filename: Mapped[str] = mapped_column(String, nullable=False)
    # Procedimiento al que aplica el documento (p. ej. "colecistectomia").
    # NULL = conocimiento general, válido para cualquier paciente. Permite
    # recuperación filtrada por el procedimiento del paciente (grounding).
    procedure: Mapped[str | None] = mapped_column(String, nullable=True)
    status: Mapped[str] = mapped_column(String, default="indexed")  # indexing|indexed|error
    n_chunks: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, onupdate=_now)

    chunks: Mapped[list["Chunk"]] = relationship(
        back_populates="document", cascade="all, delete-orphan"
    )


class Chunk(Base):
    __tablename__ = "chunks"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    document_id: Mapped[str] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), index=True
    )
    page: Mapped[int] = mapped_column(Integer, default=1)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    embedding: Mapped[list[float]] = mapped_column(Vector(EMBEDDING_DIM))

    document: Mapped["Document"] = relationship(back_populates="chunks")


class Conversation(Base):
    __tablename__ = "conversations"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    patient_id: Mapped[str | None] = mapped_column(ForeignKey("patients.id"), nullable=True)
    status: Mapped[str] = mapped_column(String, default="active")  # active|closed
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    turns: Mapped[list["Turn"]] = relationship(
        back_populates="conversation", cascade="all, delete-orphan"
    )


class Turn(Base):
    """Traza completa por turno (RF-05) — nada es caja negra."""

    __tablename__ = "turns"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    conversation_id: Mapped[str] = mapped_column(
        ForeignKey("conversations.id", ondelete="CASCADE"), index=True
    )
    patient_utterance: Mapped[str] = mapped_column(Text, default="")
    extracted_symptoms: Mapped[dict] = mapped_column(JSON, default=dict)
    # Evidencia RAG: [{document, page, chunk_id, score}]
    retrieved_chunks: Mapped[list] = mapped_column(JSON, default=list)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    triggered_rules: Mapped[list] = mapped_column(JSON, default=list)
    risk_level: Mapped[str] = mapped_column(String, default="NORMAL")
    critical_override: Mapped[bool] = mapped_column(default=False)
    final_response: Mapped[str] = mapped_column(Text, default="")
    latency_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    conversation: Mapped["Conversation"] = relationship(back_populates="turns")


class Alert(Base):
    __tablename__ = "alerts"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    conversation_id: Mapped[str] = mapped_column(ForeignKey("conversations.id"), index=True)
    patient_id: Mapped[str | None] = mapped_column(ForeignKey("patients.id"), nullable=True)
    risk_level: Mapped[str] = mapped_column(String)  # ALTO|CRÍTICO
    triggered_rules: Mapped[list] = mapped_column(JSON, default=list)
    symptoms: Mapped[dict] = mapped_column(JSON, default=dict)
    transcript: Mapped[str] = mapped_column(Text, default="")
    status: Mapped[str] = mapped_column(String, default="pending")  # pending|attended
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class Summary(Base):
    __tablename__ = "summaries"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    conversation_id: Mapped[str] = mapped_column(ForeignKey("conversations.id"), unique=True)
    data: Mapped[dict] = mapped_column(JSON, default=dict)  # JSON estructurado (RF-10)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

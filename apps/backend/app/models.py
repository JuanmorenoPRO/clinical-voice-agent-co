"""Modelo de datos (architecture.md §5).

Seis tablas en SQLite. **Los vectores no están aquí**: viven en ChromaDB, y esta
base solo guarda los metadatos de los documentos. La consecuencia de diseño está
en `rag/store.py`: como los dos almacenes no comparten transacción, toda consulta
filtra por los documentos vivos de esta tabla, de modo que un vector huérfano no
pueda servirse nunca.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .db import Base


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
    # indexing|indexed|error|sin_capa_texto. El último no es un fallo: es un PDF
    # escaneado sin texto extraíble, y se muestra así en la consola en vez de
    # desaparecer sin explicación.
    status: Mapped[str] = mapped_column(String, default="indexed")
    n_chunks: Mapped[int] = mapped_column(Integer, default=0)
    n_pages: Mapped[int] = mapped_column(Integer, default=0)
    # Fuera del alcance clínico del procedimiento: se indexa y se puede consultar
    # como conocimiento general, pero nunca se sirve como evidencia del
    # procedimiento del paciente. Ver rag/retrieve.py::_allowed_document_ids.
    off_scope: Mapped[bool] = mapped_column(Boolean, default=False)
    topic: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, onupdate=_now)


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

    # Estado del guion tras este turno (fase, slot en curso, re-preguntas gastadas).
    # La memoria de la llamada vive aquí y no en el contexto del modelo: ahorra
    # ~800 tokens por turno y evita que un 3B se olvide de lo que ya preguntó.
    agent_state: Mapped[dict] = mapped_column(JSON, default=dict)
    intent: Mapped[str] = mapped_column(String, default="respuesta")

    # Consumo, para las métricas obligatorias del README. Se guardan por turno y
    # se agregan por llamada, de modo que el número reportado y el log salgan de
    # la misma fuente y no puedan divergir.
    llm_calls: Mapped[int] = mapped_column(Integer, default=0)
    tokens_in: Mapped[int] = mapped_column(Integer, default=0)
    tokens_out: Mapped[int] = mapped_column(Integer, default=0)
    # El modelo falló o expiró y el turno se resolvió solo con el léxico.
    degraded: Mapped[bool] = mapped_column(Boolean, default=False)

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
    # pending|attended|deleted. `deleted` es un borrado lógico: la fila se queda
    # (la deduplicación de alertas la sigue viendo, ver routers/console.py).
    status: Mapped[str] = mapped_column(String, default="pending")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class Summary(Base):
    __tablename__ = "summaries"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    conversation_id: Mapped[str] = mapped_column(ForeignKey("conversations.id"), unique=True)
    data: Mapped[dict] = mapped_column(JSON, default=dict)  # JSON estructurado (RF-10)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

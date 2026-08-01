"""Contratos Pydantic compartidos entre módulos y expuestos por la API.

Estos son los ~5 tipos que el frontend espeja a mano en TypeScript (ADR-008).
`Symptoms` es la salida estructurada del LLM que consume el Motor de Decisión;
mantenerlo estable es clave: el 7 de agosto solo se **amplía** con el vocabulario
del dataset real, sin cambiar el contrato.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

RiskLevel = Literal["NORMAL", "ALTO", "CRÍTICO"]


class Symptoms(BaseModel):
    """Síntomas extraídos de la conversación (tarea de NLP del LLM, ADR-001).

    El LLM solo *extrae*; nunca decide el nivel de riesgo. Todos los campos son
    opcionales porque el paciente reporta de forma parcial y ambigua.
    ⏳ 7 ago: ampliar con campos/vocabulario del dataset colombiano real.
    """

    pain_level: int | None = Field(None, ge=0, le=10, description="Escala de dolor 0–10")
    medication_effective: bool | None = Field(
        None, description="¿La medicación controla el dolor?"
    )
    temperature_c: float | None = Field(None, description="Temperatura en °C si la reporta")
    fever: bool | None = None
    heavy_bleeding: bool | None = Field(None, description="Sangrado abundante")
    breathing_difficulty: bool | None = None
    loss_of_consciousness: bool | None = None
    # Texto libre para síntomas no mapeados (coloquialismos: "me siento maluco").
    other: list[str] = Field(default_factory=list)


class Source(BaseModel):
    """Cita a la fuente (RF-05)."""

    document: str
    page: int
    chunk_id: str
    score: float


class RagResult(BaseModel):
    """Contrato del Servicio de Recuperación (architecture.md §3.3)."""

    answer: str
    confidence: float
    sources: list[Source] = Field(default_factory=list)
    has_evidence: bool = True


class LLMTurnOutput(BaseModel):
    """Salida de la ÚNICA llamada LLM por turno (ADR-006)."""

    sintomas: Symptoms
    respuesta: str


class DecisionResult(BaseModel):
    risk_level: RiskLevel
    triggered_rules: list[str] = Field(default_factory=list)
    safety_script: str | None = None  # texto fijo si el nivel es CRÍTICO


# --- Requests/Responses de la API ---


class TurnRequest(BaseModel):
    conversation_id: str | None = None
    patient_id: str | None = None
    text: str


class TurnResponse(BaseModel):
    conversation_id: str
    turn_id: str
    response: str
    risk_level: RiskLevel
    triggered_rules: list[str]
    symptoms: Symptoms
    sources: list[Source]
    critical_override: bool
    alert_id: str | None = None

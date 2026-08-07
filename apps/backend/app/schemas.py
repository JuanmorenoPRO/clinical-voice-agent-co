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

# Vocabulario de criticidad del dataset oficial. Mapea 1:1 con RiskLevel; se expone
# en paralelo para que el informe y la evaluación hablen el idioma del ground truth.
TriageColor = Literal["verde", "amarillo", "rojo"]
EscalationAction = Literal["ninguna", "seguimiento", "enfermeria_prioritaria", "emergencia_123"]

# Los cuatro slots estructurados del dataset. El vocabulario es literal:
# `trayectorias_postop_silver.xlsx` los usa con estos mismos valores.
Mobility = Literal["normal", "limitada_esperada", "incapacitante_nueva"]
Wound = Literal["normal", "eritema_leve", "secrecion_purulenta"]
Appetite = Literal["normal", "levemente_disminuido", "muy_disminuido"]
Sleep = Literal["normal", "levemente_alterado", "muy_alterado"]


class Symptoms(BaseModel):
    """Síntomas extraídos de la conversación (tarea de NLP del LLM, ADR-001).

    El LLM solo *extrae*; nunca decide el nivel de riesgo. Todos los campos son
    opcionales porque el paciente reporta de forma parcial y ambigua, y un campo en
    `None` **nunca** se interpreta como normalidad (ver decision/rules.py).
    """

    # --- los seis slots del guion canónico del dataset -----------------------
    pain_level: int | None = Field(None, ge=0, le=10, description="Escala de dolor 0–10")
    temperature_c: float | None = Field(None, description="Temperatura en °C si la reporta")
    fever: bool | None = None
    mobility: Mobility | None = Field(None, description="Movilidad / capacidad de caminar")
    wound: Wound | None = Field(None, description="Estado de la herida quirúrgica")
    appetite: Appetite | None = None
    sleep: Sleep | None = None

    medication_effective: bool | None = Field(
        None, description="¿La medicación controla el dolor?"
    )

    # --- banderas rojas de emergencia -----------------------------------------
    # No están en el dataset, pero sí en los escenarios que el jurado interpreta
    # en vivo. Se conservan tal cual: son las que disparan el guion del 123.
    heavy_bleeding: bool | None = Field(None, description="Sangrado abundante")
    breathing_difficulty: bool | None = None
    loss_of_consciousness: bool | None = None
    chest_pain: bool | None = Field(None, description="Dolor en el pecho / torácico")
    altered_mental_status: bool | None = Field(
        None, description="Confusión, desorientación o estado mental alterado"
    )
    seizure: bool | None = Field(None, description="Convulsión / crisis convulsiva")

    # --- contexto y trazabilidad ----------------------------------------------
    day_postop: int | None = Field(None, description="Día postoperatorio (1, 3, 7, 14)")
    # De dónde salió cada slot: paciente | cuidador | lexicon | llm. Importa porque
    # un cuidador que reporta una bandera roja debe escalar igual que el paciente.
    sources: dict[str, str] = Field(default_factory=dict)
    # Slots que quedaron sin respuesta tras agotar las re-preguntas.
    unanswered: list[str] = Field(default_factory=list)
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
    # Mismo nivel en el vocabulario del dataset: verde↔NORMAL, amarillo↔ALTO,
    # rojo↔CRÍTICO. No es un cuarto nivel, es la misma escala con otro nombre.
    triage_color: TriageColor = "verde"
    # Qué se hace y qué se le comunica al paciente como siguiente paso.
    escalation_action: EscalationAction = "ninguna"
    triggered_rules: list[str] = Field(default_factory=list)
    safety_script: str | None = None  # texto fijo si el nivel es CRÍTICO
    # Fracción de los 6 slots del guion que se logró responder. Alimenta la política
    # de escalamiento por incertidumbre y el resumen de la llamada.
    completeness: float = 0.0


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

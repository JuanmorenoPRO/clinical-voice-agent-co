"""Modelos de datos del framework de evaluación (Pydantic v2).

Definen los contratos que atraviesan todo el pipeline:

    Scenario  --(ConversationRunner)-->  Transcript  --(Evaluators)-->  ScenarioResult

Ninguno de estos modelos depende del transporte (texto/voz/HTTP): un `Transcript`
es el mismo objeto sin importar cómo se produjo, lo que permite reutilizar los
evaluadores y el reporte cuando llegue la versión de voz (STT→agente→TTS).
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

# Nivel de riesgo tal como lo pide el usuario (verde/amarillo/rojo). Se mapea a los
# niveles internos del agente (NORMAL/ALTO/CRÍTICO) en `framework.config`.
Risk = Literal["green", "yellow", "red"]

# Categorías de escenarios (una carpeta por categoría bajo tests/scenarios/).
Category = Literal[
    "green",
    "yellow",
    "red",
    "emotional",
    "edge_cases",
    "memory",
    "colombian_language",
    "adversarial",
    "knowledge",
]


class Expected(BaseModel):
    """Comportamiento esperado del agente para un escenario.

    Todos los campos son opcionales: cada categoría usa solo los que le aplican
    (p.ej. `remember` para memoria, `must_interpret` para lenguaje colombiano).
    """

    risk: Risk | None = None
    should_escalate: bool | None = None
    should_reassure: bool | None = None
    # Memoria: datos que el agente debe recordar/considerar en turnos posteriores.
    # p.ej. {"ubicacion": "hombro izquierdo"}.
    remember: dict[str, str] = Field(default_factory=dict)
    # Lenguaje colombiano: coloquialismos que deben interpretarse como síntoma.
    # p.ej. ["me duele un berraco", "me dio una calentura"].
    must_interpret: list[str] = Field(default_factory=list)
    # Cosas que la respuesta NUNCA debe hacer (para el evaluador clínico).
    # Valores soportados: "diagnostico", "tratamiento".
    forbid: list[str] = Field(default_factory=list)
    # Conocimiento vivo (categoría `knowledge`): True = la pregunta NO está en el
    # corpus y el agente debe declarar el límite; False = SÍ está y debe responder
    # citando la fuente recuperada. None = no aplica (el evaluador `knowledge` no
    # juzga el escenario).
    should_abstain: bool | None = None


class Scenario(BaseModel):
    """Un caso de prueba conversacional (se serializa como un .json)."""

    name: str
    risk: Risk
    category: Category
    # Paciente sembrado opcional (habilita RAG filtrado por procedimiento).
    patient_id: str | None = None
    messages: list[str]
    expected: Expected = Field(default_factory=Expected)


class TurnRecord(BaseModel):
    """Un turno de la conversación = mensaje del paciente + `TurnResponse` aplanado.

    Es la proyección del `app.schemas.TurnResponse` a un modelo propio del framework
    para no acoplar los evaluadores al backend.
    """

    patient_text: str
    response: str
    risk_level: str  # NORMAL | ALTO | CRÍTICO (nivel interno del agente)
    triggered_rules: list[str] = Field(default_factory=list)
    symptoms: dict = Field(default_factory=dict)
    critical_override: bool = False
    sources: list[dict] = Field(default_factory=list)
    alert_id: str | None = None
    latency_ms: int | None = None


# Orden de severidad de los niveles internos del agente (peor = mayor).
_RISK_ORDER = {"NORMAL": 0, "ALTO": 1, "CRÍTICO": 2}


class Transcript(BaseModel):
    """La conversación completa producida por un `ConversationRunner`."""

    conversation_id: str
    turns: list[TurnRecord] = Field(default_factory=list)
    # Lo que el agente dice ANTES de que el paciente hable. No es un `TurnRecord`
    # porque no tiene texto del paciente y contarlo como turno falsearía los
    # evaluadores de estilo. No se envía a `process_turn`: `CallState` ya arranca
    # en TAMIZAJE dando por hecha la pregunta de dolor que la apertura lleva
    # dentro (ver agent/script.py). Está aquí para que el informe muestre la
    # conversación tal como suena, empezando por el agente — sin esto parecía que
    # el paciente hablaba primero y que el agente le respondía con una
    # reformulación salida de la nada.
    apertura: str | None = None
    # Riesgo tras aplicar la política de incertidumbre al cerrar
    # (`engine.evaluate(final=True)`, vía `summary.close_conversation`). Va aparte
    # porque no es un turno hablado: es la reevaluación de la llamada completa.
    # Sin esto el arnés no medía el final real — al paciente guionizado se le
    # acaban los mensajes, la llamada nunca se cerraba y una conversación que no
    # se pudo evaluar se reportaba NORMAL en vez de escalada.
    risk_al_cierre: str | None = None

    @property
    def final(self) -> TurnRecord:
        """El último turno (respuesta final del agente)."""
        return self.turns[-1]

    @property
    def max_risk(self) -> str:
        """El peor nivel de riesgo alcanzado en toda la conversación.

        La escalación se evalúa sobre la conversación entera, no solo el último
        turno: una emergencia en el turno 2 cuenta aunque el turno final baje.
        """
        if not self.turns:
            return "NORMAL"
        niveles = [t.risk_level for t in self.turns]
        # El cierre cuenta: una llamada que se quedó a medias escala al colgar, y
        # ese escalamiento es tan real como el de un turno.
        if self.risk_al_cierre:
            niveles.append(self.risk_al_cierre)
        return max(niveles, key=lambda r: _RISK_ORDER.get(r, 0))

    @property
    def escalated(self) -> bool:
        """True si el agente escaló en algún momento (ALTO/CRÍTICO o alerta creada)."""
        return any(
            t.risk_level in ("ALTO", "CRÍTICO") or t.alert_id is not None
            for t in self.turns
        )

    @property
    def full_text(self) -> str:
        """Transcripción legible paciente/agente (para prompts de juez y reportes)."""
        lines: list[str] = []
        for t in self.turns:
            lines.append(f"Paciente: {t.patient_text}")
            lines.append(f"Agente: {t.response}")
        return "\n".join(lines)


class EvalResult(BaseModel):
    """Resultado de un evaluador para un escenario."""

    evaluator: str
    score: float = Field(ge=0.0, le=1.0)  # normalizado 0–1
    passed: bool
    reason: str
    # Detalles libres: p.ej. la empatía guarda su puntaje crudo 1–10 aquí.
    details: dict = Field(default_factory=dict)


class ScenarioResult(BaseModel):
    """Resultado agregado de un escenario tras correr todos los evaluadores."""

    scenario: Scenario
    transcript: Transcript
    evals: list[EvalResult] = Field(default_factory=list)
    passed: bool
    overall_score: float = Field(ge=0.0, le=1.0)
    # Marcadores de seguridad para las métricas globales.
    false_positive: bool = False  # escaló cuando NO debía
    false_negative: bool = False  # NO escaló cuando debía (fallo crítico)

    def eval_by_name(self, name: str) -> EvalResult | None:
        for e in self.evals:
            if e.evaluator == name:
                return e
        return None

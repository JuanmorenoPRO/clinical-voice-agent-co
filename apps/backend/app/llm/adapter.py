"""Interfaz única del LLM (ADR-002).

Nada fuera de este paquete conoce al proveedor concreto. El modelo se selecciona
con LLM_PROVIDER; hoy es `ollama` (llama3.2:3b), el único de la lista permitida
por la compuerta G3 que sigue siendo invocable — ver docs/spikes-7-agosto.md.

El contrato cambió respecto a la versión anterior. Antes había una única llamada
`turn()` que extraía síntomas Y redactaba la respuesta. Ahora son dos operaciones
separadas porque el guion de la conversación es determinista (app/agent/script.py)
y el LLM ya no decide qué decir:

  extract()         siempre que haya que interpretar al paciente
  reply_grounded()  solo si el paciente hizo una pregunta clínica

La redacción de las preguntas y los acuses empáticos NO pasa por aquí: sale del
banco de plantillas, porque un 3B genera acuses inservibles ("agudo", "se siente
mal") y cuesta ~11 tokens que en voz se notan.
"""
from __future__ import annotations

from typing import Protocol

from pydantic import BaseModel, Field

from ..schemas import Symptoms


class LLMUsage(BaseModel):
    """Contabilidad de una llamada. Alimenta las métricas obligatorias del README."""

    model: str = ""
    tokens_in: int = 0
    tokens_out: int = 0
    latency_ms: float = 0.0
    purpose: str = ""


class Extraction(BaseModel):
    """Lo que el LLM devuelve de un turno del paciente.

    `symptoms` trae solo los campos que se lograron leer; el resto queda en None y
    el orquestador lo fusiona con lo que ya sabía y con lo que sacó el léxico.
    """

    symptoms: Symptoms = Field(default_factory=Symptoms)
    intent: str = "respuesta"
    usage: LLMUsage = Field(default_factory=LLMUsage)
    # True si el modelo falló o expiró y solo hay lo que dio el léxico determinista.
    degraded: bool = False


class LLMAdapter(Protocol):
    async def extract(
        self, *, slot: str | None, question: str, utterance: str
    ) -> Extraction:
        """Interpreta lo que dijo el paciente para el `slot` que se está preguntando.

        La implementación restringe la generación con un JSON Schema, de modo que
        un valor fuera del vocabulario sea imposible por construcción. `slot=None`
        para turnos fuera del guion (apertura, cierre, pregunta suelta): entonces
        solo se vigilan banderas de emergencia e intención.

        Nunca lanza: ante un fallo del modelo devuelve `degraded=True` con síntomas
        vacíos, y el turno se completa con lo que aportó el léxico.
        """
        ...

    async def reply_grounded(
        self, *, question: str, evidence: str, patient_context: str
    ) -> tuple[str, LLMUsage]:
        """Responde UNA pregunta clínica usando exclusivamente `evidence`.

        Si la evidencia no responde, devuelve la frase de abstención. El
        orquestador valida después que no aparezcan cifras ni fármacos ausentes
        de la evidencia.
        """
        ...

    async def summarize(self, *, system_prompt: str, user_prompt: str) -> str:
        """Texto libre para el resumen narrativo de la llamada (opcional)."""
        ...

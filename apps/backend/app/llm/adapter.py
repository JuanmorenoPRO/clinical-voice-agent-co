"""Interfaz única del LLM (ADR-002).

Nada fuera de este paquete conoce al proveedor concreto. El modelo se selecciona
con LLM_PROVIDER; hoy es `groq` (`llama-3.3-70b-versatile`), el sucesor vigente
de Llama en Groq permitido por la compuerta G3 — ver docs/spikes-7-agosto.md.

El contrato cambió respecto a la versión anterior. Antes había una única llamada
`turn()` que extraía síntomas Y redactaba la respuesta. Ahora son dos operaciones
separadas porque el guion de la conversación es determinista (app/agent/script.py)
y el LLM ya no decide qué decir:

  extract()         siempre que haya que interpretar al paciente
  reply_grounded()  solo si el paciente hizo una pregunta clínica

La redacción del turno volvió al LLM con `compose_reply()`, pero con el guion
como restricción: el sistema decide QUÉ (qué slot preguntar, cuándo escalar,
cuándo cerrar) y el modelo redacta CÓMO, viendo el historial reciente. La
decisión original de sacarla —un 3B generaba acuses inservibles ("agudo", "se
siente mal")— era correcta para `llama3.2:3b`, pero se mantuvo intacta al saltar
a un 70B en Groq y el resultado era un agente de piezas pegadas que no escucha.
Las plantillas de `agent/phrasing.py` siguen siendo el respaldo: si el redactor
falla, expira o su salida no pasa las guardas, el turno sale determinista.
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


class ComposeContext(BaseModel):
    """Todo lo que ve el redactor de un turno. Lo arma `agent/composer.py`.

    El LLM no decide nada con esto: `pregunta_guion` y `objetivo` ya vienen
    resueltos por la máquina de estados. El contexto existe para que la
    redacción suene a alguien que escuchó la conversación, no a piezas pegadas.
    """

    # Últimos turnos, del más antiguo al más reciente:
    # [{"rol": "paciente"|"agente", "texto": "..."}]
    historial: list[dict[str, str]] = Field(default_factory=list)
    utterance: str
    intent: str = "respuesta"
    # Texto canónico de la siguiente pregunta del guion (de phrasing, con su
    # rotación). La salida DEBE terminar con ella o una variante fiel.
    pregunta_guion: str
    # action.kind: preguntar | repreguntar | seguimiento | ofrecer_salida |
    # confirmar | cerrar
    objetivo: str
    # Evidencia RAG ya recortada y pertinente; None => si el paciente preguntó,
    # hay que decirle que no está en los documentos.
    evidencia: str | None = None
    # Resumen legible del cuadro acumulado ("dolor 4/10, sin fiebre, ...").
    sintomas: str = ""
    procedimiento: str | None = None
    nombre: str | None = None
    riesgo: str = "NORMAL"
    # Instrucciones puntuales del sistema para este turno, ya redactadas
    # ("El paciente además se está despidiendo: ...").
    notas: list[str] = Field(default_factory=list)
    # El paciente consultó por un síntoma de alarma en este turno: la validación
    # endurece las guardas de veredicto aunque el riesgo siga en NORMAL.
    alarma: bool = False


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

    async def compose_reply(self, *, ctx: ComposeContext) -> tuple[str, LLMUsage] | None:
        """Redacta el turno completo que se dirá en voz alta.

        Integra en una sola frase natural: acuse de lo que dijo el paciente,
        respuesta a su pregunta (desde `ctx.evidencia`, o abstención coherente)
        y la pregunta del guion con la que debe terminar.

        Nunca lanza: devuelve None ante fallo/timeout/salida vacía, y el
        orquestador cae a las plantillas de `agent/phrasing.py`. La validación
        de la salida (cifras, veredictos, inyección) es de `agent/composer.py`,
        no de aquí.
        """
        ...

    async def summarize(self, *, system_prompt: str, user_prompt: str) -> str:
        """Texto libre para el resumen narrativo de la llamada (opcional)."""
        ...

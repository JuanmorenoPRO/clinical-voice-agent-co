"""El esqueleto determinista de la llamada: qué se pregunta y en qué orden.

El dataset oficial revela que el agente de referencia sigue un guion fijo de seis
preguntas —dolor, fiebre, movilidad, herida, apetito, sueño— siempre en ese orden,
en los turnos 0, 2, 4, 6, 8 y 10. No es casualidad: son exactamente las seis
variables con las que `trayectorias_postop_silver.xlsx` describe cada caso, y por
tanto las que determinan la criticidad.

Que el guion viva en código y no en el prompt tiene tres consecuencias que importan:

  - **Un modelo de 3B no puede perderse.** No decide qué preguntar, solo interpreta
    la respuesta. Si falla, el guion sigue en pie y hay un texto canónico al que caer.
  - **La inyección de prompt se vuelve estructuralmente inofensiva.** Puede alterar
    el fraseo de una frase; no puede saltarse una pregunta, cambiar el orden ni
    suprimir un escalamiento.
  - **Se puede pre-sintetizar el audio.** Las seis preguntas son texto fijo, así que
    el TTS de un turno normal cuesta 0 ms en vez de ~250.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from ..schemas import Symptoms


class Phase(StrEnum):
    APERTURA = "apertura"        # identificarse, confirmar paciente y día postop
    TAMIZAJE = "tamizaje"        # los seis slots, en orden
    ABIERTO = "abierto"          # "¿algo más que quiera contarme?"
    CIERRE = "cierre"            # recapitulación y próximos pasos
    ESCALAMIENTO = "escalamiento"  # guion de seguridad; entra desde cualquier fase
    TERMINADA = "terminada"


# Orden canónico del dataset. El campo es el de `Symptoms` que rellena cada slot.
SLOTS: tuple[str, ...] = ("dolor", "fiebre", "movilidad", "herida", "apetito", "sueno")
SLOTS_INICIAL = SLOTS[0]

SLOT_FIELD: dict[str, str] = {
    "dolor": "pain_level",
    "fiebre": "fever",
    "movilidad": "mobility",
    "herida": "wound",
    "apetito": "appetite",
    "sueno": "sleep",
}

# Máximo de reformulaciones antes de dar el slot por perdido y seguir. Dos es lo
# que aguanta una conversación sin volverse un interrogatorio; a partir de ahí
# insistir irrita y no consigue el dato.
MAX_REPREGUNTAS = 2


@dataclass
class CallState:
    """Memoria de la llamada. Vive en la base, no en el contexto del modelo.

    Esto es deliberado: mantener el historial fuera del prompt recorta ~800 tokens
    por turno y elimina la deriva de un modelo pequeño que se olvida de lo que ya
    preguntó.
    """

    # Arranca en TAMIZAJE preguntando por el dolor, no en APERTURA: el saludo ya
    # lleva esa pregunta (ver phrasing.APERTURA). Sin esto, el primer turno del
    # paciente —que responde al dolor— se procesaba sin slot asignado, y el guion
    # quedaba un paso por detrás de la conversación durante toda la llamada.
    phase: Phase = Phase.TAMIZAJE
    slot_actual: str | None = SLOTS_INICIAL
    repreguntas: dict[str, int] = field(default_factory=dict)
    resueltos: list[str] = field(default_factory=list)
    sin_responder: list[str] = field(default_factory=list)
    turnos: int = 0

    def to_dict(self) -> dict:
        return {
            "phase": str(self.phase), "slot_actual": self.slot_actual,
            "repreguntas": self.repreguntas, "resueltos": self.resueltos,
            "sin_responder": self.sin_responder, "turnos": self.turnos,
        }

    @classmethod
    def from_dict(cls, d: dict | None) -> "CallState":
        d = d or {}
        return cls(
            phase=Phase(d.get("phase", Phase.APERTURA)),
            slot_actual=d.get("slot_actual"),
            repreguntas=dict(d.get("repreguntas") or {}),
            resueltos=list(d.get("resueltos") or []),
            sin_responder=list(d.get("sin_responder") or []),
            turnos=int(d.get("turnos") or 0),
        )


@dataclass(frozen=True)
class Action:
    """Qué toca hacer en este turno. Es dato, no texto: el fraseo va aparte."""

    kind: str          # preguntar | repreguntar | cerrar | escalar | responder_y_seguir
    slot: str | None = None
    phase: Phase = Phase.TAMIZAJE
    intento: int = 0


def slots_pendientes(state: CallState, symptoms: Symptoms) -> list[str]:
    """Slots que aún no tienen valor y no se han dado por perdidos."""
    return [
        s for s in SLOTS
        if getattr(symptoms, SLOT_FIELD[s]) is None and s not in state.sin_responder
    ]


def next_action(state: CallState, symptoms: Symptoms, *, escalar: bool = False) -> Action:
    """Decide el siguiente movimiento. Función pura: se testea sin LLM ni red.

    `escalar` lo impone el motor de decisión desde fuera; cuando llega, corta el
    guion sin importar en qué punto esté. Una emergencia no espera a que terminen
    las preguntas.

    Una vez escalado, el cuadro crítico queda en `symptoms` para siempre —no hay
    forma de "des-escalar" dentro de la llamada—, así que `escalar` seguiría
    siendo True en todos los turnos siguientes. Sin este corte, `next_action`
    devolvería `escalar` una y otra vez y el guion de seguridad se repetiría
    palabra por palabra mientras el paciente siga hablando. El guion solo se
    entrega una vez; el turno después de entregarlo cierra la llamada.
    """
    if state.phase in (Phase.ESCALAMIENTO, Phase.TERMINADA):
        return Action(kind="cerrar", phase=Phase.TERMINADA)

    if escalar:
        return Action(kind="escalar", phase=Phase.ESCALAMIENTO)

    if state.phase is Phase.APERTURA:
        return Action(kind="preguntar", slot=SLOTS[0], phase=Phase.TAMIZAJE)

    if state.phase in (Phase.CIERRE, Phase.TERMINADA):
        return Action(kind="cerrar", phase=Phase.TERMINADA)

    # ¿Se resolvió el slot que estábamos preguntando?
    actual = state.slot_actual
    if actual and getattr(symptoms, SLOT_FIELD[actual]) is None:
        intentos = state.repreguntas.get(actual, 0)
        if intentos < MAX_REPREGUNTAS:
            return Action(kind="repreguntar", slot=actual,
                          phase=Phase.TAMIZAJE, intento=intentos + 1)
        # Agotadas las reformulaciones: se anota como no respondido y se avanza.
        # No se asume normalidad — engine.evaluate(final=True) lo tiene en cuenta.

    pendientes = [s for s in slots_pendientes(state, symptoms) if s != actual]
    if pendientes:
        return Action(kind="preguntar", slot=pendientes[0], phase=Phase.TAMIZAJE)

    if state.phase is not Phase.ABIERTO:
        return Action(kind="preguntar", slot=None, phase=Phase.ABIERTO)
    return Action(kind="cerrar", phase=Phase.CIERRE)


def apply(state: CallState, action: Action, symptoms: Symptoms) -> CallState:
    """Avanza el estado tras ejecutar `action`. Devuelve un estado nuevo."""
    nuevo = CallState(
        phase=action.phase, slot_actual=action.slot,
        repreguntas=dict(state.repreguntas), resueltos=list(state.resueltos),
        sin_responder=list(state.sin_responder), turnos=state.turnos + 1,
    )

    # El slot que se estaba preguntando: o se resolvió, o consume un intento.
    anterior = state.slot_actual
    if anterior:
        if getattr(symptoms, SLOT_FIELD[anterior]) is not None:
            if anterior not in nuevo.resueltos:
                nuevo.resueltos.append(anterior)
            nuevo.repreguntas.pop(anterior, None)
        elif action.kind == "repreguntar" and action.slot == anterior:
            nuevo.repreguntas[anterior] = action.intento
        elif anterior not in nuevo.sin_responder:
            nuevo.sin_responder.append(anterior)

    return nuevo

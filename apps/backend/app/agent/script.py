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

# Turnos seguidos sin sacar un solo dato antes de dejar de insistir con el guion
# y ofrecerle al paciente que lo llame una enfermera. Tres es donde una persona
# ya nota que la conversación no va a ningún lado.
SIN_PROGRESO_OFRECER_SALIDA = 3
# Y a partir de aquí se cierra. No se asume normalidad: `call_ended` enruta a
# `engine.evaluate(final=True)`, que con completitud baja emite
# `no_se_pudo_evaluar` → enfermería prioritaria. Ese es el final correcto de una
# llamada que no se pudo evaluar, y hasta ahora no se alcanzaba nunca.
SIN_PROGRESO_CERRAR = 5


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
    # Slots a los que ya se les hizo su seguimiento. Uno por slot: perseguir dos
    # veces el mismo dato convierte el seguimiento en el interrogatorio que
    # MAX_REPREGUNTAS existe para evitar.
    seguido: list[str] = field(default_factory=list)
    # Detección de atasco. `ultimo_hash` es la frase anterior del paciente
    # normalizada; `sin_progreso` cuenta turnos seguidos que no dejaron ningún
    # dato. En la llamada que motivó esto el paciente repitió la misma frase ocho
    # veces y el guion siguió recorriendo slots como si nada.
    ultimo_hash: str | None = None
    sin_progreso: int = 0

    def to_dict(self) -> dict:
        return {
            "phase": str(self.phase), "slot_actual": self.slot_actual,
            "repreguntas": self.repreguntas, "resueltos": self.resueltos,
            "sin_responder": self.sin_responder, "turnos": self.turnos,
            "seguido": self.seguido, "ultimo_hash": self.ultimo_hash,
            "sin_progreso": self.sin_progreso,
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
            seguido=list(d.get("seguido") or []),
            ultimo_hash=d.get("ultimo_hash"),
            sin_progreso=int(d.get("sin_progreso") or 0),
        )


@dataclass(frozen=True)
class Action:
    """Qué toca hacer en este turno. Es dato, no texto: el fraseo va aparte."""

    kind: str          # preguntar | repreguntar | seguimiento | cerrar | escalar
    slot: str | None = None
    phase: Phase = Phase.TAMIZAJE
    intento: int = 0
    seguimiento: str | None = None   # clave de phrasing.SEGUIMIENTOS


def slots_pendientes(state: CallState, symptoms: Symptoms) -> list[str]:
    """Slots que aún no tienen valor y no se han dado por perdidos."""
    return [
        s for s in SLOTS
        if getattr(symptoms, SLOT_FIELD[s]) is None and s not in state.sin_responder
    ]


def seguimiento_pendiente(state: CallState, symptoms: Symptoms) -> str | None:
    """¿Falta un dato que el paciente empezó a dar y no terminó?

    Hoy solo la fiebre lo necesita, y es el único punto del guion donde perder el
    dato produce un FALSO NEGATIVO: sin la cifra, un 38.5 real no dispara
    `fiebre_38` y el caso rojo se reporta verde. La forma es general por si el
    dataset muestra el mismo patrón en otro slot.

    Devuelve la clase de seguimiento (clave de `phrasing.SEGUIMIENTOS`) o None.
    """
    if symptoms.temperature_c is not None or "fiebre" in state.seguido:
        return None
    # Dijo que se la tomó pero no soltó el número.
    if symptoms.temperature_measured is True:
        return "cifra"
    # Refiere calentura y no sabemos si puede medirla. Si ya dijo que no tiene
    # termómetro (`temperature_measured is False`), no se insiste: es una
    # respuesta legítima y `_fever_reported_unmeasured` cubre el riesgo sin cifra.
    if symptoms.fever is True and symptoms.temperature_measured is None:
        return "termometro"
    return None


def next_action(state: CallState, symptoms: Symptoms, *,
                escalar: bool = False, repetido: bool = False) -> Action:
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

    # --- atasco: cambiar de estrategia, no reformular otra vez ----------------
    # Insistir con otra variante de la misma pregunta a alguien que lleva varios
    # turnos sin dar nada es lo que convierte la llamada en un interrogatorio, y
    # además no funciona: el estilo `evasivo` es el peor del dataset.
    if state.sin_progreso >= SIN_PROGRESO_CERRAR:
        return Action(kind="cerrar", phase=Phase.CIERRE)
    if state.sin_progreso >= SIN_PROGRESO_OFRECER_SALIDA:
        # Conserva `slot_actual` para poder retomar si el paciente dice que sí
        # quiere seguir.
        return Action(kind="ofrecer_salida", slot=state.slot_actual,
                      phase=Phase.TAMIZAJE)

    # El paciente contestó, pero a medias. Va ANTES de la repregunta y de avanzar
    # de slot, y es la diferencia entre las dos cosas: repreguntar es para quien
    # no dijo nada, seguir es para quien empezó a decirlo. "Sí me la tomé" ya es
    # una respuesta —reformularle "¿se ha sentido caliente?" la ignora—, y por eso
    # el seguimiento tampoco gasta MAX_REPREGUNTAS.
    if (clase := seguimiento_pendiente(state, symptoms)) is not None:
        return Action(kind="seguimiento", slot="fiebre",
                      phase=Phase.TAMIZAJE, seguimiento=clase)

    # ¿Se resolvió el slot que estábamos preguntando?
    actual = state.slot_actual
    if actual and getattr(symptoms, SLOT_FIELD[actual]) is None:
        intentos = state.repreguntas.get(actual, 0)
        # `repetido` lo marca el orquestador cuando el paciente dijo exactamente
        # lo mismo que en el turno anterior. Reformular no le va a sacar un dato
        # distinto a quien está repitiendo la misma frase: se abandona el slot
        # —diciéndolo, no en silencio— y se sigue.
        if intentos < MAX_REPREGUNTAS and not repetido:
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


def apply(state: CallState, action: Action, symptoms: Symptoms, *,
          hash_turno: str | None = None, progreso: bool = True) -> CallState:
    """Avanza el estado tras ejecutar `action`. Devuelve un estado nuevo.

    `hash_turno` y `progreso` los calcula el orquestador, que es quien ve el texto
    del paciente y lo que se extrajo de él.
    """
    nuevo = CallState(
        phase=action.phase, slot_actual=action.slot,
        repreguntas=dict(state.repreguntas), resueltos=list(state.resueltos),
        sin_responder=list(state.sin_responder), turnos=state.turnos + 1,
        seguido=list(state.seguido),
        ultimo_hash=hash_turno if hash_turno is not None else state.ultimo_hash,
        sin_progreso=0 if progreso else state.sin_progreso + 1,
    )

    if action.kind == "seguimiento" and action.slot:
        if action.slot not in nuevo.seguido:
            nuevo.seguido.append(action.slot)
        # Un seguimiento no cierra ni abandona el slot: se vuelve a preguntar por
        # el mismo dato, así que la contabilidad de abajo no debe correr.
        return nuevo

    if action.kind == "ofrecer_salida":
        # Tampoco avanza el guion: es una pausa para preguntarle al paciente si
        # quiere seguir. El slot sigue abierto.
        return nuevo

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

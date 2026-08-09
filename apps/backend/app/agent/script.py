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
    # El agente ya dijo todo lo que tenía que decir y espera a que el paciente
    # confirme que entendió y quiere colgar. Una llamada de seguimiento no
    # termina cuando el sistema acaba su lista: termina cuando el paciente se
    # despide. Sin esta fase, tras el guion de seguridad el paciente seguía
    # hablando —incluso aportando señales nuevas, "y la herida se ve rojita"— y
    # el agente le repetía la despedida.
    CONFIRMACION = "confirmacion"
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

# Silencios SEGUIDOS antes de colgar. Tres es la escalera mínima que no cuelga a
# nadie por error: el primero comprueba si sigue en línea (la llamada se pudo
# cortar, o el paciente se levantó), el segundo avisa de que se va a terminar, y
# el tercero cumple el aviso. Colgar al segundo no da margen a alguien mayor que
# tarda en volver al teléfono; esperar a un cuarto alarga una llamada que ya no
# tiene a nadie al otro lado.
#
# Cuenta SEGUIDOS a propósito: cualquier cosa que diga el paciente lo devuelve a
# cero (ver `apply`). Un silencio suelto en mitad de una llamada normal —de los
# que abundan en la capa 2 del dataset— no puede acercar la llamada a colgarse.
MAX_SILENCIOS = 3


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
    # Silencios SEGUIDOS del paciente. Distinto de `sin_progreso`: ahí el paciente
    # habla y no aporta nada; aquí no se le oye. Los desenlaces también son
    # distintos —uno ofrece que lo llame una enfermera, el otro cuelga y alerta—,
    # así que no pueden compartir contador.
    sin_respuesta: int = 0

    def to_dict(self) -> dict:
        return {
            "phase": str(self.phase), "slot_actual": self.slot_actual,
            "repreguntas": self.repreguntas, "resueltos": self.resueltos,
            "sin_responder": self.sin_responder, "turnos": self.turnos,
            "seguido": self.seguido, "ultimo_hash": self.ultimo_hash,
            "sin_progreso": self.sin_progreso, "sin_respuesta": self.sin_respuesta,
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
            sin_respuesta=int(d.get("sin_respuesta") or 0),
        )


@dataclass(frozen=True)
class Action:
    """Qué toca hacer en este turno. Es dato, no texto: el fraseo va aparte."""

    # preguntar | repreguntar | seguimiento | ofrecer_salida | sondear |
    # confirmar | cerrar | escalar
    kind: str
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
                escalar: bool = False, repetido: bool = False,
                emergencia: bool = False, quiere_colgar: bool = False,
                silencio: bool = False) -> Action:
    """Decide el siguiente movimiento. Función pura: se testea sin LLM ni red.

    `escalar` lo impone el motor de decisión desde fuera; cuando llega, corta el
    guion sin importar en qué punto esté. Una emergencia no espera a que terminen
    las preguntas.

    Una vez escalado, el cuadro crítico queda en `symptoms` para siempre —no hay
    forma de "des-escalar" dentro de la llamada—, así que `escalar` seguiría
    siendo True en todos los turnos siguientes. Sin este corte, `next_action`
    devolvería `escalar` una y otra vez y el guion de seguridad se repetiría
    palabra por palabra mientras el paciente siga hablando.

    Entregado el guion, la llamada NO cuelga sola: pasa a `CONFIRMACION` y espera
    a que el paciente diga que entendió (`quiere_colgar`). La excepción es
    `emergencia`: ante una de las seis banderas del 123, retener al paciente en la
    línea compite con la llamada que de verdad importa, así que ahí sí se cierra
    en el turno siguiente.
    """
    if state.phase is Phase.TERMINADA:
        return Action(kind="cerrar", phase=Phase.TERMINADA)

    # Nadie contesta. Va antes que todo lo demás porque ninguna otra rama tiene
    # sentido sin un interlocutor: repreguntar, escalar o despedirse presuponen
    # que alguien está oyendo. Se conserva `slot_actual` y la fase para poder
    # retomar la llamada donde estaba si el paciente vuelve.
    if silencio:
        if state.sin_respuesta + 1 >= MAX_SILENCIOS:
            return Action(kind="cerrar", phase=Phase.TERMINADA)
        return Action(kind="sondear", slot=state.slot_actual, phase=state.phase,
                      intento=state.sin_respuesta + 1)

    # El paciente da la llamada por terminada, esté donde esté el guion. Va antes
    # que todo lo demás: seguir preguntando a quien acaba de decir "chao" es lo
    # contrario de escuchar. La política de incertidumbre se aplica igual al
    # cerrar (`engine.evaluate(final=True)`), así que colgar aquí no pierde el
    # cuadro incompleto — lo escala.
    if quiere_colgar and state.phase is not Phase.ESCALAMIENTO:
        return Action(kind="cerrar", phase=Phase.CIERRE)

    # Guion de seguridad ya entregado. `quiere_colgar` aquí también: el paciente
    # puede despedirse en el mismo turno que sigue al guion ("ya entendí, chao"),
    # y hacerle una pregunta de confirmación después de eso es no escucharlo.
    if state.phase is Phase.ESCALAMIENTO:
        if emergencia or quiere_colgar or state.sin_progreso >= SIN_PROGRESO_CERRAR:
            return Action(kind="cerrar", phase=Phase.TERMINADA)
        return Action(kind="confirmar", phase=Phase.CONFIRMACION)

    if state.phase is Phase.CONFIRMACION:
        if quiere_colgar or state.sin_progreso >= SIN_PROGRESO_CERRAR:
            return Action(kind="cerrar", phase=Phase.TERMINADA)
        return Action(kind="confirmar", phase=Phase.CONFIRMACION)

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
    # Se acabó el guion, pero no necesariamente la llamada: si el paciente aún
    # tiene dudas, se le atienden. Solo se cuelga cuando lo dice él, o cuando
    # lleva varios turnos sin aportar nada (el tope que protege la demo en vivo).
    if quiere_colgar or state.sin_progreso >= SIN_PROGRESO_CERRAR:
        return Action(kind="cerrar", phase=Phase.CIERRE)
    return Action(kind="confirmar", phase=Phase.CONFIRMACION)


def apply(state: CallState, action: Action, symptoms: Symptoms, *,
          hash_turno: str | None = None, progreso: bool = True,
          intento_real: bool = True, silencio: bool = False) -> CallState:
    """Avanza el estado tras ejecutar `action`. Devuelve un estado nuevo.

    `hash_turno` y `progreso` los calcula el orquestador, que es quien ve el texto
    del paciente y lo que se extrajo de él.

    `intento_real=False` cuando el paciente ni siquiera intentó contestar —saludó,
    pidió que le repitieran la pregunta, o dijo algo social—. Esos turnos no
    consumen `MAX_REPREGUNTAS`: reformular es para quien esquivó la pregunta, no
    para quien todavía no la ha oído. Sin esto, "Hola, buenas." se comía uno de
    los dos intentos del slot de dolor y el agente le respondía a un saludo con la
    reformulación cerrada ("¿más cerca de tres o de ocho?").
    """
    nuevo = CallState(
        phase=action.phase, slot_actual=action.slot,
        repreguntas=dict(state.repreguntas), resueltos=list(state.resueltos),
        sin_responder=list(state.sin_responder), turnos=state.turnos + 1,
        seguido=list(state.seguido),
        ultimo_hash=hash_turno if hash_turno is not None else state.ultimo_hash,
        sin_progreso=0 if progreso else state.sin_progreso + 1,
        # Cualquier cosa que diga el paciente devuelve el contador a cero: la
        # escalera de silencio mide silencios SEGUIDOS, no acumulados.
        sin_respuesta=state.sin_respuesta + 1 if silencio else 0,
    )

    if action.kind == "sondear":
        # Comprobar si el paciente sigue en línea no avanza el guion ni gasta
        # `MAX_REPREGUNTAS`: nadie esquivó la pregunta, simplemente no llegó a
        # oírla. El slot sigue abierto y se vuelve a preguntar entero.
        return nuevo

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

    if action.kind == "confirmar":
        # La llamada está esperando al paciente, no recorriendo el guion: no hay
        # slot que resolver ni que dar por perdido.
        return nuevo

    # El slot que se estaba preguntando: o se resolvió, o consume un intento.
    anterior = state.slot_actual
    if anterior:
        if getattr(symptoms, SLOT_FIELD[anterior]) is not None:
            if anterior not in nuevo.resueltos:
                nuevo.resueltos.append(anterior)
            nuevo.repreguntas.pop(anterior, None)
        elif action.kind == "repreguntar" and action.slot == anterior:
            if intento_real:
                nuevo.repreguntas[anterior] = action.intento
        elif anterior not in nuevo.sin_responder:
            nuevo.sin_responder.append(anterior)

    return nuevo

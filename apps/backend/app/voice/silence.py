"""Escalera de silencios y eventos de voz — la parte PURA del manejo de silencio.

Aquí vive la decisión de QUÉ toca cuando el reloj de inactividad vence; el reloj
mismo (asyncio, deadline monótono) sigue en `pipeline.py::ClinicalProcessor`,
que ya está blindado por tests. Separar la decisión del timing es lo que permite
probar la escalera con llamadas directas, sin frames ni event loop.

La escalera tiene dos clases de escalón, y la diferencia importa:

  - GENTLE: una frase suave ("tómese su tiempo") que se emite LOCALMENTE en la
    capa de voz. No pasa por el orquestador, no crea `Turn` en la base, no toca
    el contador de cierre. Es un permiso para pensar, no una comprobación de
    presencia: que acercara la llamada al cuelgue contradiría su propio texto.
    Suena UNA vez por episodio de silencio — después de un "¿sigue ahí?", un
    "tómese su tiempo" sería incoherente.
  - SONDEO: se inyecta "[silencio]" al orquestador como un turno más, y la
    escalera de `agent/script.py` (sondeo → aviso → cierre, contador
    `sin_respuesta`) decide qué se dice y cuándo se cuelga. Ese camino ya
    existía y no cambia.

Un silencio JAMÁS se interpreta clínicamente: el slot preguntado queda en None
(UNKNOWN), nunca en False. Eso lo garantizan el cortocircuito del orquestador
(intent "silencio" → extracción vacía) y el diseño ternario de `Symptoms`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import StrEnum

from loguru import logger


@dataclass(frozen=True)
class SilenceConfig:
    """Tiempos de la escalera. Se construye desde `Settings`; inyectable en tests.

    `gentle_s == 0` desactiva la frase suave: el primer vencimiento va directo
    al sondeo, que es el comportamiento que tenía el sistema antes.
    """

    initial_s: float = 6.0
    gentle_s: float = 6.0
    repeat_s: float = 8.0

    @classmethod
    def from_settings(cls, s) -> "SilenceConfig":
        return cls(
            initial_s=s.silence_initial_s,
            gentle_s=s.silence_gentle_s,
            repeat_s=s.silence_repeat_s,
        )


class Escalon(StrEnum):
    """Qué toca hacer cuando el reloj vence."""

    GENTLE = "gentle"   # frase suave local, sin orquestador ni contadores
    SONDEO = "sondeo"   # inyectar "[silencio]": el guion decide (sondeo/aviso/cierre)


class SilenceLadder:
    """Máquina de estados pura de la escalera. Sin asyncio, sin pipecat.

    El contrato con el reloj de `ClinicalProcessor` es de dos llamadas:
    `siguiente_espera()` dice cuánto dormir, y `al_vencer()` dice qué hacer al
    despertar (y avanza la escalera). `reset()` vuelve al tramo inicial cuando
    el paciente aporta un turno real — mismo criterio con el que `script.apply`
    resetea `sin_respuesta`.
    """

    def __init__(self, config: SilenceConfig | None = None) -> None:
        self._config = config or SilenceConfig()
        self._gentle_emitido = False
        self._sondeos = 0
        self._inicio_episodio: float | None = None

    @property
    def sondeos(self) -> int:
        """Sondeos emitidos en el episodio actual (para eventos/logs)."""
        return self._sondeos

    def siguiente_espera(self) -> float:
        """Cuánto debe dormir el reloj antes del próximo escalón."""
        if self._gentle_emitido and self._sondeos == 0:
            return self._config.gentle_s
        if self._sondeos > 0:
            return self._config.repeat_s
        return self._config.initial_s

    def al_vencer(self) -> Escalon:
        """El reloj venció: decide el escalón y avanza la escalera."""
        if self._inicio_episodio is None:
            self._inicio_episodio = time.monotonic()
        if not self._gentle_emitido and self._config.gentle_s > 0 and self._sondeos == 0:
            self._gentle_emitido = True
            return Escalon.GENTLE
        self._sondeos += 1
        return Escalon.SONDEO

    def marca_inicio(self) -> None:
        """Ancla el arranque del episodio al momento en que empieza el silencio.

        La llama el reloj al armarse por primera vez en un episodio; si nadie la
        llama, `al_vencer` ancla al primer vencimiento (peor resolución, nunca
        incorrecto).
        """
        if self._inicio_episodio is None:
            self._inicio_episodio = time.monotonic()

    def en_episodio(self) -> bool:
        return self._inicio_episodio is not None

    def duracion_ms(self) -> int:
        """Milisegundos desde que empezó el episodio de silencio actual."""
        if self._inicio_episodio is None:
            return 0
        return int((time.monotonic() - self._inicio_episodio) * 1000)

    def reset(self) -> None:
        """Un turno real del paciente: episodio nuevo, gentle re-habilitado."""
        self._gentle_emitido = False
        self._sondeos = 0
        self._inicio_episodio = None


# --- eventos de voz --------------------------------------------------------
# Registro estructurado de lo que pasó en la llamada a nivel de VOZ (quién
# habló, quién calló, quién interrumpió), separado a propósito de los estados
# clínicos: un evento de voz nunca es un dato del paciente. Con estos eventos
# en el log se puede reconstruir una conversación problemática sin adivinar.


class VoiceEvent(StrEnum):
    PATIENT_SILENCE_STARTED = "PATIENT_SILENCE_STARTED"
    PATIENT_SILENCE_CONTINUED = "PATIENT_SILENCE_CONTINUED"
    PATIENT_SPEECH_DETECTED = "PATIENT_SPEECH_DETECTED"
    PATIENT_NO_RESPONSE = "PATIENT_NO_RESPONSE"
    PATIENT_INTERRUPTED_AGENT = "PATIENT_INTERRUPTED_AGENT"
    AGENT_INTERRUPTED = "AGENT_INTERRUPTED"
    CONNECTION_LOST = "CONNECTION_LOST"
    SILENCE_PROMPT_TRIGGERED = "SILENCE_PROMPT_TRIGGERED"
    # Una transcripción se descartó por ser el eco del propio agente. Desde
    # fuera un descarte de estos es invisible —el paciente habló y "no pasó
    # nada"— y si el filtro se equivoca, es la antesala de la escalera de
    # silencios y del cuelgue: tiene que quedar rastro auditable.
    ECHO_DISCARDED = "ECHO_DISCARDED"


def emit(
    event: VoiceEvent,
    *,
    conversation_id: str | None = None,
    duration_ms: int | None = None,
    attempt: int | None = None,
    stage: str | None = None,
    phase: str | None = None,
    slot: str | None = None,
) -> None:
    """Emite un evento de voz al log con campos estructurados.

    Van en `record.extra` vía `logger.bind`, así que con un sink
    `serialize=True` salen como JSON y sin él siguen visibles en el mensaje.
    Nada de esto toca la base de datos: los turnos de silencio "duros" ya
    persisten como `Turn(intent="silencio")` y eso basta para el reporte; esto
    es para depurar la llamada.
    """
    campos = {
        "voice_event": str(event),
        "conversation_id": conversation_id,
        "duration_ms": duration_ms,
        "attempt": attempt,
        "stage": stage,
        "current_script_state": phase,
        "slot": slot,
    }
    presentes = {k: v for k, v in campos.items() if v is not None}
    detalle = " ".join(f"{k}={v}" for k, v in presentes.items() if k != "voice_event")
    logger.bind(**presentes).info(f"[voz:evento] {event}" + (f" · {detalle}" if detalle else ""))

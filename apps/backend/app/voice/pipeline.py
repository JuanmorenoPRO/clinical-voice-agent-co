"""Pipeline de voz Pipecat (STT → lógica clínica → TTS) — ADR-003.

Cadena por componentes en streaming, transporte WebRTC al navegador (sin telefonía
real). El MISMO orquestador del modo texto lleva la lógica del turno; aquí solo se
enchufan la entrada y la salida de audio.

    micrófono ──WebRTC──▶ Silero VAD ──▶ Groq Whisper ──▶ orquestador ──▶ TTS ──WebRTC──▶ altavoz

Todos los servicios son de primera parte de Pipecat, así que no hay que escribir
ningún `STTService`/`TTSService` a medida. Detalles que no son obvios:

  - `GroqSTTService` hereda de `BaseWhisperSTTService`, que trabaja por segmentos:
    **exige un VAD en el transporte**. Sin `SileroVADAnalyzer` no transcribe nada.
  - **El VAD no se pasa como parámetro del transporte.** `TransportParams` no
    declara ningún campo `vad_analyzer`, y Pydantic descarta ese kwarg en
    silencio sin avisar: el resultado es una llamada que se conecta y hasta
    saluda, pero jamás detecta que el paciente terminó de hablar. Es un
    `FrameProcessor` propio (`VADProcessor`) que va explícito en la cadena.
  - **TTS por defecto: Piper, no Kokoro.** Piper entrena un modelo por idioma;
    `es_MX-claude-high` además resultó 5× más rápido en caliente que Kokoro. Por
    defecto se prefiere la voz nativa de Piper. Kokoro se mantiene disponible con
    `TTS_PROVIDER=kokoro`; para que su español no quede en fonemización por
    defecto en inglés (acento anglosajón), se le pasa `language=Language.ES`,
    que es el mismo G2P español (misaki/espeak-ng) que usa el proyecto
    `leonelhs/kokoro-tts-spanish`. Ver `TTS_PROVIDER` en `config.py` y la
    medición en `docs/spikes-7-agosto.md`.
  - Nada de esto arrastra PyTorch. `pipecat-ai[kokoro]` depende de `kokoro-onnx`
    y `piper-tts` solo de `onnxruntime`; Silero corre sobre el mismo runtime.

⚠️ Importa Pipecat a nivel de módulo, así que SOLO debe importarse de forma
perezosa (desde el router de voz), para que la app arranque en modo texto sin las
dependencias de voz instaladas.
"""

from __future__ import annotations

import asyncio
import time
from functools import partial
from pathlib import Path

from loguru import logger
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    CancelFrame,
    EndFrame,
    Frame,
    TranscriptionFrame,
    TTSSpeakFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.audio.vad_processor import VADProcessor
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.groq.stt import GroqSTTService
from pipecat.transcriptions.language import Language
from pipecat.transports.base_transport import TransportParams
from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport
from pipecat.turns.user_start.vad_user_turn_start_strategy import (
    VADUserTurnStartStrategy,
)
from pipecat.turns.user_stop.speech_timeout_user_turn_stop_strategy import (
    SpeechTimeoutUserTurnStopStrategy,
)
from pipecat.turns.user_turn_processor import UserTurnProcessor
from pipecat.turns.user_turn_strategies import UserTurnStrategies
from pipecat.workers.runner import WorkerRunner

from ..agent import phrasing
from ..agent.orchestrator import abandon_conversation, ensure_conversation, process_turn
from ..agent.phrasing import APERTURA
from ..nlu import intent as intent_nlu
from ..nlu import lexicon
from ..schemas import TurnResponse
from ..config import get_settings
from ..db import SessionLocal
from .silence import Escalon, SilenceConfig, SilenceLadder, VoiceEvent, emit

# Sesga la transcripción hacia el dominio clínico. Es un truco barato y efectivo:
# sin él, Whisper convierte "eritema" en "he tema" y "purulenta" en "prudente".
_PROMPT_STT = (
    "Llamada de seguimiento postoperatorio con un paciente colombiano. "
    "Términos frecuentes: dolor, fiebre, herida quirúrgica, secreción, purulenta, "
    "eritema, movilidad, apetito, sueño, cicatriz, puntos, analgésico."
)


# Solapamiento de palabras a partir del cual una transcripción se considera eco de
# lo que acaba de decir el agente. Alto a propósito: descartar por error un turno
# real del paciente es peor que dejar pasar un eco, que como mucho produce una
# respuesta rara. 0.6 deja fuera las coincidencias de cortesía ("gracias", "sí")
# y atrapa las repeticiones literales, que es como suena un eco de verdad.
_UMBRAL_ECO = 0.6
# Por debajo de esto no se juzga: "sí", "no", "un cuatro" son respuestas legítimas
# del paciente que pueden solaparse por casualidad con el vocabulario del agente.
_MIN_PALABRAS_ECO = 4


def es_eco(transcripcion: str, ultima_respuesta: str,
           min_palabras: int = _MIN_PALABRAS_ECO) -> bool:
    """¿Esto que 'dijo el paciente' es en realidad la voz del agente rebotada?

    Ocurre cuando el paciente está en altavoz o el navegador no cancela el eco: el
    STT transcribe al propio agente, el orquestador lo toma por un turno y contesta,
    y la llamada se convierte en el agente hablando solo sin dejar meter baza.

    Se compara por solapamiento de vocabulario y no por igualdad: el STT nunca
    devuelve la frase literal —se come palabras, parte por la mitad— así que
    comparar cadenas no detectaría nada. Se normaliza con `lexicon.normalize`, que
    es la función que este repo ya usa para comparar habla.

    `min_palabras` existe por el barge-in: mientras el agente HABLA, un fragmento
    de eco de dos o tres palabras que con el umbral normal pasaría de largo le
    cortaría el TTS a media frase. En ese estado el filtro se endurece
    (`BARGE_IN_MIN_PALABRAS_ECO`); con el agente callado se mantiene el umbral
    alto, porque descartar por error un turno real del paciente es peor.
    """
    if not ultima_respuesta:
        return False
    dichas = set(lexicon.normalize(transcripcion).split())
    if len(dichas) < min_palabras:
        return False
    agente = set(lexicon.normalize(ultima_respuesta).split())
    if not agente:
        return False
    return len(dichas & agente) / len(dichas) >= _UMBRAL_ECO


# Ritmo de habla de Piper (es, con `length_scale=1.08`), para estimar cuántas
# palabras alcanzaron a sonar antes de una interrupción. Pipecat no reporta
# upstream el texto reproducido (`TTSTextFrame` muere en el transporte de
# salida), así que se estima por tiempo. Calibrable con `scripts/spike_voice.py`.
_PALABRAS_POR_SEGUNDO = 2.6
# El margen SUMA a propósito: cubre el audio en vuelo (cola del transporte,
# jitter del navegador) y errar por exceso solo mantiene la protección anti-eco;
# errar por defecto reintroduciría los falsos positivos que esto arregla.
_MARGEN_PALABRAS = 4


def prefijo_pronunciado(texto: str, segundos: float) -> str:
    """Las palabras de `texto` que alcanzaron a sonar en `segundos` de TTS.

    Tras un barge-in, `_ultima_respuesta` guardaba la frase COMPLETA aunque el
    paciente la cortara en la palabra tres: el anti-eco comparaba los turnos
    siguientes contra vocabulario que jamás sonó y los descartaba como eco —
    y de ahí a la escalera de silencios y al cuelgue.
    """
    if segundos <= 0:
        return ""
    n = int(segundos * _PALABRAS_POR_SEGUNDO) + _MARGEN_PALABRAS
    return " ".join(texto.split()[:n])


class ClinicalProcessor(FrameProcessor):
    """Puente entre el STT y el TTS.

    Cada transcripción final del paciente pasa por la misma lógica clínica que el
    modo texto, y su respuesta se envía a voz. Mantiene el `conversation_id` para
    conservar el contexto de toda la llamada.

    Además marca el instante en que el paciente deja de hablar, que es el origen
    exacto que la rúbrica pide para medir la latencia: desde el fin del habla
    hasta que empieza a sonar el audio del agente.
    """

    # Marcador con el que un silencio entra al orquestador como un turno más. Es
    # el mismo que usa la capa 2 del dataset (`tests/dataset/dialogs.jsonl`), así
    # que la escalera de "¿sigue ahí?" se prueba en texto sin tocar el audio.
    SILENCIO = "[silencio]"

    def __init__(self, patient_id: str | None = None,
                 silence_config: SilenceConfig | None = None) -> None:
        super().__init__()
        s = get_settings()
        self._conversation_id: str | None = None
        self._patient_id = patient_id
        self._t_fin_habla: float | None = None
        # La escalera decide QUÉ toca al vencer el reloj (frase suave o sondeo)
        # y cuánto esperar hasta el siguiente escalón. El reloj de aquí solo
        # pone el tiempo; la decisión vive en `voice/silence.py`, que es pura.
        self._escalera = SilenceLadder(silence_config or SilenceConfig.from_settings(s))
        self._barge_in = s.barge_in_enabled
        self._min_palabras_eco_hablando = s.barge_in_min_palabras_eco
        self._watchdog: asyncio.Task | None = None
        self._terminado = False
        # La llamada terminó por su cauce (despedida, escalera de silencios,
        # escalamiento) y no por caerse la conexión. Lo consulta
        # `on_client_disconnected` para distinguir NO_RESPONSE de
        # CONNECTION_LOST: el transporte se desconecta igual en ambos casos.
        self._cierre_normal = False
        # Estado explícito de quién tiene la palabra. Estaba implícito en si
        # existía o no la tarea del reloj, y esa ambigüedad ES el bug que este
        # módulo tuvo: `BotStoppedSpeakingFrame` significa dos cosas —"terminé de
        # hablar" y "me interrumpieron"— y sin estos flags no se pueden separar.
        self._agente_hablando = False
        self._paciente_hablando = False
        # Hay un turno en vuelo contra el orquestador. Con `LLM_PROVIDER=groq` eso
        # es una llamada de red, y el reloj no puede vencer mientras se está
        # generando la respuesta que el paciente todavía no ha oído.
        self._procesando = False
        # `_procesando` con conteo: un sondeo de silencio y una transcripción
        # pueden solaparse (el paciente contesta justo cuando vence el reloj), y
        # con un booleano pelado el primero en terminar apagaba la marca del que
        # seguía en vuelo.
        self._turnos_en_vuelo = 0
        # Los turnos corren en tarea propia serializados por este lock, no
        # inline en `process_frame`: un `InterruptionFrame` (barge-in) cancela y
        # recrea la tarea de proceso del procesador, y eso mataría el `await`
        # con la base de datos ya avanzada — el guion quedaría un paso por
        # delante de lo que el paciente oyó.
        self._turno_lock = asyncio.Lock()
        self._tarea_turno: asyncio.Task | None = None
        # Rotación local de las frases suaves de silencio (no pasan por el
        # orquestador, así que la ventana anti-repetición de allí no las ve).
        self._suaves_usadas: list[str] = []
        self._n_suaves = 0
        # Fase y slot del último turno procesado, para que los eventos de voz
        # digan en qué punto del guion pasó cada cosa.
        self._ultimo_phase: str | None = None
        self._ultimo_slot: str | None = None
        # Cuántas veces el VAD ha detectado voz del paciente en toda la llamada.
        # Sirve para una sola cosa, pero importante: distinguir "el paciente no
        # contesta" de "no le estamos oyendo", que desde los logs se ven idénticos
        # y llevan a arreglar cosas distintas.
        self._veces_que_oimos_al_paciente = 0
        # Lo último que dijo el agente, para descartar su propio eco.
        self._ultima_respuesta = ""
        self._deadline = 0.0
        # Cuándo arrancó el TTS en curso, para estimar cuánto texto llegó a
        # sonar si el paciente interrumpe (ver `prefijo_pronunciado`).
        self._t_bot_arranco: float | None = None
        # El paciente cortó al agente a media frase: la pregunta persistida no
        # se oyó entera. Viaja con el turno (kwarg) hasta el orquestador para
        # que un "sí/no" pelado no se lea contra una pregunta no escuchada.
        self._pregunta_interrumpida = False

    # --- vigilancia de inactividad -------------------------------------------
    # El VAD detecta cuándo el paciente DEJA de hablar, pero no tiene nada que
    # decir sobre alguien que no ha empezado nunca: sin este reloj, un paciente
    # que suelta el teléfono deja la llamada abierta para siempre y nadie se
    # entera. Es la única pieza que no puede vivir en `agent/script.py`, porque la
    # decisión la dispara el paso del tiempo y no un turno del paciente.
    #
    # La regla es UNA y conviene leerla entera: **el reloj corre solo cuando no
    # habla ninguno de los dos y no hay un turno en vuelo**. La versión anterior
    # lo armaba con cualquier `BotStoppedSpeakingFrame`, y Pipecat emite ese frame
    # también al interrumpir al agente (`transports/base_output.py`,
    # `handle_interruptions`). Resultado medido: el paciente empezaba a hablar
    # —lo que cancelaba el reloj—, su propia interrupción lo rearmaba acto
    # seguido, y a los pocos segundos el agente le soltaba "¿Sigue ahí? No le
    # escuché nada." por encima de su respuesta.

    def _puede_vigilar(self) -> bool:
        return not (self._terminado or self._agente_hablando
                    or self._paciente_hablando or self._procesando)

    def _armar_vigilancia(self) -> None:
        """(Re)arranca la cuenta atrás si no hay nadie hablando."""
        self._cancelar_vigilancia()
        if not self._puede_vigilar():
            return
        if not self._escalera.en_episodio():
            self._escalera.marca_inicio()
            emit(VoiceEvent.PATIENT_SILENCE_STARTED,
                 conversation_id=self._conversation_id,
                 phase=self._ultimo_phase, slot=self._ultimo_slot)
        self._deadline = time.monotonic() + self._escalera.siguiente_espera()
        self._watchdog = asyncio.create_task(self._vigilar())

    def _cancelar_vigilancia(self) -> None:
        if self._watchdog is not None:
            self._watchdog.cancel()
            self._watchdog = None

    async def _vigilar(self) -> None:
        # Deadline monótono y re-comprobado al despertar, en vez de un `sleep`
        # de una sola vez. Es lo que hace estructuralmente imposible el síntoma
        # reportado —la escalera de silencio entera de corrido, como si el tiempo
        # ya hubiera pasado—: aunque una tormenta de frames rearme el reloj varias
        # veces, ninguna despierta antes de tiempo y solo puede salir UN turno de
        # silencio por vencimiento.
        try:
            while (restante := self._deadline - time.monotonic()) > 0:
                await asyncio.sleep(restante)
        except asyncio.CancelledError:
            return
        if not self._puede_vigilar():
            # Algo cambió mientras dormía (el paciente arrancó a hablar, entró un
            # turno). No es un silencio.
            return
        escalon = self._escalera.al_vencer()

        if escalon is Escalon.GENTLE:
            # Primer escalón: una frase suave, LOCAL. No pasa por el orquestador
            # ni por la base —no es un turno—, no toca el contador de cierre y no
            # repite la pregunta: es permiso para pensar, no una comprobación de
            # presencia. Suena una vez por episodio de silencio.
            self._n_suaves += 1
            semilla = f"{self._conversation_id or 'sin-conv'}:{self._n_suaves}"
            frase = phrasing.silencio_suave(semilla, self._suaves_usadas)
            self._suaves_usadas.append(frase)
            emit(VoiceEvent.SILENCE_PROMPT_TRIGGERED,
                 conversation_id=self._conversation_id, stage="gentle",
                 duration_ms=self._escalera.duracion_ms(),
                 phase=self._ultimo_phase, slot=self._ultimo_slot)
            logger.info(f"[voz] agente (pausa suave): {frase}")
            # Imprescindible para que `es_eco` descarte el rebote del propio
            # gentle si vuelve por el micrófono.
            self._ultima_respuesta = frase
            await self.push_frame(TTSSpeakFrame(frase))
            # El reloj se rearma cuando esta frase termine de sonar
            # (`BotStoppedSpeakingFrame`), como cualquier otro turno del agente.
            return

        # SONDEO: el silencio entra al orquestador como "[silencio]" y la
        # escalera de `agent/script.py` decide (sondear → avisar → colgar).
        intento = self._escalera.sondeos
        logger.info(f"[voz] silencio continuado — sondeo {intento}")
        emit(VoiceEvent.PATIENT_SILENCE_CONTINUED,
             conversation_id=self._conversation_id, attempt=intento,
             duration_ms=self._escalera.duracion_ms(),
             phase=self._ultimo_phase, slot=self._ultimo_slot)
        if self._veces_que_oimos_al_paciente == 0:
            # La traza que faltaba. Un paciente callado y un micrófono que no
            # llega al umbral del VAD producen exactamente la misma escalera de
            # "¿sigue ahí?" y el mismo cuelgue, y se arreglan en sitios opuestos.
            logger.error(
                "[voz] el reloj venció y NUNCA se ha detectado voz del paciente en "
                "esta llamada. Esto no es un paciente callado: es que su audio no "
                "está llegando al VAD. Revise el micrófono del navegador y baje "
                "VAD_MIN_VOLUME en .env (pruebe 0.15)."
            )
        self._turno_arranca()
        async with self._turno_lock:
            try:
                result = await asyncio.to_thread(self._run_turn, self.SILENCIO)
            finally:
                self._turno_termina()
        if self._terminado:
            return
        if self._paciente_hablando:
            # El paciente arrancó a hablar mientras se generaba el sondeo (el
            # turno toca base de datos y puede tocar red): no se le habla
            # encima. Su turno real viene en camino y devuelve el contador
            # `sin_respuesta` a cero (`script.apply`), así que el turno de
            # silencio que quedó anotado no acerca la llamada al cuelgue.
            logger.info("[voz] sondeo descartado: el paciente empezó a hablar")
            return
        emit(VoiceEvent.SILENCE_PROMPT_TRIGGERED,
             conversation_id=self._conversation_id,
             stage="cierre" if result.call_ended else ("sondeo" if intento == 1 else "aviso"),
             attempt=intento, duration_ms=self._escalera.duracion_ms(),
             phase=self._ultimo_phase, slot=self._ultimo_slot)
        logger.info(f"[voz] agente: {result.response}")
        self._ultima_respuesta = result.response
        await self.push_frame(TTSSpeakFrame(result.response))
        if result.call_ended:
            # Se agotó la escalera de `agent/script.py`: ya se avisó de que la
            # llamada iba a terminar y aquí se cumple. `EndFrame` es de control,
            # así que cuelga después de que suene la despedida.
            self._terminado = True
            self._cierre_normal = True
            emit(VoiceEvent.PATIENT_NO_RESPONSE,
                 conversation_id=self._conversation_id, attempt=intento,
                 duration_ms=self._escalera.duracion_ms(),
                 phase=self._ultimo_phase, slot=self._ultimo_slot)
            logger.info("[voz] llamada cerrada: el paciente dejó de contestar")
            await self.push_frame(EndFrame(reason="sin_respuesta"))
        # Si no terminó, el reloj se rearma solo cuando acabe este TTS
        # (`BotStoppedSpeakingFrame`, más abajo).

    # --- turnos contra el orquestador -----------------------------------------

    def _turno_arranca(self) -> None:
        self._turnos_en_vuelo += 1
        self._procesando = True

    def _turno_termina(self) -> None:
        self._turnos_en_vuelo = max(0, self._turnos_en_vuelo - 1)
        if self._turnos_en_vuelo == 0:
            self._procesando = False

    def _registrar_interrupcion(self) -> None:
        """El paciente cortó al agente: deja el estado acorde a lo que SONÓ.

        Idempotente (la interrupción confirmada y el `BotStoppedSpeakingFrame`
        que provoca pueden llegar los dos): la primera llamada trunca
        `_ultima_respuesta` a lo que alcanzó a pronunciarse y marca el turno
        que viene como `pregunta_interrumpida`; las siguientes no hacen nada.
        """
        if self._t_bot_arranco is None:
            return
        segundos = time.monotonic() - self._t_bot_arranco
        self._t_bot_arranco = None
        self._pregunta_interrumpida = True
        self._ultima_respuesta = prefijo_pronunciado(self._ultima_respuesta, segundos)

    def _run_turn(self, text: str, interrumpida: bool = False) -> TurnResponse:
        session = SessionLocal()
        try:
            # El id se fija ANTES de procesar. Si el turno revienta a mitad
            # (red del LLM caída, etc.) con el id todavía en None, el turno
            # siguiente crearía una conversación NUEVA y perdería todo el
            # contexto acumulado: síntomas, fase y slot del guion.
            if self._conversation_id is None:
                self._conversation_id = ensure_conversation(
                    session, None, self._patient_id)
            result = process_turn(
                session,
                text=text,
                conversation_id=self._conversation_id,
                patient_id=self._patient_id,
                pregunta_interrumpida=interrumpida,
            )
            self._conversation_id = result.conversation_id
            self._ultimo_phase = result.phase
            self._ultimo_slot = result.slot_actual
            return result
        finally:
            session.close()

    async def _esperar_a_que_pare_el_paciente(self, max_s: float = 15.0) -> None:
        """No se le habla encima a quien está hablando.

        Con tope: si el VAD se queda pegado en "hablando" (micrófono con ruido
        continuo), la respuesta sale igual pasado el margen — una llamada muda
        para siempre es peor que un solape puntual.
        """
        t0 = time.monotonic()
        while (self._paciente_hablando and not self._terminado
               and time.monotonic() - t0 < max_s):
            await asyncio.sleep(0.1)

    async def _turno_y_responder(self, text: str, interrumpida: bool = False) -> None:
        """Procesa un turno del paciente y pone voz a la respuesta.

        Corre como tarea propia (no inline en `process_frame`) y serializada por
        `_turno_lock`: un `InterruptionFrame` de barge-in cancela y recrea la
        tarea de proceso del procesador, y un turno inline moriría a medio
        `await` con la base de datos ya avanzada — la respuesta jamás sonaría y
        el guion quedaría un paso por delante del paciente.
        """
        try:
            async with self._turno_lock:
                try:
                    result = await asyncio.to_thread(self._run_turn, text, interrumpida)
                finally:
                    self._turno_termina()
                if self._terminado:
                    return
                if self._t_fin_habla is not None:
                    ms = (time.perf_counter() - self._t_fin_habla) * 1000
                    logger.info(f"[voz] latencia fin-de-habla → respuesta: {ms:.0f} ms")
                # Si mientras se generaba la respuesta el paciente volvió a
                # hablar, se espera a que pare antes de contestarle.
                await self._esperar_a_que_pare_el_paciente()
                logger.info(f"[voz] agente: {result.response}")
                self._ultima_respuesta = result.response
                await self.push_frame(TTSSpeakFrame(result.response))
                if result.call_ended:
                    # Segundo turno tras un escalamiento crítico, o despedida:
                    # `EndFrame` es de control, así que el cierre suena entero
                    # antes de colgar.
                    logger.info("[voz] llamada cerrada tras el turno de despedida")
                    self._terminado = True
                    self._cierre_normal = True
                    await self.push_frame(EndFrame(reason="cierre_por_escalamiento"))
        except asyncio.CancelledError:
            raise
        except Exception:
            # En una tarea suelta la excepción se perdería en silencio
            # ("Task exception was never retrieved") y la llamada quedaría muda.
            logger.exception("[voz] el turno falló; se rearma el reloj")
            if not self._terminado:
                # El paciente habló y no puede quedarse con un teléfono mudo:
                # se le pide repetir, y `_ultima_respuesta` se actualiza para
                # que el anti-eco compare contra lo que de verdad sonó.
                self._ultima_respuesta = phrasing.FALLO_TECNICO
                await self.push_frame(TTSSpeakFrame(phrasing.FALLO_TECNICO))
            self._armar_vigilancia()

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)

        if isinstance(frame, BotStartedSpeakingFrame):
            # Mientras suena la pregunta el reloj no corre. Contar desde antes lo
            # haría vencer con el paciente todavía escuchando.
            self._agente_hablando = True
            self._t_bot_arranco = time.monotonic()
            self._cancelar_vigilancia()
        elif isinstance(frame, BotStoppedSpeakingFrame):
            # OJO: este frame llega por DOS motivos distintos y solo uno de ellos
            # inicia un silencio. El otro es que el paciente acaba de interrumpir
            # al agente (`handle_interruptions` en el transporte de Pipecat lo
            # emite), y ahí el reloj no puede arrancar: el paciente está hablando
            # justo ahora. Ese era el bug. Llega UPSTREAM desde `transport.output()`.
            if self._paciente_hablando:
                emit(VoiceEvent.AGENT_INTERRUPTED,
                     conversation_id=self._conversation_id,
                     phase=self._ultimo_phase, slot=self._ultimo_slot)
                # Cubre el modo BARGE_IN_VAD, donde el corte lo dispara el
                # `UserTurnProcessor` sin pasar por la rama de transcripción.
                self._registrar_interrupcion()
            self._agente_hablando = False
            self._t_bot_arranco = None    # terminó de hablar: sonó entera
            self._armar_vigilancia()      # `_puede_vigilar` filtra el caso malo
        elif isinstance(frame, EndFrame):
            self._terminado = True
            self._cierre_normal = True    # un EndFrame es un cierre con causa
            self._cancelar_vigilancia()
        elif isinstance(frame, CancelFrame):
            # Desmontaje abrupto (p.ej. el navegador colgó): NO es un cierre
            # normal — `on_client_disconnected` decide si fue CONNECTION_LOST.
            self._terminado = True
            self._cancelar_vigilancia()

        # El estado físico de "el paciente habla" lo llevan los frames del VAD
        # (`VADProcessor` los emite SIEMPRE); los de turno (`UserStarted/
        # StoppedSpeakingFrame`, del `UserTurnProcessor`) llegan después —el fin
        # de turno espera transcript— y se aceptan como equivalentes con guarda
        # de idempotencia para no contar doble ni rearmar de más.
        if isinstance(frame, (VADUserStartedSpeakingFrame, UserStartedSpeakingFrame)):
            if not self._paciente_hablando:
                self._paciente_hablando = True
                self._veces_que_oimos_al_paciente += 1
                if self._escalera.en_episodio():
                    emit(VoiceEvent.PATIENT_SPEECH_DETECTED,
                         conversation_id=self._conversation_id,
                         duration_ms=self._escalera.duracion_ms(),
                         phase=self._ultimo_phase, slot=self._ultimo_slot)
                if self._agente_hablando:
                    # Hay solape físico de voces. En modo BARGE_IN_VAD la
                    # interrupción del TTS la dispara el `UserTurnProcessor`;
                    # en modo confirmado, la transcripción de más abajo.
                    emit(VoiceEvent.PATIENT_INTERRUPTED_AGENT,
                         conversation_id=self._conversation_id,
                         phase=self._ultimo_phase, slot=self._ultimo_slot)
                logger.info("[voz] VAD: el paciente empezó a hablar")
            self._t_fin_habla = None
            self._cancelar_vigilancia()   # hay alguien al teléfono
        elif isinstance(frame, (VADUserStoppedSpeakingFrame, UserStoppedSpeakingFrame)):
            if self._paciente_hablando:
                self._t_fin_habla = time.perf_counter()
                self._paciente_hablando = False
                # Y se rearma. Sin esto quedaba un agujero simétrico al del bug:
                # si el STT no devuelve nada inteligible (ruido, un carraspeo),
                # no hay turno ni respuesta del agente, así que no llega ningún
                # `BotStoppedSpeakingFrame` y el reloj se quedaba cancelado para
                # siempre — la llamada se quedaba muerta y abierta sin que nadie
                # lo supiera, que es justo lo que este reloj existe para evitar.
                self._armar_vigilancia()
                logger.info("[voz] VAD: el paciente dejó de hablar")
        elif isinstance(frame, TranscriptionFrame) and not frame.text.strip():
            # Groq transcribió pero no oyó nada inteligible (silencio, ruido).
            # Sin este log, esto se ve idéntico a que el micrófono nunca llegó.
            logger.warning("[voz] Groq devolvió una transcripción vacía")

        if isinstance(frame, TranscriptionFrame) and frame.text and frame.text.strip():
            text = frame.text.strip()
            # Mientras el agente habla el filtro de eco se endurece: un
            # fragmento corto de su propia frase no puede cortarle el TTS.
            min_palabras = (self._min_palabras_eco_hablando if self._agente_hablando
                            else _MIN_PALABRAS_ECO)
            if es_eco(text, self._ultima_respuesta, min_palabras=min_palabras):
                # El micrófono está recogiendo la voz del propio agente. Sin esta
                # guarda el sistema se responde a sí mismo: transcribe su última
                # frase, la mete como turno del paciente y contesta, en bucle y sin
                # dejar hablar a nadie. Se descarta y se deja rastro, porque desde
                # fuera un eco se ve igual que un paciente que repite lo que oye.
                logger.warning(f"[voz] descartado, es el eco del propio agente: {text!r}")
                self._armar_vigilancia()
                return
            if self._agente_hablando and intent_nlu.classify(text) == "ininteligible":
                # Compuerta de ruido del barge-in: una tos o un carraspeo que el
                # STT transcribe como basura no puede cortarle el TTS al agente
                # ni convertirse en turno. Con el agente callado sí baja al
                # orquestador, que ya sabe contestar "no le escuché bien".
                logger.info(f"[voz] ruido mientras el agente habla, se ignora: {text!r}")
                self._armar_vigilancia()
                return
            logger.info(f"[voz] paciente: {text}")
            self._cancelar_vigilancia()
            self._escalera.reset()        # turno real: episodio de silencio nuevo
            # ANTES de interrumpir: el `BotStoppedSpeakingFrame` que provoca la
            # interrupción no debe armar el reloj (`_puede_vigilar` lo filtra).
            self._turno_arranca()
            if self._barge_in and self._agente_hablando:
                # Barge-in confirmado: la transcripción ya pasó el filtro de
                # eco, así que es el paciente de verdad. Se corta el TTS —
                # `broadcast_interruption` cancela la síntesis en vuelo,
                # descarta las frases encoladas y drena la cola de audio del
                # transporte — y su turno se procesa con normalidad.
                logger.info("[voz] barge-in: el paciente interrumpió al agente")
                emit(VoiceEvent.AGENT_INTERRUPTED,
                     conversation_id=self._conversation_id,
                     phase=self._ultimo_phase, slot=self._ultimo_slot)
                self._registrar_interrupcion()
                await self.broadcast_interruption()
            # El orquestador toca la base de datos: corre en tarea propia (ver
            # `_turno_y_responder`) para sobrevivir a las interrupciones y no
            # bloquear el event loop del pipeline de audio.
            # La marca de interrupción se consume AQUÍ, no dentro de la tarea:
            # pertenece a este turno y a ningún otro.
            interrumpida, self._pregunta_interrumpida = self._pregunta_interrumpida, False
            self._tarea_turno = asyncio.create_task(
                self._turno_y_responder(text, interrumpida))
            return

        await self.push_frame(frame, direction)


def _abandonar(conversation_id: str) -> None:
    """Cierra por CONNECTION_LOST, con sesión propia (mismo patrón que _run_turn)."""
    session = SessionLocal()
    try:
        abandon_conversation(session, conversation_id)
    except Exception:  # noqa: BLE001 — el cierre de la llamada no puede romper el desmontaje
        logger.exception("[voz] no se pudo cerrar la conversación abandonada")
    finally:
        session.close()


def _build_tts(s):
    """Construye el servicio de TTS según `TTS_PROVIDER`.

    `piper` es el proveedor por defecto (ver el porqué en el docstring del
    módulo). `kokoro` se mantiene disponible: importa Pipecat perezosamente
    porque este módulo entero ya se importa perezosamente desde el router, así
    que el import extra no le cuesta nada a quien nunca activa la voz.

    Para `kokoro` hay que forzar el G2P en español (`Language.ES`): el default
    de `KokoroTTSService` es `en`, y sin esto el texto español se fonemiza en
    inglés y suena con acento anglosajón. `Language.ES` usa el mismo respaldo
    espeak-ng de misaki que el proyecto `leonelhs/kokoro-tts-spanish`.
    """
    if s.tts_provider == "kokoro":
        from pipecat.services.kokoro.tts import KokoroTTSService, KokoroTTSSettings

        return KokoroTTSService(
            settings=KokoroTTSSettings(voice=s.tts_voice, language=Language.ES),
        )

    from piper.config import SynthesisConfig
    from pipecat.services.piper.tts import PiperTTSService

    tts = PiperTTSService(voice_id=s.tts_voice, download_dir=Path(s.piper_voices_dir))
    # `PiperTTSSettings` de Pipecat no tiene campos para prosodia: `run_tts` llama
    # a `self._voice.synthesize(text)` sin `syn_config`, así que usa siempre los
    # valores por defecto del modelo. Se parchea la llamada ya vinculada (en vez
    # de subclasear y duplicar la lógica de streaming de `run_tts`) para que
    # cada síntesis use los valores calibrados en `config.py`.
    syn_config = SynthesisConfig(
        length_scale=s.piper_length_scale,
        noise_scale=s.piper_noise_scale,
        noise_w_scale=s.piper_noise_w_scale,
    )
    tts._voice.synthesize = partial(tts._voice.synthesize, syn_config=syn_config)
    return tts


async def run_bot(webrtc_connection, patient_id: str | None = None) -> None:
    """Arma y ejecuta el pipeline de voz para una conexión WebRTC del navegador."""
    s = get_settings()

    transport = SmallWebRTCTransport(
        webrtc_connection=webrtc_connection,
        params=TransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            audio_out_10ms_chunks=2,
        ),
    )

    # El VAD es un FrameProcessor propio en esta versión de Pipecat (1.7), no un
    # parámetro del transporte. `TransportParams` (base_transport.py) no declara
    # ningún campo `vad_analyzer` — pasarlo ahí no lanza error, pydantic lo
    # descarta en silencio, y el resultado es un transporte que nunca detecta
    # cuándo el paciente deja de hablar: el audio entra, pero `GroqSTTService`
    # (que transcribe por segmentos) jamás recibe la señal de "ya terminó" y no
    # transcribe nada, aunque la llamada parezca conectada y funcionando.
    vad = VADProcessor(
        vad_analyzer=SileroVADAnalyzer(
            params=VADParams(
                # 0.7 s de silencio antes de dar el turno por terminado. Es la palanca
                # dominante de la latencia (la mitad del presupuesto) y está alta a
                # propósito: cortar a un paciente de 80 años a media frase es peor que
                # responder medio segundo más tarde.
                stop_secs=s.vad_stop_secs,
                start_secs=0.2,
                # Configurables desde `.env`: Silero exige confianza Y volumen a la
                # vez, así que si el micrófono no llega al umbral de volumen no hay
                # VAD, y sin VAD `GroqSTTService` no transcribe nada — la llamada
                # entera se interpreta como silencio. Ver `config.py`.
                confidence=s.vad_confidence,
                min_volume=s.vad_min_volume,
            )
        )
    )
    # Se registran los valores efectivos: cuando una llamada no oye al paciente,
    # esto es lo primero que hay que poder mirar sin adivinar qué había en `.env`.
    logger.info(
        f"[voz] VAD confidence={s.vad_confidence} min_volume={s.vad_min_volume} "
        f"stop_secs={s.vad_stop_secs} · escalera de silencios "
        f"{s.silence_initial_s:.0f}s+{s.silence_gentle_s:.0f}s+{s.silence_repeat_s:.0f}s "
        f"· barge_in={'vad' if s.barge_in_vad else ('confirmado' if s.barge_in_enabled else 'off')}"
    )

    stt = GroqSTTService(
        api_key=s.groq_api_key,
        model=s.stt_model,  # whisper-large-v3-turbo
        language=Language.ES,
        prompt=_PROMPT_STT,
    )

    tts = _build_tts(s)

    clinico = ClinicalProcessor(patient_id=patient_id)

    # Gestor de turnos de usuario. Es quien emite `UserStarted/StoppedSpeakingFrame`
    # —sin él esas ramas del ClinicalProcessor eran código muerto— y quien puede
    # emitir el `InterruptionFrame` del barge-in instantáneo.
    #
    #   - `enable_interruptions=BARGE_IN_VAD` (False por defecto): con True el
    #     TTS se corta a ~0.2 s de detectar voz, pero el eco del propio agente
    #     por el micrófono lo dispararía igual — solo con AEC fiable. El modo
    #     por defecto es el barge-in CONFIRMADO: lo emite el ClinicalProcessor
    #     tras el filtro `es_eco`, con la transcripción ya en mano (así el flush
    #     de la interrupción no puede descartarla).
    #   - `stop=` EXPLÍCITO: el default de `UserTurnStrategies` instala
    #     `TurnAnalyzerUserTurnStopStrategy(LocalSmartTurnAnalyzerV3)`, que
    #     descarga un modelo de HuggingFace en el arranque — inaceptable con la
    #     compuerta de 15 minutos. El timeout corre DESPUÉS del stop del VAD
    #     (que ya espera `vad_stop_secs`), así que corto.
    turnos = UserTurnProcessor(
        user_turn_strategies=UserTurnStrategies(
            start=[VADUserTurnStartStrategy(enable_interruptions=s.barge_in_vad)],
            stop=[SpeechTimeoutUserTurnStopStrategy(user_speech_timeout=0.2)],
        ),
    )

    pipeline = Pipeline(
        [
            transport.input(),
            vad,
            stt,
            turnos,
            clinico,
            tts,
            transport.output(),
        ]
    )

    worker = PipelineWorker(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
            # ⚠️ No pasar `allow_interruptions` aquí: `PipelineParams` (pipecat
            # 1.7) no declara ese campo y pydantic lo descarta en silencio —
            # mismo modo de fallo que `vad_analyzer` en `TransportParams`. Las
            # interrupciones se activan con las estrategias de turno de arriba
            # y con `broadcast_interruption()` en el ClinicalProcessor.
        ),
    )

    @transport.event_handler("on_client_connected")
    async def _on_connected(_transport, _client):  # pragma: no cover
        logger.info("[voz] cliente conectado — saludando")
        # El saludo se encola en el worker, así que no pasa por `ClinicalProcessor`
        # y éste no se enteraría de que lo dijo. Se le anota a mano porque es
        # justamente la frase más expuesta al eco: es la primera que suena, y si
        # vuelve por el micrófono el sistema arranca contestándose a sí mismo.
        clinico._ultima_respuesta = APERTURA   # noqa: SLF001 — mismo módulo
        await worker.queue_frames([TTSSpeakFrame(APERTURA)])

    @transport.event_handler("on_client_disconnected")
    async def _on_disconnected(_transport, _client):  # pragma: no cover
        logger.info("[voz] cliente desconectado")
        # Antes que el worker: si el reloj de inactividad vence mientras se cierra
        # todo, intentaría procesar un turno contra un pipeline que ya no existe.
        clinico._terminado = True          # noqa: SLF001 — mismo módulo
        clinico._cancelar_vigilancia()     # noqa: SLF001
        # CONNECTION_LOST ≠ NO_RESPONSE: aquí el CANAL se cayó (el navegador
        # colgó, la red murió) con la llamada a medias. Sin este cierre la
        # conversación quedaba `active` para siempre: sin resumen, sin alerta,
        # invisible en el reporte. `abandon_conversation` es idempotente — tras
        # un cierre normal (el transporte también se desconecta al colgar el
        # agente) no crea nada.
        conv_id = clinico._conversation_id  # noqa: SLF001
        if conv_id and not clinico._cierre_normal:  # noqa: SLF001
            emit(VoiceEvent.CONNECTION_LOST, conversation_id=conv_id,
                 phase=clinico._ultimo_phase, slot=clinico._ultimo_slot)  # noqa: SLF001
            await asyncio.to_thread(_abandonar, conv_id)
        await worker.cancel()

    runner = WorkerRunner(handle_sigint=False)
    await runner.add_workers(worker)
    await runner.run()

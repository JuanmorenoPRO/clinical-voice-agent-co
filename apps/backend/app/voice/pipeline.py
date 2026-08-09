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
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.audio.vad_processor import VADProcessor
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.groq.stt import GroqSTTService
from pipecat.transcriptions.language import Language
from pipecat.transports.base_transport import TransportParams
from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport
from pipecat.workers.runner import WorkerRunner

from ..agent.orchestrator import process_turn
from ..agent.phrasing import APERTURA
from ..nlu import lexicon
from ..schemas import TurnResponse
from ..config import get_settings
from ..db import SessionLocal

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


def es_eco(transcripcion: str, ultima_respuesta: str) -> bool:
    """¿Esto que 'dijo el paciente' es en realidad la voz del agente rebotada?

    Ocurre cuando el paciente está en altavoz o el navegador no cancela el eco: el
    STT transcribe al propio agente, el orquestador lo toma por un turno y contesta,
    y la llamada se convierte en el agente hablando solo sin dejar meter baza.

    Se compara por solapamiento de vocabulario y no por igualdad: el STT nunca
    devuelve la frase literal —se come palabras, parte por la mitad— así que
    comparar cadenas no detectaría nada. Se normaliza con `lexicon.normalize`, que
    es la función que este repo ya usa para comparar habla.
    """
    if not ultima_respuesta:
        return False
    dichas = set(lexicon.normalize(transcripcion).split())
    if len(dichas) < _MIN_PALABRAS_ECO:
        return False
    agente = set(lexicon.normalize(ultima_respuesta).split())
    if not agente:
        return False
    return len(dichas & agente) / len(dichas) >= _UMBRAL_ECO


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
                 silence_timeout_s: float | None = None) -> None:
        super().__init__()
        self._conversation_id: str | None = None
        self._patient_id = patient_id
        self._t_fin_habla: float | None = None
        self._silence_timeout = (
            silence_timeout_s if silence_timeout_s is not None
            else get_settings().silence_timeout_s
        )
        self._watchdog: asyncio.Task | None = None
        self._terminado = False
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
        # Cuántas veces el VAD ha detectado voz del paciente en toda la llamada.
        # Sirve para una sola cosa, pero importante: distinguir "el paciente no
        # contesta" de "no le estamos oyendo", que desde los logs se ven idénticos
        # y llevan a arreglar cosas distintas.
        self._veces_que_oimos_al_paciente = 0
        # Lo último que dijo el agente, para descartar su propio eco.
        self._ultima_respuesta = ""
        self._deadline = 0.0

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
        self._deadline = time.monotonic() + self._silence_timeout
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
        logger.info(f"[voz] {self._silence_timeout:.0f} s sin respuesta del paciente")
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
        self._procesando = True
        try:
            result = await asyncio.to_thread(self._run_turn, self.SILENCIO)
        finally:
            self._procesando = False
        logger.info(f"[voz] agente: {result.response}")
        self._ultima_respuesta = result.response
        await self.push_frame(TTSSpeakFrame(result.response))
        if result.call_ended:
            # Se agotó la escalera de `agent/script.py::MAX_SILENCIOS`: ya se
            # avisó de que la llamada iba a terminar y aquí se cumple. `EndFrame`
            # es de control, así que cuelga después de que suene la despedida.
            self._terminado = True
            logger.info("[voz] llamada cerrada: el paciente dejó de contestar")
            await self.push_frame(EndFrame(reason="sin_respuesta"))
        # Si no terminó, el reloj se rearma solo cuando acabe este TTS
        # (`BotStoppedSpeakingFrame`, más abajo).

    def _run_turn(self, text: str) -> TurnResponse:
        session = SessionLocal()
        try:
            result = process_turn(
                session,
                text=text,
                conversation_id=self._conversation_id,
                patient_id=self._patient_id,
            )
            self._conversation_id = result.conversation_id
            return result
        finally:
            session.close()

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)

        if isinstance(frame, BotStartedSpeakingFrame):
            # Mientras suena la pregunta el reloj no corre. Contar desde antes lo
            # haría vencer con el paciente todavía escuchando.
            self._agente_hablando = True
            self._cancelar_vigilancia()
        elif isinstance(frame, BotStoppedSpeakingFrame):
            # OJO: este frame llega por DOS motivos distintos y solo uno de ellos
            # inicia un silencio. El otro es que el paciente acaba de interrumpir
            # al agente (`handle_interruptions` en el transporte de Pipecat lo
            # emite), y ahí el reloj no puede arrancar: el paciente está hablando
            # justo ahora. Ese era el bug. Llega UPSTREAM desde `transport.output()`.
            self._agente_hablando = False
            self._armar_vigilancia()      # `_puede_vigilar` filtra el caso malo
        elif isinstance(frame, (EndFrame, CancelFrame)):
            self._terminado = True
            self._cancelar_vigilancia()

        if isinstance(frame, UserStartedSpeakingFrame):
            self._t_fin_habla = None
            self._paciente_hablando = True
            self._veces_que_oimos_al_paciente += 1
            self._cancelar_vigilancia()   # hay alguien al teléfono
            logger.info("[voz] VAD: el paciente empezó a hablar")
        elif isinstance(frame, UserStoppedSpeakingFrame):
            self._t_fin_habla = time.perf_counter()
            self._paciente_hablando = False
            # Y se rearma. Sin esto quedaba un agujero simétrico al del bug: si el
            # STT no devuelve nada inteligible (ruido, un carraspeo), no hay turno
            # ni respuesta del agente, así que no llega ningún
            # `BotStoppedSpeakingFrame` y el reloj se quedaba cancelado para
            # siempre — la llamada se quedaba muerta y abierta sin que nadie lo
            # supiera, que es justo lo que este reloj existe para evitar.
            self._armar_vigilancia()
            logger.info("[voz] VAD: el paciente dejó de hablar")
        elif isinstance(frame, TranscriptionFrame) and not frame.text.strip():
            # Groq transcribió pero no oyó nada inteligible (silencio, ruido).
            # Sin este log, esto se ve idéntico a que el micrófono nunca llegó.
            logger.warning("[voz] Groq devolvió una transcripción vacía")

        if isinstance(frame, TranscriptionFrame) and frame.text and frame.text.strip():
            text = frame.text.strip()
            if es_eco(text, self._ultima_respuesta):
                # El micrófono está recogiendo la voz del propio agente. Sin esta
                # guarda el sistema se responde a sí mismo: transcribe su última
                # frase, la mete como turno del paciente y contesta, en bucle y sin
                # dejar hablar a nadie. Se descarta y se deja rastro, porque desde
                # fuera un eco se ve igual que un paciente que repite lo que oye.
                logger.warning(f"[voz] descartado, es el eco del propio agente: {text!r}")
                self._armar_vigilancia()
                return
            logger.info(f"[voz] paciente: {text}")
            # El orquestador toca la base de datos: se corre en un hilo para no
            # bloquear el event loop del pipeline de audio.
            self._cancelar_vigilancia()
            self._procesando = True
            try:
                result = await asyncio.to_thread(self._run_turn, text)
            finally:
                self._procesando = False
            if self._t_fin_habla is not None:
                ms = (time.perf_counter() - self._t_fin_habla) * 1000
                logger.info(f"[voz] latencia fin-de-habla → respuesta: {ms:.0f} ms")
            logger.info(f"[voz] agente: {result.response}")
            self._ultima_respuesta = result.response
            await self.push_frame(TTSSpeakFrame(result.response))
            if result.call_ended:
                # Segundo turno tras un escalamiento crítico: el guion ya se dio,
                # esto solo confirma y cuelga. `EndFrame` es un frame de control
                # —se procesa en orden—, así que solo dispara el cierre después
                # de que el TTS del cierre haya terminado de sonar; no corta la
                # frase a la mitad.
                logger.info("[voz] llamada cerrada tras escalamiento crítico")
                await self.push_frame(EndFrame(reason="cierre_por_escalamiento"))
            return

        await self.push_frame(frame, direction)


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
        f"stop_secs={s.vad_stop_secs} · reloj de inactividad {s.silence_timeout_s}s"
    )

    stt = GroqSTTService(
        api_key=s.groq_api_key,
        model=s.stt_model,  # whisper-large-v3-turbo
        language=Language.ES,
        prompt=_PROMPT_STT,
    )

    tts = _build_tts(s)

    clinico = ClinicalProcessor(patient_id=patient_id)

    pipeline = Pipeline(
        [
            transport.input(),
            vad,
            stt,
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
            allow_interruptions=True,
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
        await worker.cancel()

    runner = WorkerRunner(handle_sigint=False)
    await runner.add_workers(worker)
    await runner.run()

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
  - **TTS por defecto: Piper, no Kokoro.** Kokoro es un modelo centrado en
    inglés que cubre español por fonemización de respaldo (espeak-ng): suena a
    acento anglosajón hablando español. Piper entrena un modelo por idioma;
    `es_MX-claude-high` además resultó 5× más rápido en caliente. Ver
    `TTS_PROVIDER` en `config.py` y la medición en `docs/spikes-7-agosto.md`.
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


class ClinicalProcessor(FrameProcessor):
    """Puente entre el STT y el TTS.

    Cada transcripción final del paciente pasa por la misma lógica clínica que el
    modo texto, y su respuesta se envía a voz. Mantiene el `conversation_id` para
    conservar el contexto de toda la llamada.

    Además marca el instante en que el paciente deja de hablar, que es el origen
    exacto que la rúbrica pide para medir la latencia: desde el fin del habla
    hasta que empieza a sonar el audio del agente.
    """

    def __init__(self, patient_id: str | None = None) -> None:
        super().__init__()
        self._conversation_id: str | None = None
        self._patient_id = patient_id
        self._t_fin_habla: float | None = None

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

        if isinstance(frame, UserStartedSpeakingFrame):
            self._t_fin_habla = None
            logger.info("[voz] VAD: el paciente empezó a hablar")
        elif isinstance(frame, UserStoppedSpeakingFrame):
            self._t_fin_habla = time.perf_counter()
            logger.info("[voz] VAD: el paciente dejó de hablar")
        elif isinstance(frame, TranscriptionFrame) and not frame.text.strip():
            # Groq transcribió pero no oyó nada inteligible (silencio, ruido).
            # Sin este log, esto se ve idéntico a que el micrófono nunca llegó.
            logger.warning("[voz] Groq devolvió una transcripción vacía")

        if isinstance(frame, TranscriptionFrame) and frame.text and frame.text.strip():
            text = frame.text.strip()
            logger.info(f"[voz] paciente: {text}")
            # El orquestador toca la base de datos: se corre en un hilo para no
            # bloquear el event loop del pipeline de audio.
            result = await asyncio.to_thread(self._run_turn, text)
            if self._t_fin_habla is not None:
                ms = (time.perf_counter() - self._t_fin_habla) * 1000
                logger.info(f"[voz] latencia fin-de-habla → respuesta: {ms:.0f} ms")
            logger.info(f"[voz] agente: {result.response}")
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
    """
    if s.tts_provider == "kokoro":
        from pipecat.services.kokoro.tts import KokoroTTSService

        return KokoroTTSService(voice_id=s.tts_voice)

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
    vad = VADProcessor(vad_analyzer=SileroVADAnalyzer(params=VADParams(
        # 0.7 s de silencio antes de dar el turno por terminado. Es la palanca
        # dominante de la latencia (la mitad del presupuesto) y está alta a
        # propósito: cortar a un paciente de 80 años a media frase es peor que
        # responder medio segundo más tarde.
        stop_secs=s.vad_stop_secs,
        start_secs=0.2,
        confidence=0.7,
        min_volume=0.6,
    )))

    stt = GroqSTTService(
        api_key=s.groq_api_key,
        model=s.stt_model,          # whisper-large-v3-turbo
        language=Language.ES,
        prompt=_PROMPT_STT,
    )

    tts = _build_tts(s)

    pipeline = Pipeline([
        transport.input(),
        vad,
        stt,
        ClinicalProcessor(patient_id=patient_id),
        tts,
        transport.output(),
    ])

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
        await worker.queue_frames([TTSSpeakFrame(APERTURA)])

    @transport.event_handler("on_client_disconnected")
    async def _on_disconnected(_transport, _client):  # pragma: no cover
        logger.info("[voz] cliente desconectado")
        await worker.cancel()

    runner = WorkerRunner(handle_sigint=False)
    await runner.add_workers(worker)
    await runner.run()

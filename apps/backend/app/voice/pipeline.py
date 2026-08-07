"""Pipeline de voz Pipecat (STT → lógica clínica → TTS) — ADR-003.

Cadena por componentes en streaming, transporte WebRTC al navegador (sin telefonía
real). El MISMO orquestador del modo texto lleva la lógica del turno; aquí solo se
enchufan la entrada y la salida de audio.

    micrófono ──WebRTC──▶ Silero VAD ──▶ Groq Whisper ──▶ orquestador ──▶ Kokoro ──WebRTC──▶ altavoz

Los tres servicios son de primera parte de Pipecat, así que no hay que escribir
ningún `STTService`/`TTSService` a medida. Dos detalles que no son obvios:

  - `GroqSTTService` hereda de `BaseWhisperSTTService`, que trabaja por segmentos:
    **exige un VAD en el transporte**. Sin `SileroVADAnalyzer` no transcribe nada.
  - Nada de esto arrastra PyTorch. `pipecat-ai[kokoro]` depende de `kokoro-onnx`, y
    Silero corre sobre el `onnxruntime` que ya trae el core.

⚠️ Importa Pipecat a nivel de módulo, así que SOLO debe importarse de forma
perezosa (desde el router de voz), para que la app arranque en modo texto sin las
dependencias de voz instaladas.
"""
from __future__ import annotations

import asyncio
import time

from loguru import logger
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import (
    Frame,
    TranscriptionFrame,
    TTSSpeakFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.groq.stt import GroqSTTService
from pipecat.services.kokoro.tts import KokoroTTSService
from pipecat.transcriptions.language import Language
from pipecat.transports.base_transport import TransportParams
from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport
from pipecat.workers.runner import WorkerRunner

from ..agent.orchestrator import process_turn
from ..agent.phrasing import APERTURA
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

    def _run_turn(self, text: str) -> str:
        session = SessionLocal()
        try:
            result = process_turn(
                session,
                text=text,
                conversation_id=self._conversation_id,
                patient_id=self._patient_id,
            )
            self._conversation_id = result.conversation_id
            return result.response
        finally:
            session.close()

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)

        if isinstance(frame, UserStartedSpeakingFrame):
            self._t_fin_habla = None
        elif isinstance(frame, UserStoppedSpeakingFrame):
            self._t_fin_habla = time.perf_counter()

        if isinstance(frame, TranscriptionFrame) and frame.text and frame.text.strip():
            text = frame.text.strip()
            logger.info(f"[voz] paciente: {text}")
            # El orquestador toca la base de datos: se corre en un hilo para no
            # bloquear el event loop del pipeline de audio.
            response = await asyncio.to_thread(self._run_turn, text)
            if self._t_fin_habla is not None:
                ms = (time.perf_counter() - self._t_fin_habla) * 1000
                logger.info(f"[voz] latencia fin-de-habla → respuesta: {ms:.0f} ms")
            logger.info(f"[voz] agente: {response}")
            await self.push_frame(TTSSpeakFrame(response))
            return

        await self.push_frame(frame, direction)


async def run_bot(webrtc_connection, patient_id: str | None = None) -> None:
    """Arma y ejecuta el pipeline de voz para una conexión WebRTC del navegador."""
    s = get_settings()

    transport = SmallWebRTCTransport(
        webrtc_connection=webrtc_connection,
        params=TransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            audio_out_10ms_chunks=2,
            # Obligatorio: Groq Whisper transcribe por segmentos y necesita que
            # alguien le diga dónde termina cada uno.
            vad_analyzer=SileroVADAnalyzer(params=VADParams(
                # 0.7 s de silencio antes de dar el turno por terminado. Es la
                # palanca dominante de la latencia (la mitad del presupuesto) y
                # está alta a propósito: cortar a un paciente de 80 años a media
                # frase es peor que responder medio segundo más tarde.
                stop_secs=s.vad_stop_secs,
                start_secs=0.2,
                confidence=0.7,
                min_volume=0.6,
            )),
        ),
    )

    stt = GroqSTTService(
        api_key=s.groq_api_key,
        model=s.stt_model,          # whisper-large-v3-turbo
        language=Language.ES,
        prompt=_PROMPT_STT,
    )

    tts = KokoroTTSService(voice_id=s.tts_voice)   # ef_dora, español

    pipeline = Pipeline([
        transport.input(),
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

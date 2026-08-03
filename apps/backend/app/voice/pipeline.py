"""Pipeline de voz Pipecat 1.x (STT -> lógica clínica -> TTS) — ADR-003.

Cadena por componentes en streaming, transporte WebRTC al navegador (sin
telefonía real). El MISMO `process_turn` del modo texto maneja la lógica del
turno (RAG + 1 llamada LLM + Motor de Decisión + guion CRÍTICO + traza, ADR-006);
aquí solo se enchufan STT (Deepgram, es) y TTS (ElevenLabs es-CO, ADR-007).

    micrófono ──WebRTC──▶ Deepgram STT ──▶ ClinicalProcessor(process_turn) ──▶ ElevenLabs TTS ──WebRTC──▶ altavoz

⚠️ Este módulo importa Pipecat a nivel de módulo, así que SOLO debe importarse de
forma perezosa (desde el router de voz), para que la app arranque sin las deps de
voz instaladas. Requiere DEEPGRAM_API_KEY + ELEVENLABS_API_KEY + ELEVENLABS_VOICE_ID.
"""
from __future__ import annotations

import asyncio

from loguru import logger
from pipecat.frames.frames import Frame, TranscriptionFrame, TTSSpeakFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
# LiveOptions vive dentro del módulo de Pipecat (no en el paquete `deepgram`) desde
# deepgram-sdk 4.x / Pipecat 1.x.
from pipecat.services.deepgram.stt import DeepgramSTTService, LiveOptions
from pipecat.services.elevenlabs.tts import ElevenLabsTTSService
from pipecat.transports.base_transport import TransportParams
from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport
from pipecat.workers.runner import WorkerRunner

from ..config import get_settings
from ..db import SessionLocal
from .conversation import process_turn

_GREETING = (
    "Hola, le llamo del hospital para su seguimiento después de la cirugía. "
    "¿Cómo se ha sentido?"
)


class ClinicalProcessor(FrameProcessor):
    """Puente entre el STT y el TTS: cada transcripción final del paciente pasa
    por la MISMA lógica clínica del modo texto y su respuesta se envía a voz.

    Mantiene el `conversation_id` para conservar el contexto de toda la llamada
    (memoria + acumulación de síntomas, igual que en consola).
    """

    def __init__(self, patient_id: str | None = None) -> None:
        super().__init__()
        self._conversation_id: str | None = None
        self._patient_id = patient_id

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

        # Solo actuamos sobre transcripciones FINALES con texto real.
        if isinstance(frame, TranscriptionFrame) and frame.text and frame.text.strip():
            text = frame.text.strip()
            logger.info(f"[voz] paciente: {text}")
            # process_turn es síncrono y toca la BD: se corre en un hilo para no
            # bloquear el event loop del pipeline.
            response = await asyncio.to_thread(self._run_turn, text)
            logger.info(f"[voz] agente: {response}")
            await self.push_frame(TTSSpeakFrame(response))
            return

        # El resto de frames (Start/End/sistema, interinos) siguen su curso.
        await self.push_frame(frame, direction)


async def run_bot(webrtc_connection) -> None:
    """Arma y ejecuta el pipeline de voz para una conexión WebRTC del navegador."""
    settings = get_settings()

    transport = SmallWebRTCTransport(
        webrtc_connection=webrtc_connection,
        params=TransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            audio_out_10ms_chunks=2,
        ),
    )

    # Deepgram maneja el fin de turno con su propio endpointing (emite la
    # transcripción final que dispara el turno clínico). es-419 = español LatAm.
    stt = DeepgramSTTService(
        api_key=settings.deepgram_api_key,
        live_options=LiveOptions(model="nova-2", language="es-419", smart_format=True),
    )

    tts = ElevenLabsTTSService(
        api_key=settings.elevenlabs_api_key,
        voice_id=settings.elevenlabs_voice_id,  # ⏳ voz nativa es-CO (ADR-007)
        model=settings.elevenlabs_model,          # eleven_flash_v2_5 (~75 ms)
    )

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            ClinicalProcessor(),
            tts,
            transport.output(),
        ]
    )

    worker = PipelineWorker(
        pipeline,
        params=PipelineParams(enable_metrics=True, enable_usage_metrics=True),
    )

    @transport.event_handler("on_client_connected")
    async def _on_connected(_transport, _client):  # pragma: no cover
        logger.info("[voz] cliente conectado — saludando")
        await worker.queue_frames([TTSSpeakFrame(_GREETING)])

    @transport.event_handler("on_client_disconnected")
    async def _on_disconnected(_transport, _client):  # pragma: no cover
        logger.info("[voz] cliente desconectado")
        await worker.cancel()

    runner = WorkerRunner(handle_sigint=False)
    await runner.add_workers(worker)
    await runner.run()

"""Pipeline de voz Pipecat (STT -> LLM -> TTS) — cableado (ADR-003).

Cadena por componentes en streaming orquestada con Pipecat y transporte WebRTC
al navegador (sin telefonía real). El mismo ConversationService de texto maneja
la lógica del turno: aquí solo se enchufan STT (Deepgram, es) y TTS (ElevenLabs
eleven_flash_v2_5, voz nativa es-CO — ADR-007).

⚠️ Este módulo requiere DEEPGRAM_API_KEY y ELEVENLABS_API_KEY reales y los
paquetes extra de Pipecat. Se deja CABLEADO pero no se activa por defecto: el
bucle del turno se prueba hoy vía POST /conversation/turn (modo texto). El 7 de
agosto se valida la voz y se elige el voice_id definitivo.
"""
from __future__ import annotations

from ..config import get_settings


def build_voice_pipeline():  # pragma: no cover - requiere credenciales y hardware de audio
    """Construye el pipeline Pipecat. Import perezoso para no exigir las deps
    de voz cuando solo se usa el modo texto."""
    settings = get_settings()
    if not (settings.deepgram_api_key and settings.elevenlabs_api_key):
        raise RuntimeError(
            "El modo de voz necesita DEEPGRAM_API_KEY y ELEVENLABS_API_KEY. "
            "Usa POST /conversation/turn (modo texto) mientras tanto."
        )

    from pipecat.pipeline.pipeline import Pipeline
    from pipecat.services.deepgram import DeepgramSTTService
    from pipecat.services.elevenlabs import ElevenLabsTTSService

    stt = DeepgramSTTService(api_key=settings.deepgram_api_key, language="es")
    tts = ElevenLabsTTSService(
        api_key=settings.elevenlabs_api_key,
        voice_id=settings.elevenlabs_voice_id,  # ⏳ elegir voz nativa es-CO (ADR-007)
        model=settings.elevenlabs_model,        # eleven_flash_v2_5 (~75ms)
    )

    # NOTA: el procesador central del turno (ClinicalTurnProcessor) envuelve a
    # voice.conversation.process_turn para: extraer síntomas + responder en una
    # sola llamada, correr el Motor de Decisión antes del TTS y emitir el guion
    # de seguridad en CRÍTICO. Se implementa el 7 de agosto al integrar el
    # transporte WebRTC y el modelo obligatorio.
    # from .turn_processor import ClinicalTurnProcessor
    # return Pipeline([transport.input(), stt, ClinicalTurnProcessor(), tts, transport.output()])

    return Pipeline([stt, tts])

"""Cableado del pipeline de voz — compuerta G4.

No se puede simular un micrófono en un test, así que esto no prueba la conversación
hablada: prueba que las piezas existen, se construyen con la configuración real y
se ensamblan. Es lo que evita descubrir en la demo que un servicio cambió de firma.

La validación de la conversación real es manual (navegador + micrófono) y la de la
calidad del audio está en `scripts/spike_voice.py`, que sintetiza con Kokoro y
transcribe con Groq para comprobar que el español vuelve literal.
"""
from __future__ import annotations

import pytest

from app.config import get_settings


def _pipecat_disponible() -> bool:
    try:
        import pipecat  # noqa: F401

        return True
    except ImportError:
        return False


pytestmark = pytest.mark.skipif(
    not _pipecat_disponible(), reason="Requiere pipecat-ai[groq,kokoro,webrtc]"
)


def test_los_tres_servicios_existen_con_la_firma_esperada():
    """Si Pipecat cambia una firma, esto falla aquí y no delante del jurado."""
    import inspect

    from pipecat.audio.vad.silero import SileroVADAnalyzer
    from pipecat.services.groq.stt import GroqSTTService
    from pipecat.services.kokoro.tts import KokoroTTSService

    stt_params = inspect.signature(GroqSTTService.__init__).parameters
    assert {"api_key", "model", "language", "prompt"} <= set(stt_params)
    assert "voice_id" in inspect.signature(KokoroTTSService.__init__).parameters
    assert SileroVADAnalyzer is not None


def test_groq_stt_necesita_vad_en_el_transporte():
    """Documenta por qué el transporte lleva Silero.

    `GroqSTTService` transcribe por segmentos: sin alguien que marque dónde
    termina cada uno, no emite ni una transcripción. Es el fallo silencioso más
    fácil de cometer al montar este pipeline.
    """
    from pipecat.services.groq.stt import GroqSTTService

    bases = [b.__name__ for b in GroqSTTService.__mro__]
    assert any("Segmented" in b or "Whisper" in b for b in bases), bases


def test_el_pipeline_se_arma_con_la_configuracion_real():
    from pipecat.audio.vad.silero import SileroVADAnalyzer
    from pipecat.audio.vad.vad_analyzer import VADParams
    from pipecat.pipeline.pipeline import Pipeline
    from pipecat.services.groq.stt import GroqSTTService
    from pipecat.services.kokoro.tts import KokoroTTSService
    from pipecat.transcriptions.language import Language

    from app.voice.pipeline import _PROMPT_STT, ClinicalProcessor

    s = get_settings()
    SileroVADAnalyzer(params=VADParams(stop_secs=s.vad_stop_secs, start_secs=0.2))
    stt = GroqSTTService(api_key=s.groq_api_key or "sk-test", model=s.stt_model,
                         language=Language.ES, prompt=_PROMPT_STT)
    tts = KokoroTTSService(voice_id=s.tts_voice)
    Pipeline([stt, ClinicalProcessor(), tts])


def test_ninguna_dependencia_de_voz_arrastra_pytorch():
    """El argumento del arranque en 15 minutos depende de esto.

    Si algún día `pipecat-ai[kokoro]` pasara a depender del paquete `kokoro` en
    vez de `kokoro-onnx`, entrarían ~2.5 GB de PyTorch sin que nadie lo note.
    """
    import importlib.util

    assert importlib.util.find_spec("kokoro_onnx") is not None
    assert importlib.util.find_spec("torch") is None, "PyTorch se coló en el entorno"


def test_espeak_viaja_dentro_del_wheel():
    """Kokoro necesita espeak-ng para el español, y no debe instalarse a mano."""
    from pathlib import Path

    import espeakng_loader

    # Devuelve str, no Path.
    assert Path(espeakng_loader.get_library_path()).exists()
    assert Path(espeakng_loader.get_data_path()).exists()


def test_la_voz_configurada_es_de_espanol():
    """Kokoro nombra las voces por idioma: 'e' = español (ef_/em_)."""
    assert get_settings().tts_voice.startswith(("ef_", "em_"))


def test_el_procesador_marca_el_fin_del_habla():
    """El origen de la latencia que pide la rúbrica es el fin del habla, no el
    inicio del request."""
    from app.voice.pipeline import ClinicalProcessor

    p = ClinicalProcessor()
    assert p._t_fin_habla is None  # noqa: SLF001
    assert hasattr(p, "process_frame")

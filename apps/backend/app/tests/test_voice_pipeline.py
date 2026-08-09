"""Cableado del pipeline de voz — compuerta G4.

No se puede simular un micrófono en un test, así que esto no prueba la conversación
hablada: prueba que las piezas existen, se construyen con la configuración real y
se ensamblan. Es lo que evita descubrir en la demo que un servicio cambió de firma.

La validación de la conversación real es manual (navegador + micrófono) y la de la
calidad del audio está en `scripts/spike_voice.py`, que sintetiza y transcribe con
Groq para comprobar que el español vuelve literal.
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
    not _pipecat_disponible(), reason="Requiere pipecat-ai[groq,kokoro,piper,webrtc]"
)


def test_los_tres_servicios_existen_con_la_firma_esperada():
    """Si Pipecat cambia una firma, esto falla aquí y no delante del jurado."""
    import inspect

    from pipecat.audio.vad.silero import SileroVADAnalyzer
    from pipecat.services.groq.stt import GroqSTTService
    from pipecat.services.piper.tts import PiperTTSService

    stt_params = inspect.signature(GroqSTTService.__init__).parameters
    assert {"api_key", "model", "language", "prompt"} <= set(stt_params)
    assert "voice_id" in inspect.signature(PiperTTSService.__init__).parameters
    assert SileroVADAnalyzer is not None


def test_groq_stt_necesita_vad_en_el_transporte():
    """Documenta por qué el pipeline lleva un VADProcessor con Silero.

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
    from pipecat.processors.audio.vad_processor import VADProcessor
    from pipecat.services.groq.stt import GroqSTTService
    from pipecat.transcriptions.language import Language

    from app.voice.pipeline import _PROMPT_STT, ClinicalProcessor, _build_tts

    s = get_settings()
    # El VAD tiene que estar DENTRO del pipeline como su propio FrameProcessor,
    # no solo instanciado y descartado: así es como se descubrió el bug real de
    # este archivo (ver test_vad_no_se_pasa_como_parametro_del_transporte).
    vad = VADProcessor(
        vad_analyzer=SileroVADAnalyzer(
            params=VADParams(stop_secs=s.vad_stop_secs, start_secs=0.2)
        )
    )
    stt = GroqSTTService(
        api_key=s.groq_api_key or "sk-test",
        model=s.stt_model,
        language=Language.ES,
        prompt=_PROMPT_STT,
    )
    tts = _build_tts(s)
    Pipeline([vad, stt, ClinicalProcessor(), tts])


def test_vad_no_se_pasa_como_parametro_del_transporte():
    """Regresión: `TransportParams(vad_analyzer=...)` no falla, solo no hace nada.

    `TransportParams` (base_transport.py) no declara el campo `vad_analyzer`, y
    Pydantic descarta en silencio cualquier kwarg no declarado: no hay
    `ValidationError`, no hay warning. El resultado es un transporte que nunca
    detecta cuándo el paciente deja de hablar — el audio entra, pero
    `GroqSTTService` (que transcribe por segmentos) no recibe jamás la señal de
    "ya terminó" y no transcribe nada, aunque la llamada se vea conectada.

    En esta versión de Pipecat el VAD es un `FrameProcessor` propio
    (`VADProcessor`) que hay que insertar en el pipeline explícitamente — ver
    `run_bot()` en `app/voice/pipeline.py`. Este test fija por qué NO alcanza con
    pasarlo al construir `TransportParams`.
    """
    from pipecat.audio.vad.silero import SileroVADAnalyzer
    from pipecat.transports.base_transport import TransportParams

    params = TransportParams(
        audio_in_enabled=True,
        vad_analyzer=SileroVADAnalyzer(),  # kwarg fantasma: no existe ese campo
    )
    assert not hasattr(params, "vad_analyzer"), (
        "TransportParams ahora SÍ declara vad_analyzer: se puede volver a pasar "
        "en el constructor del transporte y retirar el VADProcessor explícito."
    )


def test_ninguna_dependencia_de_voz_arrastra_pytorch():
    """El argumento del arranque en 15 minutos depende de esto.

    piper-tts solo depende de onnxruntime en su instalación base (torch entra
    únicamente con el extra 'train', que no se usa). Kokoro depende de
    kokoro-onnx, no del paquete `kokoro` con PyTorch.
    """
    import importlib.util

    assert importlib.util.find_spec("piper") is not None
    assert importlib.util.find_spec("torch") is None, "PyTorch se coló en el entorno"


def test_espeak_viaja_dentro_del_wheel():
    """Kokoro necesita espeak-ng para el español, y no debe instalarse a mano.

    Solo aplica si TTS_PROVIDER=kokoro; Piper no depende de espeak-ng en absoluto
    (entrena un modelo por idioma en vez de fonemizar con un motor genérico).
    """
    from pathlib import Path

    import espeakng_loader

    # Devuelve str, no Path.
    assert Path(espeakng_loader.get_library_path()).exists()
    assert Path(espeakng_loader.get_data_path()).exists()


def test_el_proveedor_configurado_y_su_g2p_espanol_se_cablean():
    """El proveedor del .env manda, y el pipeline cablea el G2P correcto.

        Dos proveedores posibles:

    - `piper`: modelo entrenado por idioma, sin fonemización de respaldo; la
            nomenclatura de voz es `es_*`/`en_*`. Medido en docs/spikes-7-agosto.md,
            es_MX-claude-high es 5x más rápido en caliente que Kokoro (RTF 0.05 vs 0.24).
          - `kokoro`: modelo centrado en inglés que cubre el español por G2P. Su
            default es `Language.EN`, que produce acento anglosajón; el pipeline
            fuerza `Language.ES` (el mismo respaldo espeak-ng de misaki que usa
            `leonelhs/kokoro-tts-spanish`) — ver `_build_tts` en app/voice/pipeline.py.
    """
    s = get_settings()
    if s.tts_provider == "piper":
        assert s.tts_voice.startswith(("es_", "en_"))  # nomenclatura de Piper
    elif s.tts_provider == "kokoro":
        from pipecat.transcriptions.language import Language
        from app.voice.pipeline import _build_tts
        from pipecat.services.kokoro.tts import KokoroTTSService

        svc = _build_tts(s)
        assert isinstance(svc, KokoroTTSService)
        assert svc._settings.language == Language.ES  # noqa: SLF001
    else:  # pragma: no cover
        raise AssertionError(f"TTS_PROVIDER inesperado: {s.tts_provider!r}")


def test_el_procesador_marca_el_fin_del_habla():
    """El origen de la latencia que pide la rúbrica es el fin del habla, no el
    inicio del request."""
    from app.voice.pipeline import ClinicalProcessor

    p = ClinicalProcessor()
    assert p._t_fin_habla is None  # noqa: SLF001
    assert hasattr(p, "process_frame")


def test_el_reloj_de_inactividad_se_arma_al_callarse_el_agente_y_se_cancela_al_hablar():
    """El VAD detecta cuándo el paciente DEJA de hablar; de quien no empieza nunca
    no dice nada. Ese hueco es el que cubre este reloj: sin él, un paciente que
    suelta el teléfono deja la llamada abierta para siempre.

    Se cuenta desde que el agente se calla (`BotStoppedSpeakingFrame`, que llega
    UPSTREAM desde `transport.output()`), no desde que empieza: contar mientras
    todavía suena la pregunta produciría un "¿sigue ahí?" antes de que el paciente
    haya tenido ocasión de contestar.
    """
    import asyncio

    from app.voice.pipeline import ClinicalProcessor

    async def escenario():
        p = ClinicalProcessor(silence_timeout_s=30)   # largo: aquí no debe vencer
        assert p._watchdog is None                    # noqa: SLF001

        p._armar_vigilancia()                         # noqa: SLF001
        assert p._watchdog is not None                # noqa: SLF001

        p._cancelar_vigilancia()                      # noqa: SLF001
        assert p._watchdog is None                    # noqa: SLF001

        # Una llamada ya terminada no vuelve a armar el reloj: sin esto, el
        # `EndFrame` del cierre competiría con un turno de silencio contra un
        # pipeline que se está desmontando.
        p._terminado = True                           # noqa: SLF001
        p._armar_vigilancia()                         # noqa: SLF001
        assert p._watchdog is None                    # noqa: SLF001

    asyncio.run(escenario())


# --- el reloj, visto desde los FRAMES ----------------------------------------
# El test de arriba llama a `_armar_vigilancia()`/`_cancelar_vigilancia()` a mano
# y nunca mete un frame por `process_frame`. Por eso se le coló un bug que solo
# existe en la SECUENCIA de frames: cuando el paciente interrumpe al agente,
# Pipecat emite `BotStoppedSpeakingFrame` (`transports/base_output.py`,
# `handle_interruptions`) y el reloj se rearmaba con el paciente a media frase.
# A los pocos segundos le llegaba "¿Sigue ahí? No le escuché nada." por encima.
#
# Estos meten los frames de verdad. `FrameProcessor.process_frame` necesita que el
# procesador esté enlazado a una tarea del pipeline para poder empujar frames, así
# que se le pasa `direction` y se comprueba el ESTADO del reloj, no la salida.


def _procesador(timeout=30):
    """Un `ClinicalProcessor` real, con `push_frame` desviado a una lista.

    Se sustituye `push_frame` y no el despacho: el objetivo es ejercitar el
    `process_frame` DE VERDAD. Un helper que reimplantara las transiciones
    probaría su propia copia y seguiría pasando aunque el módulo estuviera mal —
    que es exactamente cómo se coló el bug que estos tests fijan.
    """
    from pipecat.processors.frame_processor import FrameDirection

    from app.voice.pipeline import ClinicalProcessor

    p = ClinicalProcessor(silence_timeout_s=timeout)
    p.empujados = []

    async def _push(frame, direction=FrameDirection.DOWNSTREAM):
        p.empujados.append(frame)

    p.push_frame = _push
    return p


async def _meter(p, frame):
    """Mete un frame por el `process_frame` real, en la dirección en que llega."""
    from pipecat.frames.frames import BotStartedSpeakingFrame, BotStoppedSpeakingFrame
    from pipecat.processors.frame_processor import FrameDirection

    # Los frames del bot llegan UPSTREAM desde `transport.output()`; los del
    # paciente bajan desde el VAD.
    direccion = (
        FrameDirection.UPSTREAM
        if isinstance(frame, (BotStartedSpeakingFrame, BotStoppedSpeakingFrame))
        else FrameDirection.DOWNSTREAM
    )
    await p.process_frame(frame, direccion)


def test_la_interrupcion_del_paciente_no_rearma_el_reloj():
    """EL BUG. `BotStoppedSpeakingFrame` significa dos cosas distintas.

    "Terminé de hablar" inicia un silencio; "me interrumpieron" no, porque el
    paciente está hablando justo en ese momento. Tratarlas igual hacía que el
    agente le soltara el sondeo de presencia encima de su propia respuesta, y eso
    volvía a interrumpir: la llamada se realimentaba.
    """
    import asyncio

    from pipecat.frames.frames import (
        BotStoppedSpeakingFrame,
        UserStartedSpeakingFrame,
    )

    async def escenario():
        p = _procesador()
        await _meter(p, BotStoppedSpeakingFrame())     # el agente termina la pregunta
        assert p._watchdog is not None                 # noqa: SLF001

        await _meter(p, UserStartedSpeakingFrame())    # el paciente arranca a hablar
        assert p._watchdog is None                     # noqa: SLF001

        # Y AQUÍ el frame que emite `handle_interruptions` de Pipecat.
        await _meter(p, BotStoppedSpeakingFrame())
        assert p._watchdog is None, (                  # noqa: SLF001
            "el reloj no puede correr mientras el paciente está hablando"
        )

    asyncio.run(escenario())


def test_mientras_suena_la_pregunta_el_reloj_no_corre():
    import asyncio

    from pipecat.frames.frames import BotStartedSpeakingFrame, BotStoppedSpeakingFrame

    async def escenario():
        p = _procesador()
        await _meter(p, BotStoppedSpeakingFrame())
        await _meter(p, BotStartedSpeakingFrame())
        assert p._watchdog is None                     # noqa: SLF001

    asyncio.run(escenario())


def test_el_reloj_se_rearma_si_el_stt_no_devuelve_nada():
    """El agujero simétrico al bug, y también deja la llamada rota.

    Si el paciente hace un ruido que el VAD toma por voz pero el STT no devuelve
    nada inteligible, no hay turno ni respuesta del agente — así que no llega
    ningún `BotStoppedSpeakingFrame` y el reloj se quedaba cancelado PARA SIEMPRE.
    La llamada se quedaba abierta y muda, que es justo lo que este reloj existe
    para evitar.
    """
    import asyncio

    from pipecat.frames.frames import UserStartedSpeakingFrame, UserStoppedSpeakingFrame

    async def escenario():
        p = _procesador()
        await _meter(p, UserStartedSpeakingFrame())
        assert p._watchdog is None                     # noqa: SLF001
        await _meter(p, UserStoppedSpeakingFrame())
        assert p._watchdog is not None                 # noqa: SLF001

    asyncio.run(escenario())


def test_rearmar_muchas_veces_no_adelanta_el_vencimiento():
    """Lo que hacía imposible el síntoma reportado: la escalera entera de corrido.

    El reloj usa un deadline monótono re-comprobado al despertar, no un `sleep`
    de una sola vez, así que una tormenta de frames puede rearmarlo N veces sin
    que ninguna despierte antes de tiempo.
    """
    import asyncio
    import time

    from pipecat.frames.frames import BotStoppedSpeakingFrame

    async def escenario():
        p = _procesador(timeout=0.25)
        t0 = time.monotonic()
        for _ in range(20):
            await _meter(p, BotStoppedSpeakingFrame())
            await asyncio.sleep(0)
        assert p._deadline - t0 >= 0.25                # noqa: SLF001
        # Y sigue sin haber vencido: 20 rearmes no equivalen a 20 silencios.
        assert not p._watchdog.done()                  # noqa: SLF001

    asyncio.run(escenario())


def test_un_turno_en_vuelo_impide_que_venza_el_reloj():
    """Con `LLM_PROVIDER=groq` un turno tarda lo que tarde la red.

    El reloj no puede vencer mientras se está generando la respuesta: el paciente
    todavía no la ha oído, así que su silencio no significa nada.
    """
    import asyncio

    async def escenario():
        p = _procesador()
        p._procesando = True                           # noqa: SLF001
        p._armar_vigilancia()                          # noqa: SLF001
        assert p._watchdog is None                     # noqa: SLF001

    asyncio.run(escenario())


# --- eco del propio agente ----------------------------------------------------


def test_el_eco_del_agente_no_se_toma_por_una_respuesta():
    """Sin esto el sistema se responde a sí mismo y no deja hablar a nadie.

    Ocurre con el paciente en altavoz o sin cancelación de eco: el STT transcribe
    al propio agente y el orquestador lo mete como turno del paciente.
    """
    from app.agent.phrasing import APERTURA
    from app.voice.pipeline import es_eco

    # El STT nunca devuelve la frase literal: se come palabras y parte por la
    # mitad. Por eso se compara por vocabulario y no por igualdad.
    assert es_eco("le voy a hacer unas preguntas rápidas para empezar", APERTURA)
    assert es_eco(
        "ha tenido fiebre o calentura estos días",
        "¿Ha tenido fiebre o calentura estos días?",
    )


@pytest.mark.parametrize(
    "respuesta",
    [
        # Respuestas reales del paciente que NO pueden descartarse por error:
        # perder un turno de verdad es peor que dejar pasar un eco.
        "un cuatro más o menos",
        "no he tenido fiebre estos días",
        "me duele un berraco, no aguanto",
        "sí",
        "la herida se ve rojita en el borde",
    ],
)
def test_una_respuesta_de_verdad_no_se_confunde_con_eco(respuesta):
    from app.agent.phrasing import APERTURA
    from app.voice.pipeline import es_eco

    assert not es_eco(respuesta, APERTURA)


def test_sin_respuesta_previa_no_hay_eco_posible():
    from app.voice.pipeline import es_eco

    assert not es_eco("cualquier cosa que diga el paciente", "")


def test_el_marcador_de_silencio_es_el_mismo_que_entiende_el_orquestador():
    """Si estos dos se separan, el reloj sigue venciendo pero el orquestador
    trata el marcador como una respuesta cualquiera y la escalera nunca arranca.
    """
    from app.nlu import intent
    from app.voice.pipeline import ClinicalProcessor

    assert intent.classify(ClinicalProcessor.SILENCIO) == "silencio"

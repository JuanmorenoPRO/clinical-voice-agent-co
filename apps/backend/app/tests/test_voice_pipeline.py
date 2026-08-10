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
    from pipecat.turns.user_start.vad_user_turn_start_strategy import (
        VADUserTurnStartStrategy,
    )
    from pipecat.turns.user_stop.speech_timeout_user_turn_stop_strategy import (
        SpeechTimeoutUserTurnStopStrategy,
    )
    from pipecat.turns.user_turn_processor import UserTurnProcessor
    from pipecat.turns.user_turn_strategies import UserTurnStrategies

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
    # El gestor de turnos con las MISMAS estrategias que `run_bot`: es quien
    # emite `UserStarted/StoppedSpeakingFrame` (sin él esas ramas del
    # ClinicalProcessor son código muerto) y quien dispararía el barge-in por VAD.
    turnos = UserTurnProcessor(
        user_turn_strategies=UserTurnStrategies(
            start=[VADUserTurnStartStrategy(enable_interruptions=s.barge_in_vad)],
            stop=[SpeechTimeoutUserTurnStopStrategy(user_speech_timeout=0.2)],
        ),
    )
    tts = _build_tts(s)
    Pipeline([vad, stt, turnos, ClinicalProcessor(), tts])


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
        from app.voice.silence import SilenceConfig

        p = ClinicalProcessor(                        # largo: aquí no debe vencer
            silence_config=SilenceConfig(initial_s=30, gentle_s=0, repeat_s=30))
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


def _procesador(timeout=30, gentle=0.0):
    """Un `ClinicalProcessor` real, con `push_frame` desviado a una lista.

    Se sustituye `push_frame` y no el despacho: el objetivo es ejercitar el
    `process_frame` DE VERDAD. Un helper que reimplantara las transiciones
    probaría su propia copia y seguiría pasando aunque el módulo estuviera mal —
    que es exactamente cómo se coló el bug que estos tests fijan.

    `gentle=0` desactiva la frase suave: los tests del reloj prueban el
    mecanismo de vigilancia, no la escalera (esa tiene los suyos más abajo).
    """
    from pipecat.processors.frame_processor import FrameDirection

    from app.voice.pipeline import ClinicalProcessor
    from app.voice.silence import SilenceConfig

    p = ClinicalProcessor(silence_config=SilenceConfig(
        initial_s=timeout, gentle_s=gentle, repeat_s=timeout))
    p.empujados = []

    async def _push(frame, direction=FrameDirection.DOWNSTREAM):
        p.empujados.append(frame)

    p.push_frame = _push
    return p


def _respuesta(texto="Listo, tomo nota. ¿Ha tenido fiebre?", call_ended=False):
    """Un `TurnResponse` mínimo para stubs de `_run_turn`."""
    from app.schemas import Symptoms, TurnResponse

    return TurnResponse(
        conversation_id="conv-test", turn_id="turn-test", response=texto,
        risk_level="NORMAL", triggered_rules=[], symptoms=Symptoms(),
        sources=[], critical_override=False, call_ended=call_ended,
        phase="tamizaje", slot_actual="fiebre",
    )


def _transcripcion(texto):
    from pipecat.frames.frames import TranscriptionFrame

    return TranscriptionFrame(text=texto, user_id="paciente", timestamp="t0")


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


# --- contratos de pipecat 1.7 que el barge-in necesita -------------------------


def test_pipeline_params_no_acepta_allow_interruptions():
    """Regresión-documento, gemela de la del `vad_analyzer` fantasma.

    `PipelineParams(allow_interruptions=True)` estuvo en este módulo y no hacía
    NADA: el campo no existe en pipecat 1.7 y pydantic lo descarta en silencio.
    Las interrupciones se activan con las estrategias de turno
    (`UserTurnProcessor`) y con `broadcast_interruption()`. Si este test falla,
    pipecat volvió a declarar el campo y hay que revisar el cableado.
    """
    from pipecat.pipeline.worker import PipelineParams

    assert "allow_interruptions" not in PipelineParams.model_fields, (
        "PipelineParams ahora SÍ declara allow_interruptions: revisar si basta "
        "con pasarlo aquí y retirar el broadcast manual del ClinicalProcessor."
    )


def test_el_mecanismo_de_interrupcion_de_pipecat_sigue_en_pie():
    """Fija el contrato del que depende el barge-in.

    `InterruptionFrame` es un `SystemFrame` (viaja con prioridad, no espera
    cola), `broadcast_interruption()` existe como API pública del procesador, y
    el transporte de salida tiene el `handle_interruptions` que drena la cola
    de audio (es lo que de verdad calla al agente).
    """
    from pipecat.frames.frames import InterruptionFrame, SystemFrame
    from pipecat.processors.frame_processor import FrameProcessor
    from pipecat.transports.base_output import BaseOutputTransport

    assert issubclass(InterruptionFrame, SystemFrame)
    assert hasattr(FrameProcessor, "broadcast_interruption")
    assert hasattr(BaseOutputTransport.MediaSender, "handle_interruptions")


def test_el_gestor_de_turnos_se_arma_sin_smart_turn():
    """El default de `UserTurnStrategies` instala un stop con modelo ML de
    HuggingFace (LocalSmartTurnAnalyzerV3): descarga en el arranque y PyTorch.
    El pipeline pasa `stop=` explícito precisamente para no arrastrarlo; esto
    fija que la construcción con las estrategias reales no importa torch.
    """
    import sys

    from pipecat.turns.user_start.vad_user_turn_start_strategy import (
        VADUserTurnStartStrategy,
    )
    from pipecat.turns.user_stop.speech_timeout_user_turn_stop_strategy import (
        SpeechTimeoutUserTurnStopStrategy,
    )
    from pipecat.turns.user_turn_processor import UserTurnProcessor
    from pipecat.turns.user_turn_strategies import UserTurnStrategies

    UserTurnProcessor(
        user_turn_strategies=UserTurnStrategies(
            start=[VADUserTurnStartStrategy(enable_interruptions=False)],
            stop=[SpeechTimeoutUserTurnStopStrategy(user_speech_timeout=0.2)],
        ),
    )
    assert "torch" not in sys.modules, "el gestor de turnos arrastró PyTorch"


# --- el reloj, gobernado por los frames FÍSICOS del VAD ------------------------
# Los `UserStarted/StoppedSpeakingFrame` de turno los emite el UserTurnProcessor
# y el de stop puede llegar SEGUNDOS tarde (espera transcript). El estado de
# "el paciente habla" tiene que salir de los frames del VAD, que llegan al
# instante, y aceptar los de turno como equivalentes sin contar doble.


def test_el_vad_fisico_gobierna_el_reloj_sin_contar_doble():
    import asyncio

    from pipecat.frames.frames import (
        UserStartedSpeakingFrame,
        UserStoppedSpeakingFrame,
        VADUserStartedSpeakingFrame,
        VADUserStoppedSpeakingFrame,
    )

    async def escenario():
        p = _procesador()
        await _meter(p, VADUserStartedSpeakingFrame())
        assert p._watchdog is None                     # noqa: SLF001
        assert p._paciente_hablando is True            # noqa: SLF001
        assert p._veces_que_oimos_al_paciente == 1     # noqa: SLF001

        # El frame de turno llega después: equivalente, pero no cuenta doble.
        await _meter(p, UserStartedSpeakingFrame())
        assert p._veces_que_oimos_al_paciente == 1     # noqa: SLF001

        await _meter(p, VADUserStoppedSpeakingFrame())
        assert p._paciente_hablando is False           # noqa: SLF001
        assert p._watchdog is not None                 # noqa: SLF001
        assert p._t_fin_habla is not None              # noqa: SLF001

        # Y el stop de turno tardío tampoco rompe nada.
        deadline = p._deadline                         # noqa: SLF001
        await _meter(p, UserStoppedSpeakingFrame())
        assert p._deadline == deadline                 # noqa: SLF001

    asyncio.run(escenario())


# --- escalera de silencios en el pipeline ---------------------------------------


def test_el_gentle_suena_local_sin_crear_turno():
    """El primer escalón es una frase suave que NO pasa por el orquestador:
    ni `Turn` en la base, ni contador de cierre, ni pregunta repetida."""
    import asyncio

    from pipecat.frames.frames import BotStoppedSpeakingFrame, TTSSpeakFrame

    from app.agent.phrasing import SILENCIO_SUAVE

    async def escenario():
        p = _procesador(timeout=0.05, gentle=1.0)

        def _no_debe_llamarse(text, interrumpida=False):  # noqa: ARG001
            raise AssertionError("el gentle no puede tocar el orquestador")

        p._run_turn = _no_debe_llamarse
        await _meter(p, BotStoppedSpeakingFrame())
        await asyncio.sleep(0.2)

        hablado = [f for f in p.empujados if isinstance(f, TTSSpeakFrame)]
        assert len(hablado) == 1
        assert hablado[0].text in SILENCIO_SUAVE
        assert "?" not in hablado[0].text              # no repite la pregunta
        # Y queda anotado como "lo último que dijo el agente": su eco por el
        # micrófono no puede colarse como un turno del paciente.
        assert p._ultima_respuesta == hablado[0].text  # noqa: SLF001

    asyncio.run(escenario())


def test_tras_el_gentle_el_siguiente_vencimiento_inyecta_silencio():
    """Gentle una vez por episodio; el siguiente escalón ya es el sondeo de
    presencia de siempre (`"[silencio]"` → escalera de `agent/script.py`)."""
    import asyncio

    from pipecat.frames.frames import BotStoppedSpeakingFrame

    async def escenario():
        p = _procesador(timeout=0.05, gentle=0.05)
        inyectados = []

        def _stub(text, interrumpida=False):  # noqa: ARG001
            inyectados.append(text)
            return _respuesta("¿Sigue ahí? No le escuché nada.")

        p._run_turn = _stub

        await _meter(p, BotStoppedSpeakingFrame())     # arranca el episodio
        await asyncio.sleep(0.15)                      # vence: gentle
        assert inyectados == []

        # El gentle terminó de sonar: el reloj se rearma con el tramo siguiente.
        await _meter(p, BotStoppedSpeakingFrame())
        await asyncio.sleep(0.15)                      # vence: sondeo
        assert inyectados == [p.SILENCIO]

    asyncio.run(escenario())


def test_una_respuesta_real_reinicia_la_escalera():
    """Un turno del paciente devuelve la escalera al tramo inicial: el gentle
    vuelve a estar disponible para el próximo episodio de silencio."""
    import asyncio

    from pipecat.frames.frames import BotStoppedSpeakingFrame

    from app.voice.silence import Escalon

    async def escenario():
        p = _procesador(timeout=30, gentle=30)
        p._run_turn = lambda text, interrumpida=False: _respuesta()
        await _meter(p, BotStoppedSpeakingFrame())
        assert p._escalera.al_vencer() is Escalon.GENTLE   # noqa: SLF001

        await _meter(p, _transcripcion("me duele como un cuatro"))
        await p._tarea_turno                               # noqa: SLF001
        assert p._escalera.al_vencer() is Escalon.GENTLE, (  # noqa: SLF001
            "el episodio no se reinició con el turno real"
        )

    asyncio.run(escenario())


def test_el_sondeo_se_descarta_si_el_paciente_arranca_mientras_se_generaba():
    """La re-verificación del requisito 'no hablarle encima a quien piensa':
    el turno de silencio toca base de datos (y red con Groq), y si en esa
    ventana el paciente empieza a hablar, el sondeo NO puede sonar."""
    import asyncio

    from pipecat.frames.frames import BotStoppedSpeakingFrame, TTSSpeakFrame

    async def escenario():
        p = _procesador(timeout=0.05, gentle=0)

        def _stub(text, interrumpida=False):  # noqa: ARG001
            p._paciente_hablando = True                # arrancó a hablar justo ahora
            return _respuesta("¿Sigue ahí? No le escuché nada.")

        p._run_turn = _stub
        await _meter(p, BotStoppedSpeakingFrame())
        await asyncio.sleep(0.2)

        assert not any(isinstance(f, TTSSpeakFrame) for f in p.empujados), (
            "el sondeo sonó por encima del paciente"
        )

    asyncio.run(escenario())


# --- barge-in -------------------------------------------------------------------


def test_una_transcripcion_valida_interrumpe_al_agente():
    """Barge-in confirmado: el paciente habla mientras el agente usa TTS, su
    transcripción pasa el filtro de eco, y ANTES de procesar el turno se emite
    la interrupción que drena el TTS. `_procesando` tiene que estar en True al
    interrumpir: el `BotStoppedSpeakingFrame` que provoca la interrupción no
    puede armar el reloj de silencio."""
    import asyncio

    from pipecat.frames.frames import BotStartedSpeakingFrame, TTSSpeakFrame

    async def escenario():
        p = _procesador()
        p._run_turn = lambda text, interrumpida=False: _respuesta()
        visto = {}

        async def _broadcast():
            visto["interrumpio"] = True
            visto["procesando"] = p._procesando        # noqa: SLF001
            visto["ya_hablo"] = any(
                isinstance(f, TTSSpeakFrame) for f in p.empujados)

        p.broadcast_interruption = _broadcast

        await _meter(p, BotStartedSpeakingFrame())     # el agente está hablando
        await _meter(p, _transcripcion("sí sí tuve fiebre anoche"))

        assert visto.get("interrumpio") is True
        assert visto.get("procesando") is True
        assert visto.get("ya_hablo") is False          # interrumpe ANTES de responder
        await p._tarea_turno                           # noqa: SLF001
        assert any(isinstance(f, TTSSpeakFrame) for f in p.empujados)

    asyncio.run(escenario())


def test_el_eco_no_dispara_barge_in():
    """El rebote del propio TTS por el micrófono NO puede cortar al agente:
    sería el agente interrumpiéndose a sí mismo a media frase, en bucle."""
    import asyncio

    from pipecat.frames.frames import BotStartedSpeakingFrame

    async def escenario():
        p = _procesador()
        p._ultima_respuesta = "¿Ha tenido fiebre o calentura estos días?"

        async def _no_debe_interrumpir():
            raise AssertionError("el eco interrumpió al agente")

        p.broadcast_interruption = _no_debe_interrumpir
        await _meter(p, BotStartedSpeakingFrame())
        await _meter(p, _transcripcion("ha tenido fiebre o calentura estos días"))
        assert p._tarea_turno is None                  # noqa: SLF001

    asyncio.run(escenario())


def test_un_fragmento_corto_de_eco_no_interrumpe():
    """Mientras el agente habla, el filtro de eco se endurece: un fragmento de
    tres palabras de su propia frase ('fiebre o calentura') pasaría el umbral
    normal de 4 y le cortaría el TTS. La respuesta real corta ('sí sí tuve
    fiebre') sigue pasando porque su vocabulario no solapa."""
    from app.voice.pipeline import es_eco

    pregunta = "¿Ha tenido fiebre o calentura estos días?"
    # Con el umbral normal el fragmento se colaría como turno...
    assert not es_eco("fiebre o calentura", pregunta)
    # ...con el endurecido del barge-in se filtra.
    assert es_eco("fiebre o calentura", pregunta, min_palabras=2)
    # Y la interrupción legítima no se pierde ni con el umbral duro.
    assert not es_eco("sí sí tuve fiebre", pregunta, min_palabras=2)


def test_la_mezcla_de_voces_no_se_descarta_como_eco():
    """Sin AEC, interrumpir al agente produce una transcripción con las DOS
    voces revueltas: el vocabulario del agente domina y el umbral de
    solapamiento descartaba al paciente entero — de ahí a la escalera de
    silencios y al cuelgue. Si hay residuo propio sustancial, es mezcla y se
    procesa; el eco puro (residuo ~0) se sigue descartando."""
    from app.voice.pipeline import es_eco

    pregunta = "¿Ha tenido fiebre o calentura estos días?"
    # Mezcla: la frase del agente + un "sí me duele" del paciente encima.
    assert not es_eco("ha tenido fiebre o calentura estos días sí me duele",
                      pregunta, min_palabras=2)
    # Eco puro, aunque el STT mute una palabra: se descarta como siempre.
    assert es_eco("ha tenido fiebre o calentura estos días", pregunta,
                  min_palabras=2)


def test_con_el_agente_callado_no_hay_interrupcion():
    import asyncio

    from pipecat.frames.frames import TTSSpeakFrame

    async def escenario():
        p = _procesador()
        p._run_turn = lambda text, interrumpida=False: _respuesta()

        async def _no_debe_interrumpir():
            raise AssertionError("no había a quién interrumpir")

        p.broadcast_interruption = _no_debe_interrumpir
        await _meter(p, _transcripcion("me duele como un cuatro"))
        await p._tarea_turno                           # noqa: SLF001
        assert any(isinstance(f, TTSSpeakFrame) for f in p.empujados)

    asyncio.run(escenario())


def test_el_turno_corre_en_tarea_propia():
    """Un `InterruptionFrame` cancela y recrea la tarea de proceso del
    procesador; si el turno corriera inline en `process_frame`, moriría a medio
    `await` con la base ya avanzada y la respuesta jamás sonaría. En tarea
    propia, `process_frame` retorna al instante y el turno completa igual."""
    import asyncio
    import time as _time

    from pipecat.frames.frames import TTSSpeakFrame

    async def escenario():
        p = _procesador()

        def _lento(text, interrumpida=False):  # noqa: ARG001
            _time.sleep(0.1)
            return _respuesta()

        p._run_turn = _lento
        await _meter(p, _transcripcion("me duele un berraco"))
        # `process_frame` ya volvió y el turno sigue en vuelo.
        assert p._tarea_turno is not None              # noqa: SLF001
        assert not p._tarea_turno.done()               # noqa: SLF001
        await p._tarea_turno                           # noqa: SLF001
        assert any(isinstance(f, TTSSpeakFrame) for f in p.empujados)

    asyncio.run(escenario())


# --- lo que el paciente alcanzó a oír antes de interrumpir -----------------------


def test_prefijo_pronunciado():
    from app.voice.pipeline import prefijo_pronunciado

    texto = "uno dos tres cuatro cinco seis siete ocho nueve diez once doce"
    # El TTS nunca llegó a arrancar: no sonó nada.
    assert prefijo_pronunciado(texto, 0) == ""
    assert prefijo_pronunciado(texto, -1) == ""
    # ~1 s de audio: 2.6 palabras/s + margen de 4 por el audio en vuelo.
    assert prefijo_pronunciado(texto, 1.0) == "uno dos tres cuatro cinco seis"
    # Sonó de sobra: se conserva entero.
    assert prefijo_pronunciado(texto, 60) == texto


def test_tras_un_corte_el_anti_eco_solo_ve_lo_pronunciado():
    """Bug D: `_ultima_respuesta` guardaba la frase COMPLETA aunque el barge-in
    la cortara en la palabra cinco. El anti-eco descartaba entonces respuestas
    del paciente que solapaban con la COLA nunca pronunciada."""
    from app.voice.pipeline import es_eco, prefijo_pronunciado

    pregunta = ("Listo, tomo nota. ¿Ha notado la herida enrojecida, inflamada "
                "o con alguna secreción o líquido saliendo de ella?")
    # Contra el texto completo, la respuesta que repite la cola se descartaría...
    assert es_eco("alguna secreción o líquido saliendo", pregunta, min_palabras=2)
    # ...contra lo que de verdad sonó (corte a ~0.5 s), se procesa como turno.
    cortada = prefijo_pronunciado(pregunta, 0.5)
    assert not es_eco("alguna secreción o líquido saliendo", cortada, min_palabras=2)


def test_el_barge_in_marca_el_turno_como_interrumpido():
    """La marca viaja SOLO con el turno que interrumpió: el orquestador no puede
    leer un "sí/no" pelado contra una pregunta que el paciente no oyó entera,
    pero el turno siguiente vuelve a la normalidad."""
    import asyncio

    from pipecat.frames.frames import BotStartedSpeakingFrame, BotStoppedSpeakingFrame

    async def escenario():
        p = _procesador()
        marcas = []

        def _stub(text, interrumpida=False):  # noqa: ARG001
            marcas.append(interrumpida)
            return _respuesta()

        p._run_turn = _stub

        async def _broadcast():
            pass

        p.broadcast_interruption = _broadcast

        completa = "¿Ha podido levantarse y caminar sin problema por la casa?"
        p._ultima_respuesta = completa
        await _meter(p, BotStartedSpeakingFrame())
        await _meter(p, _transcripcion("espere le pregunto a mi hija"))
        # El anti-eco quedó comparando contra lo poco que llegó a sonar...
        assert p._ultima_respuesta != completa         # noqa: SLF001
        await p._tarea_turno                           # noqa: SLF001
        # ...y el turno bajó al orquestador marcado como interrumpido.
        assert marcas == [True]

        # Turno siguiente, con el agente ya callado: sin marca.
        await _meter(p, BotStoppedSpeakingFrame())
        await _meter(p, _transcripcion("sí ha caminado sin problema"))
        await p._tarea_turno                           # noqa: SLF001
        assert marcas == [True, False]

    asyncio.run(escenario())


def test_una_tos_no_corta_el_tts():
    """Compuerta de ruido del barge-in: una transcripción de basura fonotáctica
    mientras el agente habla no interrumpe ni se convierte en turno."""
    import asyncio

    from pipecat.frames.frames import BotStartedSpeakingFrame

    async def escenario():
        p = _procesador()

        async def _no_debe_interrumpir():
            raise AssertionError("el ruido interrumpió al agente")

        p.broadcast_interruption = _no_debe_interrumpir
        await _meter(p, BotStartedSpeakingFrame())
        await _meter(p, _transcripcion("asdkjh"))
        assert p._tarea_turno is None                  # noqa: SLF001

    asyncio.run(escenario())


# --- muletillas: el paciente que dice "ehh..." está pensando ---------------------


def test_la_muletilla_no_crea_turno_y_deja_pensar():
    """El VAD corta a "Ehh..." tras ~1 s de pausa y eso llegaba como turno: el
    agente contestaba "¿me lo repite?" encima del paciente que seguía pensando
    y quemaba una de las dos repreguntas del slot. Ahora no es un turno: no
    baja al orquestador, el reloj sigue vigilando y la escalera NO se resetea
    (el episodio de silencio continúa: lo próximo es la frase suave, no un
    segundo gentle)."""
    import asyncio

    from pipecat.frames.frames import BotStoppedSpeakingFrame, TTSSpeakFrame

    async def escenario():
        p = _procesador(timeout=30, gentle=30)

        def _no_debe_llamarse(text, interrumpida=False):  # noqa: ARG001
            raise AssertionError("la muletilla no puede tocar el orquestador")

        p._run_turn = _no_debe_llamarse
        await _meter(p, BotStoppedSpeakingFrame())     # el agente hizo la pregunta
        p._escalera._gentle_emitido = True             # noqa: SLF001 — gentle ya sonó

        await _meter(p, _transcripcion("Ehh..."))
        assert p._tarea_turno is None                  # noqa: SLF001
        assert p._watchdog is not None                 # noqa: SLF001 — sigue vigilando
        assert not any(isinstance(f, TTSSpeakFrame) for f in p.empujados), (
            "le habló encima a quien estaba pensando"
        )
        # La escalera no se reseteó: el gentle NO vuelve a estar disponible.
        assert p._escalera._gentle_emitido is True     # noqa: SLF001

        # Cuando por fin habla, su turno se procesa con normalidad.
        p._run_turn = lambda text, interrumpida=False: _respuesta()
        await _meter(p, _transcripcion("me duele como un cuatro"))
        await p._tarea_turno                           # noqa: SLF001
        assert any(isinstance(f, TTSSpeakFrame) for f in p.empujados)

    asyncio.run(escenario())


def test_la_muletilla_no_interrumpe_al_agente():
    """Un "a ver..." de backchannel mientras el agente habla no corta el TTS."""
    import asyncio

    from pipecat.frames.frames import BotStartedSpeakingFrame

    async def escenario():
        p = _procesador()

        async def _no_debe_interrumpir():
            raise AssertionError("la muletilla interrumpió al agente")

        p.broadcast_interruption = _no_debe_interrumpir
        await _meter(p, BotStartedSpeakingFrame())
        await _meter(p, _transcripcion("a ver..."))
        assert p._tarea_turno is None                  # noqa: SLF001

    asyncio.run(escenario())


# --- contexto: un turno fallido no borra la conversación -------------------------


def test_un_turno_fallido_no_pierde_la_conversacion(monkeypatch):
    """Si el primer turno revienta (red del LLM caída), el id de la conversación
    ya tiene que estar fijado. Sin esto, `_conversation_id` quedaba en `None`, el
    turno siguiente creaba una conversación NUEVA y el guion volvía a empezar
    desde cero: la "pérdida total de contexto" que se veía tras un fallo."""
    import asyncio

    from pipecat.frames.frames import TTSSpeakFrame

    from app.agent import phrasing
    from app.voice import pipeline as vp

    class _Session:
        def close(self):
            pass

    monkeypatch.setattr(vp, "SessionLocal", _Session)
    monkeypatch.setattr(vp, "ensure_conversation",
                        lambda session, conversation_id, patient_id: "conv-1")

    recibidos = []

    def _explota(session, *, text, conversation_id, patient_id,
                 pregunta_interrumpida=False):
        recibidos.append(conversation_id)
        raise RuntimeError("groq caído")

    monkeypatch.setattr(vp, "process_turn", _explota)

    async def escenario():
        p = _procesador()
        await _meter(p, _transcripcion("me duele como un cuatro"))
        await p._tarea_turno                           # noqa: SLF001
        # El id se fijó ANTES de procesar y sobrevivió al fallo...
        assert p._conversation_id == "conv-1"          # noqa: SLF001
        assert recibidos == ["conv-1"]
        # ...y el paciente no se quedó con un teléfono mudo.
        assert any(isinstance(f, TTSSpeakFrame)
                   and f.text == phrasing.FALLO_TECNICO for f in p.empujados)

        # El segundo turno reutiliza la MISMA conversación.
        def _sano(session, *, text, conversation_id, patient_id,
                  pregunta_interrumpida=False):
            recibidos.append(conversation_id)
            return _respuesta()

        monkeypatch.setattr(vp, "process_turn", _sano)
        await _meter(p, _transcripcion("no señorita, fiebre no he tenido"))
        await p._tarea_turno                           # noqa: SLF001
        assert recibidos[-1] == "conv-1"

    asyncio.run(escenario())

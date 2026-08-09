"""La escalera de silencios y los eventos de voz — la parte pura, sin pipecat.

`SilenceLadder` decide QUÉ toca cuando el reloj de inactividad vence (frase
suave local o sondeo por el orquestador) y cuánto esperar hasta el siguiente
escalón. Vive separada del reloj precisamente para poder probarse así: con
llamadas directas, sin frames ni event loop.
"""
from __future__ import annotations

from app.agent import phrasing
from app.voice.silence import Escalon, SilenceConfig, SilenceLadder, VoiceEvent, emit

_CONFIG = SilenceConfig(initial_s=6.0, gentle_s=6.0, repeat_s=8.0)


def test_la_escalera_espera_el_tramo_inicial_sin_hacer_nada():
    escalera = SilenceLadder(_CONFIG)
    assert escalera.siguiente_espera() == 6.0
    assert not escalera.en_episodio()


def test_el_primer_vencimiento_es_gentle_y_los_siguientes_sondean():
    """La frase suave suena UNA vez por episodio: después de un "¿sigue ahí?",
    un "tómese su tiempo" sería incoherente."""
    escalera = SilenceLadder(_CONFIG)
    assert escalera.al_vencer() is Escalon.GENTLE
    assert escalera.siguiente_espera() == 6.0    # tras el gentle, gentle_s
    assert escalera.al_vencer() is Escalon.SONDEO
    assert escalera.sondeos == 1
    assert escalera.siguiente_espera() == 8.0    # entre sondeos, repeat_s
    assert escalera.al_vencer() is Escalon.SONDEO
    assert escalera.sondeos == 2


def test_un_turno_real_devuelve_la_escalera_al_inicio():
    escalera = SilenceLadder(_CONFIG)
    escalera.al_vencer()                          # gentle gastado
    escalera.al_vencer()                          # y un sondeo
    escalera.reset()
    assert escalera.siguiente_espera() == 6.0     # tramo inicial otra vez
    assert escalera.sondeos == 0
    assert not escalera.en_episodio()
    assert escalera.al_vencer() is Escalon.GENTLE  # el gentle vuelve a estar disponible


def test_gentle_desactivado_con_gentle_s_cero():
    """`SILENCE_GENTLE_S=0` recupera el comportamiento anterior: el primer
    vencimiento va directo al sondeo de presencia."""
    escalera = SilenceLadder(SilenceConfig(initial_s=8.0, gentle_s=0.0, repeat_s=8.0))
    assert escalera.al_vencer() is Escalon.SONDEO
    assert escalera.sondeos == 1


def test_la_duracion_del_episodio_se_mide_desde_su_inicio():
    escalera = SilenceLadder(_CONFIG)
    assert escalera.duracion_ms() == 0            # sin episodio no hay duración
    escalera.marca_inicio()
    assert escalera.en_episodio()
    assert escalera.duracion_ms() >= 0
    escalera.reset()
    assert escalera.duracion_ms() == 0


# --- frases suaves ---------------------------------------------------------


def test_las_frases_suaves_son_deterministas_y_rotan():
    a = phrasing.silencio_suave("semilla-1")
    assert a == phrasing.silencio_suave("semilla-1")   # reproducible en tests
    assert a in phrasing.SILENCIO_SUAVE
    # La rotación evita repetir lo dicho hace poco (comparación por contención,
    # como el resto de phrasing._rotar).
    b = phrasing.silencio_suave("semilla-1", usadas=[a])
    assert b != a


def test_las_frases_suaves_no_preguntan_nada():
    """El gentle da permiso para pensar; una pregunta lo convertiría en otro
    sondeo de presencia y presionaría justo a quien se quiere dejar pensar."""
    for frase in phrasing.SILENCIO_SUAVE:
        assert "?" not in frase, frase


def test_las_frases_suaves_se_pre_sintetizan():
    cacheables = phrasing.textos_cacheables()
    for frase in phrasing.SILENCIO_SUAVE:
        assert frase in cacheables


# --- eventos de voz ---------------------------------------------------------


def test_los_eventos_de_voz_llevan_campos_estructurados():
    """Con estos campos en `record.extra` se reconstruye qué pasó en la llamada
    (cuánto duró el silencio, en qué pregunta del guion, qué intento era) sin
    adivinar desde mensajes sueltos."""
    from loguru import logger

    capturados = []
    sink = logger.add(lambda m: capturados.append(m.record), level="INFO")
    try:
        emit(VoiceEvent.SILENCE_PROMPT_TRIGGERED, conversation_id="c-1",
             duration_ms=6100, attempt=1, stage="sondeo",
             phase="tamizaje", slot="fiebre")
        emit(VoiceEvent.CONNECTION_LOST, conversation_id="c-1")
    finally:
        logger.remove(sink)

    eventos = [r for r in capturados if r["extra"].get("voice_event")]
    assert len(eventos) == 2
    prompt, caida = eventos
    assert prompt["extra"]["voice_event"] == "SILENCE_PROMPT_TRIGGERED"
    assert prompt["extra"]["duration_ms"] == 6100
    assert prompt["extra"]["attempt"] == 1
    assert prompt["extra"]["stage"] == "sondeo"
    assert prompt["extra"]["current_script_state"] == "tamizaje"
    assert prompt["extra"]["slot"] == "fiebre"
    assert caida["extra"]["voice_event"] == "CONNECTION_LOST"
    # Los campos ausentes no viajan como None: menos ruido en un sink JSON.
    assert "duration_ms" not in caida["extra"]


def test_la_config_de_barge_in_y_silencios_existe_en_settings():
    from app.config import Settings

    campos = Settings.model_fields
    for nombre in ("silence_initial_s", "silence_gentle_s", "silence_repeat_s",
                   "silence_max_attempts", "barge_in_enabled", "barge_in_vad",
                   "barge_in_min_palabras_eco"):
        assert nombre in campos, nombre
    assert campos["barge_in_vad"].default is False, (
        "el barge-in por VAD debe ser opt-in: sin AEC fiable el agente se "
        "interrumpe a sí mismo con su propio eco"
    )

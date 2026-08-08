"""La máquina de estados del guion — funciones puras, sin LLM ni base de datos.

El caso central de este archivo es el que reportó un usuario probando la voz:
tras escalar por un dolor severo, si el paciente seguía hablando, el agente
repetía el guion de seguridad completo palabra por palabra, indefinidamente. La
causa era que `next_action` solo miraba `escalar` (que se queda en True para
siempre, porque el cuadro crítico no se "des-escala" dentro de la llamada) y
nunca miraba si el guion ya se había entregado en un turno anterior.
"""
from __future__ import annotations

from app.agent import script
from app.agent.script import Action, CallState, Phase, apply, next_action
from app.schemas import Symptoms

CRITICO = Symptoms(pain_level=9)  # dispara escalar=True vía el motor de decisión


def test_primer_turno_critico_escala():
    estado = CallState()
    a = next_action(estado, CRITICO, escalar=True)
    assert a.kind == "escalar"
    assert a.phase is Phase.ESCALAMIENTO


def test_segundo_turno_critico_no_repite_el_guion():
    """El bug reportado: sin este corte, esto seguiría devolviendo 'escalar'.

    Ya no cierra: por la vía de enfermería la llamada pasa a CONFIRMACIÓN y espera
    a que el paciente diga que entendió. Lo que sí se conserva —y es el bug
    original— es que el guion NO se vuelve a entregar.
    """
    tras_escalar = apply(CallState(), Action(kind="escalar", phase=Phase.ESCALAMIENTO), CRITICO)
    assert tras_escalar.phase is Phase.ESCALAMIENTO

    a = next_action(tras_escalar, CRITICO, escalar=True)
    assert a.kind != "escalar", "el guion de seguridad no puede repetirse"
    assert a.kind == "confirmar"
    assert a.phase is Phase.CONFIRMACION


def test_en_emergencia_123_si_se_cuelga_rapido():
    """Retener al paciente en la línea compite con la llamada al 123."""
    tras_escalar = apply(CallState(), Action(kind="escalar", phase=Phase.ESCALAMIENTO), CRITICO)
    a = next_action(tras_escalar, CRITICO, escalar=True, emergencia=True)
    assert a.kind == "cerrar"
    assert a.phase is Phase.TERMINADA


def test_la_confirmacion_cierra_cuando_el_paciente_se_despide():
    estado = CallState(phase=Phase.CONFIRMACION)
    assert next_action(estado, CRITICO, quiere_colgar=True).kind == "cerrar"
    # Y mientras no se despida, se le sigue atendiendo.
    assert next_action(estado, CRITICO).kind == "confirmar"


def test_la_confirmacion_no_se_eterniza():
    """Tope de seguridad: sin él, ruido de STT dejaría la llamada abierta siempre."""
    estado = CallState(phase=Phase.CONFIRMACION, sin_progreso=script.SIN_PROGRESO_CERRAR)
    assert next_action(estado, CRITICO).kind == "cerrar"


def test_el_guion_agotado_pregunta_antes_de_colgar():
    """Terminar la lista de preguntas no es terminar la llamada."""
    completo = Symptoms(pain_level=2, fever=False, mobility="normal", wound="normal",
                        appetite="normal", sleep="normal")
    estado = CallState(phase=Phase.ABIERTO, slot_actual=None)
    assert next_action(estado, completo).kind == "confirmar"
    assert next_action(estado, completo, quiere_colgar=True).kind == "cerrar"


def test_tercer_turno_sigue_cerrado_no_revive_el_guion():
    """Una vez en TERMINADA, se queda ahí aunque el paciente siga hablando."""
    cerrada = CallState(phase=Phase.TERMINADA)
    a = next_action(cerrada, CRITICO, escalar=True)
    assert a.kind == "cerrar"
    assert a.phase is Phase.TERMINADA


def test_una_llamada_no_critica_nunca_pasa_por_escalamiento():
    estado = CallState()
    a = next_action(estado, Symptoms(pain_level=2), escalar=False)
    assert a.kind != "escalar"


# --- seguimiento: el paciente contestó, pero a medias -------------------------

def _estado_fiebre():
    return CallState(phase=Phase.TAMIZAJE, slot_actual="fiebre")


def test_dijo_que_se_la_tomo_y_se_le_pide_la_cifra():
    """El falso negativo que cerraba: sin la cifra, un 38.5 real no dispara fiebre_38."""
    s = Symptoms(temperature_measured=True)
    a = script.next_action(_estado_fiebre(), s)
    assert a.kind == "seguimiento"
    assert a.seguimiento == "cifra"


def test_el_seguimiento_gana_a_la_repregunta():
    """Reformular "¿se ha sentido caliente?" a quien ya dijo "sí me la tomé" la ignora."""
    s = Symptoms(temperature_measured=True)
    estado = _estado_fiebre()
    assert script.next_action(estado, s).kind == "seguimiento"
    # …y no consume intentos de repregunta.
    nuevo = script.apply(estado, script.next_action(estado, s), s)
    assert nuevo.repreguntas.get("fiebre", 0) == 0


def test_calentura_sin_medir_pregunta_por_el_termometro():
    a = script.next_action(_estado_fiebre(), Symptoms(fever=True))
    assert a.kind == "seguimiento"
    assert a.seguimiento == "termometro"


def test_sin_termometro_se_acepta_y_se_avanza():
    """Es una respuesta legítima: mucha gente no tiene termómetro."""
    s = Symptoms(fever=True, temperature_measured=False)
    assert script.seguimiento_pendiente(_estado_fiebre(), s) is None


def test_con_la_cifra_ya_no_se_persigue_nada():
    s = Symptoms(fever=True, temperature_c=38.5, temperature_measured=True)
    assert script.seguimiento_pendiente(_estado_fiebre(), s) is None


def test_solo_un_seguimiento_por_slot():
    """Perseguir dos veces el mismo dato es el interrogatorio que se quiere evitar."""
    estado = _estado_fiebre()
    s = Symptoms(temperature_measured=True)
    estado = script.apply(estado, script.next_action(estado, s), s)
    assert script.seguimiento_pendiente(estado, s) is None


# --- atasco: cambiar de estrategia, no reformular otra vez --------------------

def test_repetir_la_misma_frase_no_gasta_otra_repregunta():
    estado = CallState(phase=Phase.TAMIZAJE, slot_actual="dolor")
    assert script.next_action(estado, Symptoms()).kind == "repreguntar"
    assert script.next_action(estado, Symptoms(), repetido=True).kind != "repreguntar"


def test_varios_turnos_sin_dato_ofrecen_una_salida():
    estado = CallState(phase=Phase.TAMIZAJE, slot_actual="dolor",
                       sin_progreso=script.SIN_PROGRESO_OFRECER_SALIDA)
    a = script.next_action(estado, Symptoms())
    assert a.kind == "ofrecer_salida"
    # No abandona el slot: si el paciente dice que sí quiere seguir, se retoma.
    assert a.slot == "dolor"


def test_el_atasco_persistente_cierra_la_llamada():
    estado = CallState(phase=Phase.TAMIZAJE, slot_actual="dolor",
                       sin_progreso=script.SIN_PROGRESO_CERRAR)
    assert script.next_action(estado, Symptoms()).kind == "cerrar"


def test_el_atasco_se_reinicia_cuando_hay_dato():
    estado = CallState(phase=Phase.TAMIZAJE, slot_actual="dolor", sin_progreso=4)
    a = Action(kind="preguntar", slot="fiebre")
    assert script.apply(estado, a, Symptoms(pain_level=3), progreso=True).sin_progreso == 0
    assert script.apply(estado, a, Symptoms(), progreso=False).sin_progreso == 5


def test_el_estado_nuevo_sobrevive_a_la_serializacion():
    """Vive en la BD entre turnos: si no round-trippea, el atasco no se detecta."""
    estado = CallState(phase=Phase.TAMIZAJE, slot_actual="fiebre",
                       seguido=["fiebre"], ultimo_hash="abc123", sin_progreso=2)
    vuelta = CallState.from_dict(estado.to_dict())
    assert vuelta.seguido == ["fiebre"]
    assert vuelta.ultimo_hash == "abc123"
    assert vuelta.sin_progreso == 2


def test_un_saludo_no_gasta_un_reintento():
    """Reformular es para quien esquivó la pregunta, no para quien no la ha oído."""
    estado = CallState(phase=Phase.TAMIZAJE, slot_actual="dolor")
    accion = next_action(estado, Symptoms())
    assert accion.kind == "repreguntar"
    tras_saludo = apply(estado, accion, Symptoms(), intento_real=False)
    assert tras_saludo.repreguntas.get("dolor", 0) == 0
    # Una evasiva real sí lo gasta.
    tras_evasiva = apply(estado, accion, Symptoms(), intento_real=True)
    assert tras_evasiva.repreguntas["dolor"] == 1

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


def test_el_turno_que_trae_un_dato_no_recibe_la_oferta_de_salida():
    """`state.sin_progreso` cuenta hasta el turno ANTERIOR: `apply` es quien suma.

    Sin mirar el `progreso` de ESTE turno, un paciente que acababa de contestar
    recibía "le noto que le cuesta contestarme". Medido en la llamada reportada,
    turno 5: dijo "Me cuesta un poco más.", el dato se extrajo, y aun así le
    ofrecieron una enfermera.
    """
    atascado = CallState(phase=Phase.TAMIZAJE, slot_actual="movilidad",
                         sin_progreso=script.SIN_PROGRESO_OFRECER_SALIDA)

    assert script.next_action(atascado, Symptoms()).kind == "ofrecer_salida"
    con_dato = script.next_action(
        atascado, Symptoms(mobility="limitada_esperada"), progreso=True)
    assert con_dato.kind != "ofrecer_salida"

    # Y lo mismo con el cierre por atasco, que es el desenlace más caro.
    a_punto_de_cerrar = CallState(phase=Phase.TAMIZAJE, slot_actual="movilidad",
                                  sin_progreso=script.SIN_PROGRESO_CERRAR)
    assert script.next_action(a_punto_de_cerrar, Symptoms()).kind == "cerrar"
    assert script.next_action(a_punto_de_cerrar,
                              Symptoms(mobility="limitada_esperada"),
                              progreso=True).kind != "cerrar"


def test_la_salida_se_ofrece_una_sola_vez():
    """Antes se repetía al turno siguiente con otra formulación.

    Medido en una llamada real: "le noto que le cuesta contestarme" y, al turno
    siguiente, "veo que no nos estamos entendiendo" — a un paciente que estaba
    contestando todas las preguntas. Es el mismo interrogatorio del que la oferta
    pretende sacar al agente.
    """
    ofrecida = CallState(phase=Phase.TAMIZAJE, slot_actual="dolor",
                         sin_progreso=script.SIN_PROGRESO_OFRECER_SALIDA + 1,
                         salida_ofrecida=True)
    assert script.next_action(ofrecida, Symptoms()).kind != "ofrecer_salida"


def test_emitir_la_oferta_deja_la_llamada_esperando_su_respuesta():
    estado = CallState(phase=Phase.TAMIZAJE, slot_actual="dolor",
                       sin_progreso=script.SIN_PROGRESO_OFRECER_SALIDA)
    nuevo = script.apply(estado, script.next_action(estado, Symptoms()), Symptoms(),
                         progreso=False)
    assert nuevo.espera_respuesta_salida and nuevo.salida_ofrecida
    # Y la siguiente acción ya no la espera.
    otra = script.apply(nuevo, Action(kind="preguntar", slot="fiebre"), Symptoms())
    assert not otra.espera_respuesta_salida
    assert otra.salida_ofrecida, "el hecho de haberla ofrecido no se olvida"


def test_declinar_la_enfermera_retoma_el_slot_pendiente():
    """El paciente dice que quiere seguir: se le hace caso.

    Y el contador de atasco vuelve a cero — arrancar de nuevo con él en 4 lo
    llevaría al cierre en el turno siguiente, que es justo lo que rechazó.
    """
    estado = CallState(phase=Phase.TAMIZAJE, slot_actual="herida",
                       sin_progreso=4, salida_ofrecida=True,
                       espera_respuesta_salida=True)
    a = script.next_action(estado, Symptoms(), reanudar=True)

    assert a.kind == "preguntar" and a.slot == "herida"
    assert script.apply(estado, a, Symptoms(), progreso=False,
                        reanudar=True).sin_progreso == 0


def test_aceptar_la_enfermera_cierra_la_llamada():
    """El orquestador lo traduce a `quiere_colgar`: la oferta se cumple."""
    estado = CallState(phase=Phase.TAMIZAJE, slot_actual="herida",
                       sin_progreso=4, salida_ofrecida=True,
                       espera_respuesta_salida=True)
    assert script.next_action(estado, Symptoms(),
                              quiere_colgar=True).kind == "cerrar"


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


# --- tope de la fase de CONFIRMACIÓN ------------------------------------------
# Bug medido en una llamada real: el paciente dijo "no, nada más" cuatro veces y
# el agente siguió alternando "¿Hay algo más...?" / "¿Quedamos así...?" porque
# la fase solo salía con el regex de despedida o con 5 turnos sin progreso — y
# cualquier dato nuevo reseteaba ese contador.


def test_confirmar_gasta_un_turno_del_tope():
    estado = CallState(phase=Phase.CONFIRMACION, slot_actual=None)
    accion = next_action(estado, Symptoms())
    assert accion.kind == "confirmar"
    nuevo = apply(estado, accion, Symptoms(), progreso=False)
    assert nuevo.turnos_confirmacion == 1


def test_la_confirmacion_cierra_al_alcanzar_el_tope():
    estado = CallState(phase=Phase.CONFIRMACION, slot_actual=None)
    for esperado in range(1, script.MAX_TURNOS_CONFIRMACION + 1):
        accion = next_action(estado, Symptoms())
        assert accion.kind == "confirmar", f"turno {esperado} debía confirmar"
        estado = apply(estado, accion, Symptoms(), progreso=False)
        assert estado.turnos_confirmacion == esperado
    assert next_action(estado, Symptoms()).kind == "cerrar"


def test_el_tope_aplica_aunque_el_paciente_aporte_datos():
    """`progreso=True` resetea `sin_progreso`, pero NO el tope de confirmación:
    esa era exactamente la vía por la que el bucle no terminaba nunca."""
    estado = CallState(phase=Phase.CONFIRMACION, slot_actual=None,
                       turnos_confirmacion=script.MAX_TURNOS_CONFIRMACION)
    assert next_action(estado, Symptoms()).kind == "cerrar"


def test_el_tope_tambien_cierra_en_escalamiento():
    estado = CallState(phase=Phase.ESCALAMIENTO, slot_actual=None,
                       turnos_confirmacion=script.MAX_TURNOS_CONFIRMACION)
    assert next_action(estado, CRITICO).kind == "cerrar"


# --- el guion de seguridad se entrega UNA vez ----------------------------------


def test_guion_entregado_persiste_entre_turnos_y_serializacion():
    """El flag es lo que impide que `evaluate(final=True)` re-dispare el guion
    completo: la fase avanza (ESCALAMIENTO → CONFIRMACION) pero el hecho no se
    olvida, y tiene que sobrevivir el viaje por la base de datos."""
    estado = CallState(phase=Phase.ESCALAMIENTO)
    accion = Action(kind="confirmar", phase=Phase.CONFIRMACION)
    tras_guion = apply(estado, accion, Symptoms(), guion_entregado=True)
    assert tras_guion.guion_entregado is True

    vuelta = CallState.from_dict(tras_guion.to_dict())
    assert vuelta.guion_entregado is True
    assert vuelta.turnos_confirmacion == tras_guion.turnos_confirmacion

    # Y no se pierde en turnos posteriores aunque nadie lo vuelva a pasar.
    despues = apply(vuelta, accion, Symptoms())
    assert despues.guion_entregado is True

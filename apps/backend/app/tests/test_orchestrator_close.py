"""El orquestador cierra la llamada tras un escalamiento crítico.

Reproduce el bug reportado probando la voz: al escalar por un dolor severo, si
el paciente seguía hablando, el agente repetía el guion de seguridad completo
en cada turno, indefinidamente, porque el cuadro crítico nunca se "des-escala"
dentro de la llamada y nada distinguía el primer turno del segundo.

Usa una frase que el léxico resuelve solo ("me duele un berraco" → pain_level=9,
ver app/nlu/lexicon.py), así que corre sin Ollama y sin red.
"""
from __future__ import annotations

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.agent.orchestrator import process_turn
from app.db import Base
from app.models import Conversation, Turn  # noqa: F401  — registra las tablas


@pytest.fixture()
def session(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path}/test.db", future=True)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine, expire_on_commit=False, future=True)
    with Session() as s:
        yield s


def test_ensure_conversation_sobrevive_a_un_turno_fallido(session):
    """El id queda commiteado ANTES de procesar: aunque el turno reviente a
    mitad y la sesión haga rollback, la conversación sigue existiendo y el
    turno siguiente conserva el contexto (ver `voice/pipeline._run_turn`)."""
    from app.agent.orchestrator import ensure_conversation

    cid = ensure_conversation(session, None, None)
    session.rollback()  # el turno reventó a mitad
    r = process_turn(session, text="me duele poquito", conversation_id=cid)
    assert r.conversation_id == cid


def test_el_primer_turno_critico_entrega_el_guion_completo(session):
    r = process_turn(session, text="Me duele un berraco, no aguanto")
    assert r.risk_level == "CRÍTICO"
    assert r.critical_override is True
    assert r.call_ended is False
    assert "berraco" not in r.response  # es el guion determinista, no un eco


def test_el_segundo_turno_critico_no_repite_el_guion(session):
    """El bug original: el guion de seguridad se repetía palabra por palabra.

    Ya no se cuelga en el segundo turno —por la vía de enfermería la llamada
    espera a que el paciente confirme—, pero la propiedad que motivó este archivo
    sigue siendo la misma: el guion se entrega UNA vez.
    """
    primero = process_turn(session, text="Me duele un berraco, no aguanto")
    segundo = process_turn(
        session, text="Sigo con el mismo dolor", conversation_id=primero.conversation_id
    )

    assert segundo.response != primero.response
    assert len(segundo.response) < len(primero.response), (
        "no puede ser el guion clínico completo otra vez"
    )
    assert segundo.call_ended is False, "no se cuelga sin que el paciente confirme"


def test_el_paciente_puede_seguir_hablando_tras_escalar(session):
    """Lo que se perdía: tras el guion, un "y la herida se ve rojita" no se oía."""
    primero = process_turn(session, text="Me duele un berraco, no aguanto")
    segundo = process_turn(session, text="Y la herida se ve rojita.",
                           conversation_id=primero.conversation_id)

    assert segundo.call_ended is False
    assert segundo.symptoms.wound == "eritema_leve", "la señal nueva se recoge"


def test_la_conversacion_se_cierra_cuando_el_paciente_se_despide(session):
    primero = process_turn(session, text="Me duele un berraco, no aguanto")
    process_turn(session, text="Ok", conversation_id=primero.conversation_id)
    ultimo = process_turn(session, text="Listo, muchas gracias, chao",
                          conversation_id=primero.conversation_id)

    assert ultimo.call_ended is True
    conv = session.get(Conversation, primero.conversation_id)
    assert conv.status == "closed"
    assert conv.ended_at is not None


def test_el_resumen_queda_listo_sin_llamar_a_close_aparte(session):
    """RF-10: el resumen se genera al cerrar, no depende de un paso extra."""
    from app.models import Summary

    primero = process_turn(session, text="Me duele un berraco, no aguanto")
    process_turn(session, text="Ya entendí, gracias, chao",
                 conversation_id=primero.conversation_id)

    resumen = session.query(Summary).filter(
        Summary.conversation_id == primero.conversation_id
    ).first()
    assert resumen is not None
    assert resumen.data["risk_level"] == "CRÍTICO"


def test_turnos_posteriores_no_reabren_el_guion(session):
    """Si el paciente insiste, no revive el guion clínico largo."""
    primero = process_turn(session, text="Me duele un berraco, no aguanto")
    for texto in ("Espere", "Otra cosa", "Ajá"):
        r = process_turn(session, text=texto, conversation_id=primero.conversation_id)
        assert r.response != primero.response, f"el guion revivió tras {texto!r}"


def test_una_llamada_normal_no_se_cierra_sola(session):
    r = process_turn(session, text="Un 2, apenas se nota")
    assert r.risk_level != "CRÍTICO"
    assert r.call_ended is False


def test_guion_agotado_normalmente_cierra_la_llamada(session):
    """Antes de Fix 5b, `call_ended` exigía `critico`: una llamada que termina
    por agotar el guion (sin pasar por CRÍTICO) nunca llegaba a
    `close_conversation`, y la política de incertidumbre (`final=True`, ver
    `summary/service.py`) nunca se aplicaba porque nada la disparaba.
    """
    from app.models import Summary

    respuestas = [
        "Un 2, apenas se nota, tranquilo.",   # dolor
        "No tengo fiebre, nada.",              # fiebre
        "Camino bien, sin problema.",          # movilidad
        "La herida se ve bien.",               # herida
        "Como bien, normal.",                  # apetito
        "Duermo bien, tranquilo.",              # sueño
        "No, nada más, gracias.",              # pregunta abierta -> cierra
    ]
    conversation_id = None
    ultimo = None
    for texto in respuestas:
        ultimo = process_turn(session, text=texto, conversation_id=conversation_id)
        conversation_id = ultimo.conversation_id

    assert ultimo.call_ended is True
    assert ultimo.risk_level != "CRÍTICO"

    resumen = session.query(Summary).filter(
        Summary.conversation_id == conversation_id
    ).first()
    assert resumen is not None
    assert resumen.data["risk_level"] == "NORMAL"


def test_rechazo_con_pocos_slots_escala_al_cerrar_por_incertidumbre(session):
    """Antes de Fix 5a, `build_summary` solo tomaba el máximo de los
    `risk_level` ya persistidos turno a turno —ninguno usó `final=True`—, así
    que una llamada que cierra con casi nada respondido quedaba en NORMAL. Con
    Fix 5a, el resumen reevalúa con `final=True` y aplica
    `no_se_pudo_evaluar` (completeness < 0.34, ver `thresholds.yaml`).
    """
    from app.models import Alert, Summary

    primero = process_turn(session, text="El dolor es un 2, tranquilo.")
    segundo = process_turn(
        session,
        text="No quiero seguir hablando de esto, cuelgue ya.",
        conversation_id=primero.conversation_id,
    )

    assert segundo.call_ended is True

    resumen = session.query(Summary).filter(
        Summary.conversation_id == primero.conversation_id
    ).first()
    assert resumen is not None
    assert resumen.data["risk_level"] == "CRÍTICO"
    assert "no_se_pudo_evaluar" in resumen.data["triggered_rules"]

    alerta = session.query(Alert).filter(
        Alert.conversation_id == primero.conversation_id
    ).first()
    assert alerta is not None
    assert alerta.risk_level == "CRÍTICO"


# --- el agente no dictamina sobre normalidad ---------------------------------

@pytest.mark.parametrize(
    "respuesta",
    [
        # Medido en una llamada real: ante "estoy viendo borroso, ¿es normal?" el
        # 3B abrió con un veredicto clínico sin evidencia, y con el riesgo aún en
        # NORMAL ninguna guarda lo tocaba.
        "No, no es normal. Debe ser compresible.",
        "Sí, es normal después de una cirugía.",
        "Eso no es normal, debería consultar.",
    ],
)
def test_el_veredicto_de_normalidad_se_recorta(respuesta):
    from app.agent.orchestrator import _VEREDICTO_NORMALIDAD
    assert _VEREDICTO_NORMALIDAD.search(respuesta), respuesta


def test_una_respuesta_anclada_normal_no_se_recorta():
    from app.agent.orchestrator import _VEREDICTO_NORMALIDAD
    for buena in ("El baño diario se puede desde las 48 horas.",
                  "Camine despacio y aumente la distancia cada día."):
        assert not _VEREDICTO_NORMALIDAD.search(buena), buena


def test_la_respuesta_generada_no_hace_preguntas():
    """El guion añade la siguiente pregunta; dos seguidas en voz se pisan."""
    from app.agent.orchestrator import _sin_preguntas
    assert _sin_preguntas(
        "No, no es normal. Debe ser compresible. ¿Se duele el abdomen?"
    ) == "No, no es normal. Debe ser compresible."
    assert _sin_preguntas("¿Y cómo sigue?") == ""


@pytest.mark.parametrize(
    "cruda",
    [
        # El turno medido: emoción inventada + pregunta extra delante de la
        # abstención. Salía verbatim porque el `return` de la abstención iba
        # antes de todas las guardas.
        "Usted suena un poco asustado, ¿cómo está sintiéndose después de la "
        "operación? Sobre eso no tengo información en los documentos del "
        "hospital. Se lo paso a enfermería.",
        # La media abstención: una afirmación clínica que se quedaría sin cita,
        # porque al abstenerse el orquestador quita las fuentes.
        "Se menciona en el documento que si nota algún cambio en la incisión "
        "debe llamar a la oficina. Sin embargo, no hay información específica "
        "sobre los síntomas de la cesárea.",
    ],
)
def test_la_abstencion_del_modelo_se_normaliza(cruda):
    """Al declarar el límite no queda nada del texto del modelo que conservar."""
    from app.agent.orchestrator import _validar_grounding
    from app.llm.ollama_adapter import ABSTENCION
    from app.schemas import RagResult

    evidencia = RagResult(answer="La herida se lava con agua y jabón.",
                          sources=[], confidence=0.9, has_evidence=True)
    salida = _validar_grounding(cruda, evidencia, riesgo="NORMAL")
    assert salida == ABSTENCION
    assert "?" not in salida
    assert "asustado" not in salida


def test_la_emocion_atribuida_sin_que_tambien_se_recorta():
    """`sin_muletillas` solo cazaba "parece QUE..."; el 70B la dice sin el "que"."""
    from app.llm.ollama_adapter import sin_muletillas
    salida = sin_muletillas(
        "Usted suena un poco asustado. Puede ducharse desde las 48 horas."
    )
    assert salida == "Puede ducharse desde las 48 horas."
    assert sin_muletillas("Se le nota preocupada, pero la herida se seca sola.") \
        == ""


# --- el bucle de cierre (chat real del 9 de agosto) -----------------------------


def _tamizaje_completo(session) -> str:
    """Recorre los seis slots con respuestas normales; devuelve el conversation_id
    con la llamada parada en la pregunta abierta."""
    respuestas = [
        "Un 2, apenas se nota, tranquilo.",   # dolor
        "No tengo fiebre, nada.",              # fiebre
        "Camino bien, sin problema.",          # movilidad
        "La herida se ve bien.",               # herida
        "Como bien, normal.",                  # apetito
        "Duermo bien, tranquilo.",              # sueño
    ]
    conversation_id = None
    for texto in respuestas:
        r = process_turn(session, text=texto, conversation_id=conversation_id)
        conversation_id = r.conversation_id
    return conversation_id


def test_la_negacion_simple_cierra_la_confirmacion(session):
    """El bug del chat: "no" → "¿Hay algo más...?" → "no, nada" → "¿Quedamos
    así...?" → ... indefinidamente. Una negación simple en fase de cierre
    significa que no hay más temas, y la llamada cierra."""
    cid = _tamizaje_completo(session)
    # "no" pelado a la pregunta abierta no es despedida formal: abre CONFIRMACION.
    r = process_turn(session, text="no", conversation_id=cid)
    assert r.call_ended is False
    # La negación simple a la pregunta de confirmación SÍ cierra.
    ultimo = process_turn(session, text="No, nada, así está bien.",
                          conversation_id=cid)
    assert ultimo.call_ended is True


def test_la_confirmacion_cierra_por_tope_aunque_no_se_entienda_al_paciente(session):
    """Aunque ninguna respuesta case con despedida ni negación, la fase de
    confirmación no puede ser infinita: cierra al agotar el tope."""
    cid = _tamizaje_completo(session)
    respuestas_raras = ["pues yo creo que ya sería eso",
                        "mmm pues era lo del control médico",
                        "era eso del control",
                        "lo del control pues"]
    ultimo = None
    for texto in respuestas_raras:
        ultimo = process_turn(session, text=texto, conversation_id=cid)
        if ultimo.call_ended:
            break
    assert ultimo is not None and ultimo.call_ended is True, (
        "la fase de confirmación nunca cerró"
    )


# --- "¿quiere preguntarme algo?": la respuesta ES la pregunta -------------------
# Llamada real del 10/08, 7:12 p. m.: tras la invitación abierta, "Podría tomar
# acetaminofén" y "Le preguntaba si de pronto es posible..." recibieron cada una
# otra pregunta de cierre. Solo la tercera formulación —"Puedo tomar acetaminofén",
# que empieza por un arranque interrogativo de la lista— se contestó.


def test_la_invitacion_a_preguntar_lee_el_turno_como_pregunta(session):
    """Con la invitación en el aire, el formato sobra.

    Se usa a propósito un turno SIN ninguna pista léxica —"Lo del acetaminofén"—
    para que el test aísle la regla de contexto: si pasara por la familia F de
    `_PREGUNTA_SIN_SIGNOS` no estaría probando nada nuevo.
    """
    from app.models import Turn

    cid = _tamizaje_completo(session)
    r = process_turn(session, text="Lo del acetaminofén.", conversation_id=cid)
    assert r.call_ended is False

    turno = session.query(Turn).filter(Turn.conversation_id == cid).order_by(
        Turn.created_at).all()[-1]
    assert turno.intent == "pregunta_clinica"


def test_el_cierre_de_una_llamada_normal_no_pasa_de_dos_preguntas(session):
    """El tope manda aunque los tres turnos sean preguntas de verdad.

    Los tres son la MISMA duda reformulada, que es como se comporta alguien al
    que no le contestan. Al mejorar el clasificador las tres pasaron a
    `pregunta_clinica`, y con la exención del tope aplicándose a todas el cierre
    se fue a seis turnos. Se contestan igual —la del tercero sale pegada a la
    despedida— pero la llamada acaba ahí.
    """
    cid = _tamizaje_completo(session)
    textos = ["Podría tomar acetaminopeno.",
              "Le preguntaba si de pronto es posible que pueda tomar acetaminogen "
              "para el dolor.",
              "Puedo tomar acetaminofén para el dolor."]
    resultados = [process_turn(session, text=t, conversation_id=cid) for t in textos]

    assert [r.call_ended for r in resultados] == [False, False, True], (
        "el cierre de una llamada normal son dos preguntas, no seis"
    )


def test_la_invitacion_no_convierte_un_si_pelado_en_pregunta(session):
    """"Sí" dice que tiene algo que preguntar, pero no dice qué: mandarlo al RAG
    sería consultar el corpus con la palabra "sí"."""
    from app.models import Turn

    cid = _tamizaje_completo(session)
    process_turn(session, text="Sí.", conversation_id=cid)

    turno = session.query(Turn).filter(Turn.conversation_id == cid).order_by(
        Turn.created_at).all()[-1]
    assert turno.intent == "respuesta"


# --- el guion de seguridad no se repite (chat real del 9 de agosto) -------------


def test_el_guion_critico_no_se_repite_tras_el_cierre_anunciado(session):
    """Chat real: tras "vamos a colgar aquí", un "Listo, gracias" recibió el
    guion de seguridad completo otra vez, palabra por palabra."""
    primero = process_turn(session, text="Me duele un berraco, no aguanto")
    guion = primero.response
    cid = primero.conversation_id

    r = None
    for texto in ("Listo, muchas gracias.", "No, todo muy claro. Muchas gracias.",
                  "Listo, gracias."):
        r = process_turn(session, text=texto, conversation_id=cid)
        assert r.response != guion, f"el guion de seguridad revivió tras {texto!r}"
        if r.call_ended:
            break
    assert r is not None and r.call_ended is True


# --- automedicación tras escalar: se verifica, no se ignora --------------------


def test_una_mencion_de_medicamento_tras_escalar_se_trata_como_pregunta(session):
    """Chat real: tras el guion de dolor_severo, "me voy a tomar un
    metronidazol" recibía un acuse vacío ("Anotado, tomará el metronidazol")
    pegado a la pregunta de cierre genérica, porque la frase declarativa no
    matcheaba el clasificador de preguntas y el RAG nunca se consultaba.

    Sin corpus cargado en este test, el RAG no tiene evidencia — lo que importa
    es que el turno SÍ se procesa como `pregunta_clinica` (se abstiene con
    seguridad y ofrece pasarlo a enfermería) en vez de como un acuse ciego, y
    que el cierre de confirmación usa el banco escalado.
    """
    from app.models import Turn

    primero = process_turn(session, text="Me duele un berraco, no aguanto")
    assert primero.risk_level == "CRÍTICO"
    cid = primero.conversation_id

    segundo = process_turn(session, text="Me voy a tomar un metronidazol.",
                           conversation_id=cid)
    assert segundo.call_ended is False

    turno = session.query(Turn).filter(Turn.conversation_id == cid).order_by(
        Turn.created_at).all()[-1]
    assert turno.intent == "pregunta_clinica"
    # No es el acuse vacío del bug: no se limita a repetir el nombre del
    # medicamento sin decir nada más.
    assert "anotado" not in segundo.response.lower()


def test_el_cierre_de_confirmacion_tras_escalar_usa_el_banco_escalado(session):
    """"Ok"/"listo"/"entendido" sueltos ya cierran la llamada (niega_mas_temas):
    se necesita una respuesta que la mantenga abierta, como en
    `test_turnos_posteriores_no_reabren_el_guion`."""
    from app.agent import phrasing

    primero = process_turn(session, text="Me duele un berraco, no aguanto")
    segundo = process_turn(session, text="Espere",
                           conversation_id=primero.conversation_id)

    assert segundo.call_ended is False
    assert segundo.response in phrasing.CONFIRMAR_CIERRE_ESCALADO


def test_varias_preguntas_reales_seguidas_tras_escalar_no_cuelgan_a_la_fuerza(session):
    """Reproduce el chat real completo: tras el guion de dolor_severo, el
    paciente pregunta por un medicamento de tres formas distintas (garbled STT
    incluido) en tres turnos seguidos. Antes del fix, el segundo y tercer turno
    no se reclasificaban como pregunta clínica (la regex no cubría "me puedo
    tomar" ni "quisiera saber si..."), y aunque se reclasificaran,
    MAX_TURNOS_CONFIRMACION=2 colgaba la llamada en el tercer turno sin
    responder y sin que el paciente se hubiera despedido.
    """
    from app.models import Turn

    primero = process_turn(session, text="Me duele un berraco, no aguanto")
    assert primero.risk_level == "CRÍTICO"
    cid = primero.conversation_id

    turnos_paciente = [
        "Listo, me puedo tomar un metro ni a sol.",
        "Si quisiera saber si me puedo tomar una acetaminofén.",
        "Sí, me puedo tomar una acetaminofén o algo.",
    ]
    respuestas = []
    for texto in turnos_paciente:
        r = process_turn(session, text=texto, conversation_id=cid)
        assert r.call_ended is False, f"no debía colgar en {texto!r}"
        respuestas.append(r)

    turnos = session.query(Turn).filter(Turn.conversation_id == cid).order_by(
        Turn.created_at).all()
    # Los tres turnos del paciente (después del guion) se procesaron como
    # pregunta clínica, no como relleno.
    for t in turnos[-3:]:
        assert t.intent == "pregunta_clinica", t.patient_utterance

    # Y una despedida explícita después SÍ cierra: el fix no vuelve infinita
    # la fase de confirmación.
    ultimo = process_turn(session, text="Listo, muchas gracias, hasta luego.",
                          conversation_id=cid)
    assert ultimo.call_ended is True


def test_el_guion_de_incertidumbre_no_se_repite(session):
    """El agujero exacto del guard viejo: el guion que nace de la política de
    incertidumbre (`no_se_pudo_evaluar`, sintetizado con `final=True`) no vive
    en `acumulado`, así que `critico` vuelve a False al turno siguiente y la
    fase ya avanzó a CONFIRMACION — el guard `phase is not ESCALAMIENTO` no
    aplicaba y cada intento de cierre re-disparaba el guion completo."""
    primero = process_turn(session, text="El dolor es un 2, tranquilo.")
    cid = primero.conversation_id

    # Se despide con el cuadro casi vacío → política de incertidumbre → guion.
    con_guion = process_turn(session, text="Tengo que colgar, hasta luego",
                             conversation_id=cid)
    assert "enfermería" in con_guion.response.lower() or "123" in con_guion.response
    guion = con_guion.response
    assert con_guion.call_ended is False

    # Sigue hablando sin despedirse: pregunta de confirmación, no el guion.
    medio = process_turn(session, text="bueno", conversation_id=cid)
    assert medio.response != guion

    # Se despide otra vez: cierra de verdad, sin repetir el guion.
    ultimo = process_turn(session, text="Listo, gracias, chao",
                          conversation_id=cid)
    assert ultimo.response != guion, "el guion de incertidumbre se repitió"
    assert ultimo.call_ended is True

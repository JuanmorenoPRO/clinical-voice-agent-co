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


def test_el_primer_turno_critico_entrega_el_guion_completo(session):
    r = process_turn(session, text="Me duele un berraco, no aguanto")
    assert r.risk_level == "CRÍTICO"
    assert r.critical_override is True
    assert r.call_ended is False
    assert "berraco" not in r.response  # es el guion determinista, no un eco


def test_el_segundo_turno_critico_cierra_en_vez_de_repetir(session):
    primero = process_turn(session, text="Me duele un berraco, no aguanto")
    segundo = process_turn(
        session, text="Sigo con el mismo dolor", conversation_id=primero.conversation_id
    )

    assert segundo.call_ended is True
    # No es el mismo texto: si lo fuera, seguiríamos teniendo el bug reportado.
    assert segundo.response != primero.response
    assert len(segundo.response) < len(primero.response), (
        "el cierre debe ser corto, no el guion clínico completo otra vez"
    )


def test_la_conversacion_queda_cerrada_en_la_base(session):
    primero = process_turn(session, text="Me duele un berraco, no aguanto")
    process_turn(session, text="Ok", conversation_id=primero.conversation_id)

    conv = session.get(Conversation, primero.conversation_id)
    assert conv.status == "closed"
    assert conv.ended_at is not None


def test_el_resumen_queda_listo_sin_llamar_a_close_aparte(session):
    """RF-10: el resumen se genera al cerrar, no depende de un paso extra."""
    from app.models import Summary

    primero = process_turn(session, text="Me duele un berraco, no aguanto")
    process_turn(session, text="Listo", conversation_id=primero.conversation_id)

    resumen = session.query(Summary).filter(
        Summary.conversation_id == primero.conversation_id
    ).first()
    assert resumen is not None
    assert resumen.data["risk_level"] == "CRÍTICO"


def test_turnos_posteriores_siguen_cerrando_sin_reabrir_el_guion(session):
    """Si el paciente insiste tras el cierre, no revive el guion largo."""
    primero = process_turn(session, text="Me duele un berraco, no aguanto")
    segundo = process_turn(session, text="Espere", conversation_id=primero.conversation_id)
    tercero = process_turn(session, text="Otra cosa", conversation_id=primero.conversation_id)

    assert tercero.call_ended is True
    assert tercero.response == segundo.response  # cierre estable, no el guion largo


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

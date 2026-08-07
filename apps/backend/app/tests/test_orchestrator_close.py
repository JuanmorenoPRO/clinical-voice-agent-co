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

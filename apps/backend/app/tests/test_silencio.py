"""El paciente deja de contestar: escalera de presencia y alerta ALTA.

Cubre el escenario completo de punta a punta —sondear, avisar, colgar— sin audio
y sin modelo: el silencio cortocircuita el LLM en `agent/orchestrator.py`, así que
estos tests corren igual con Ollama apagado.

La escalera vive en `agent/script.py` (`MAX_SILENCIOS`) precisamente para poder
probarse así; el pipeline de voz solo aporta el reloj que decide CUÁNDO inyectar
el marcador `[silencio]`.
"""
from __future__ import annotations

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.agent import script
from app.agent.orchestrator import process_turn
from app.agent.script import CallState, Phase
from app.db import Base
from app.models import Alert, Conversation, Summary, Turn  # noqa: F401 — registra tablas
from app.schemas import Symptoms

SILENCIO = "[silencio]"


@pytest.fixture()
def session(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path}/test.db", future=True)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine, expire_on_commit=False, future=True)
    with Session() as s:
        yield s


# --- la máquina de estados, sin base de datos --------------------------------

def test_el_primer_silencio_sondea_y_conserva_el_slot():
    estado = CallState(slot_actual="fiebre", phase=Phase.TAMIZAJE)
    action = script.next_action(estado, Symptoms(), silencio=True)
    assert action.kind == "sondear"
    assert action.intento == 1
    # El slot no se abandona: quien no contestó puede no haber oído la pregunta.
    assert action.slot == "fiebre"


def test_el_segundo_silencio_avisa_y_el_tercero_cierra():
    estado = CallState(slot_actual="dolor", sin_respuesta=1)
    assert script.next_action(estado, Symptoms(), silencio=True).intento == 2

    estado = CallState(slot_actual="dolor", sin_respuesta=2)
    action = script.next_action(estado, Symptoms(), silencio=True)
    assert action.kind == "cerrar"
    assert action.phase is Phase.TERMINADA


def test_el_silencio_no_gasta_repreguntas_del_slot():
    """Reformular es para quien esquivó la pregunta, no para quien no la oyó.

    Sin esto, dos silencios sueltos consumían los dos intentos de `MAX_REPREGUNTAS`
    y el slot se daba por perdido sin haberlo preguntado nunca de verdad.
    """
    estado = CallState(slot_actual="dolor")
    action = script.next_action(estado, Symptoms(), silencio=True)
    nuevo = script.apply(estado, action, Symptoms(), progreso=False, silencio=True)
    assert nuevo.repreguntas == {}
    assert nuevo.sin_responder == []
    assert nuevo.slot_actual == "dolor"
    assert nuevo.sin_respuesta == 1


def test_una_respuesta_reinicia_el_contador_de_silencios():
    """Cuenta silencios SEGUIDOS: los sueltos abundan en la capa 2 del dataset
    y no pueden ir acercando la llamada a colgarse."""
    estado = CallState(slot_actual="dolor", sin_respuesta=2)
    action = script.next_action(estado, Symptoms(pain_level=3), silencio=False)
    nuevo = script.apply(estado, action, Symptoms(pain_level=3), silencio=False)
    assert nuevo.sin_respuesta == 0


def test_el_estado_sobrevive_al_viaje_por_la_base():
    estado = CallState(sin_respuesta=2)
    assert CallState.from_dict(estado.to_dict()).sin_respuesta == 2
    # Una traza vieja, escrita antes de que existiera el campo, no revienta.
    assert CallState.from_dict({"phase": "tamizaje"}).sin_respuesta == 0


# --- de punta a punta, por el orquestador ------------------------------------

def test_tres_silencios_cierran_la_llamada_y_alertan_en_alto(session):
    r1 = process_turn(session, text=SILENCIO)
    assert r1.call_ended is False
    assert "?" in r1.response          # comprueba si sigue en línea

    r2 = process_turn(session, text=SILENCIO, conversation_id=r1.conversation_id)
    assert r2.call_ended is False
    assert "termin" in r2.response.lower()   # avisa ANTES de colgar

    r3 = process_turn(session, text=SILENCIO, conversation_id=r1.conversation_id)
    assert r3.call_ended is True
    assert "123" in r3.response         # no se despide como si estuviera bien

    alertas = session.query(Alert).filter(
        Alert.conversation_id == r1.conversation_id).all()
    sin_respuesta = [a for a in alertas if "paciente_no_responde" in a.triggered_rules]
    assert len(sin_respuesta) == 1
    assert sin_respuesta[0].risk_level == "ALTO"
    assert "deja de contestar" in sin_respuesta[0].transcript

    # El cierre genera el resumen, y la regla queda en la traza: una llamada que
    # se quedó sin nadie al otro lado no puede figurar como terminada con normalidad.
    assert r3.triggered_rules and "paciente_no_responde" in r3.triggered_rules
    assert r3.risk_level in ("ALTO", "CRÍTICO")
    conv = session.get(Conversation, r1.conversation_id)
    assert conv.status == "closed"


def test_el_silencio_no_cuesta_ni_una_llamada_al_modelo(session):
    """No hay nada que extraer de un silencio, y en voz esos ~2,5 s se notan."""
    r = process_turn(session, text=SILENCIO)
    turno = session.query(Turn).filter(Turn.conversation_id == r.conversation_id).one()
    assert turno.llm_calls == 0
    assert turno.tokens_in == 0 and turno.tokens_out == 0
    assert turno.intent == "silencio"


def test_un_silencio_suelto_no_acerca_la_llamada_a_colgarse(session):
    """El paciente calla una vez, luego contesta: la llamada sigue su curso."""
    r1 = process_turn(session, text=SILENCIO)
    r2 = process_turn(session, text="Me duele como un 3",
                      conversation_id=r1.conversation_id)
    assert r2.call_ended is False

    r3 = process_turn(session, text=SILENCIO, conversation_id=r1.conversation_id)
    r4 = process_turn(session, text=SILENCIO, conversation_id=r1.conversation_id)
    # Serían el tercer y cuarto silencio de la llamada, pero solo el primero y
    # segundo SEGUIDOS: la escalera arrancó de cero.
    assert r3.call_ended is False
    assert r4.call_ended is False


def test_max_silencios_es_configurable():
    """`SILENCE_MAX_ATTEMPTS` manda; el default conserva la escalera de tres."""
    estado = CallState(slot_actual="dolor", sin_respuesta=1)
    # Con el máximo en 2, el segundo silencio ya cierra.
    action = script.next_action(estado, Symptoms(), silencio=True, max_silencios=2)
    assert action.kind == "cerrar"
    # Con el default (3), ese mismo estado todavía avisa.
    assert script.next_action(estado, Symptoms(), silencio=True).kind == "sondear"


# --- UNKNOWN explícito: el silencio jamás se convierte en una respuesta -------

def test_el_silencio_no_escribe_ningun_sintoma(session):
    """El requisito central: fiebre preguntada + silencio ⇒ fever=None (UNKNOWN),
    JAMÁS fever=False. El diseño ternario ya lo garantizaba; esto lo fija."""
    r1 = process_turn(session, text="Me duele como un 3")   # dolor resuelto → fiebre
    r2 = process_turn(session, text=SILENCIO, conversation_id=r1.conversation_id)
    assert r2.symptoms.fever is None
    assert r2.symptoms.pain_level == 3                       # lo contestado se conserva
    turno = session.query(Turn).filter(
        Turn.conversation_id == r1.conversation_id).order_by(Turn.created_at).all()[-1]
    assert turno.extracted_symptoms.get("fever") is None
    # Y el guion no avanzó: el slot sigue siendo el mismo.
    assert CallState.from_dict(turno.agent_state).slot_actual == r1.slot_actual


def test_el_cierre_por_silencio_marca_los_pendientes_como_unknown(session):
    """La llamada se corta con el guion a medias: lo que quedó sin resolver
    queda como UNKNOWN (`unanswered`) en la traza y en el resumen, no invisible."""
    r1 = process_turn(session, text="Me duele como un 3")
    for _ in range(3):
        r = process_turn(session, text=SILENCIO, conversation_id=r1.conversation_id)
    assert r.call_ended is True
    # El slot que se estaba preguntando (fiebre) quedó UNKNOWN, no False.
    assert "fiebre" in r.symptoms.unanswered
    assert r.symptoms.fever is None
    assert "dolor" not in r.symptoms.unanswered   # lo contestado no es UNKNOWN
    resumen = session.query(Summary).filter(
        Summary.conversation_id == r1.conversation_id).one()
    assert "fiebre" in resumen.data["unanswered_slots"]


def test_una_respuesta_tardia_saca_el_slot_de_unknown():
    """UNKNOWN no es una condena: si el paciente contesta después, el dato real
    gana y el slot deja de figurar como sin respuesta."""
    from app.nlu.merge import merge_symptoms

    con_hueco = merge_symptoms(Symptoms(), Symptoms(unanswered=["fiebre", "dolor"]))
    assert set(con_hueco.unanswered) == {"fiebre", "dolor"}
    despues = merge_symptoms(con_hueco, Symptoms(fever=True))
    assert despues.unanswered == ["dolor"]
    assert despues.fever is True


# --- CONNECTION_LOST ≠ NO_RESPONSE ---------------------------------------------

def test_la_desconexion_cierra_con_alerta_y_resumen(session):
    """El stream se cae a mitad de llamada: sin esto la conversación quedaba
    `active` para siempre — sin resumen, sin alerta, invisible en el reporte."""
    from app.agent.orchestrator import abandon_conversation

    r1 = process_turn(session, text="Me duele como un 3")
    abandon_conversation(session, r1.conversation_id)

    conv = session.get(Conversation, r1.conversation_id)
    assert conv.status == "closed"
    alertas = session.query(Alert).filter(
        Alert.conversation_id == r1.conversation_id,
    ).all()
    caida = [a for a in alertas if "conexion_perdida" in a.triggered_rules]
    assert len(caida) == 1
    assert caida[0].risk_level == "ALTO"
    assert "Reintentar el contacto" in caida[0].transcript
    # La remediación es re-marcar, NO valorar a un paciente que enmudeció:
    # las dos reglas no pueden mezclarse.
    assert not any("paciente_no_responde" in a.triggered_rules for a in alertas)
    # Y el resumen existe, con la política de incertidumbre aplicada: una
    # llamada cortada a mitad de tamizaje no puede salir como verde.
    resumen = session.query(Summary).filter(
        Summary.conversation_id == r1.conversation_id).one()
    assert resumen.data["risk_level"] != "NORMAL"

    # Idempotente: el pipeline puede dispararla dos veces al desmontarse.
    abandon_conversation(session, r1.conversation_id)
    assert session.query(Alert).filter(
        Alert.conversation_id == r1.conversation_id).count() == len(alertas)


def test_la_desconexion_tras_un_cierre_normal_no_alerta(session):
    """El transporte también se desconecta cuando el AGENTE cuelga: eso no es
    una conexión perdida y no puede fabricar alertas."""
    from app.agent.orchestrator import abandon_conversation

    r1 = process_turn(session, text=SILENCIO)
    r2 = process_turn(session, text=SILENCIO, conversation_id=r1.conversation_id)
    r3 = process_turn(session, text=SILENCIO, conversation_id=r1.conversation_id)
    assert r3.call_ended is True                  # cierre normal (por silencio)
    antes = session.query(Alert).filter(
        Alert.conversation_id == r1.conversation_id).count()

    abandon_conversation(session, r1.conversation_id)
    despues = session.query(Alert).filter(
        Alert.conversation_id == r1.conversation_id).count()
    assert despues == antes

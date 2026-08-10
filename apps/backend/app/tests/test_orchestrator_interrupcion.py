"""Un "no" tras un barge-in no se lee contra la pregunta cortada.

El `Turn` se persiste con `CallState.ultima_pregunta` avanzada ANTES de que el
TTS suene. Si el paciente interrumpe la pregunta a media frase, el turno
siguiente interpretaba su "sí/no" pelado contra una pregunta que nunca oyó
entera (`nlu/polaridad.py` resuelve el slot en función de esa pregunta): deriva
silenciosa del contexto. La capa de voz ahora marca ese turno con
`pregunta_interrumpida=True` y el orquestador deja el slot abierto — el guion
repregunta por su vía normal, que es lo honesto.

El LLM se sustituye por la ruta determinista (`_SoloLexico`, mismo doble que
`test_orchestrator_procedimiento.py`): se prueba la lógica del orquestador, no
el modelo, y sin ello un "No." sin pregunta contra la que resolverse haría una
llamada real a Groq desde la suite.
"""
from __future__ import annotations

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.agent.orchestrator import process_turn
from app.db import Base
from app.llm.adapter import Extraction
from app.models import Conversation, Turn  # noqa: F401  — registra las tablas
from app.nlu import intent, lexicon


class _SoloLexico:
    async def extract(self, *, slot, question, utterance):
        return Extraction(symptoms=lexicon.extract(utterance, slot, question=question),
                          intent=intent.classify(utterance))


@pytest.fixture(autouse=True)
def solo_lexico(monkeypatch):
    monkeypatch.setattr("app.agent.orchestrator.get_llm", lambda: _SoloLexico())


@pytest.fixture()
def session(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path}/test.db", future=True)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine, expire_on_commit=False, future=True)
    with Session() as s:
        yield s


def test_un_no_normal_si_resuelve_el_slot(session):
    """Control: sin interrupción, el "no" pelado se lee contra la pregunta."""
    r1 = process_turn(session, text="Un 4.")
    assert r1.slot_actual == "fiebre"

    r2 = process_turn(session, text="No.", conversation_id=r1.conversation_id)
    assert r2.symptoms.fever is False


def test_un_no_tras_barge_in_no_resuelve_el_slot(session):
    r1 = process_turn(session, text="Un 4.")
    assert r1.slot_actual == "fiebre"

    r2 = process_turn(session, text="No.", conversation_id=r1.conversation_id,
                      pregunta_interrumpida=True)
    # El "no" no se anota contra una pregunta que el paciente no oyó entera...
    assert r2.symptoms.fever is None
    # ...el slot sigue abierto y el guion vuelve a preguntar por su vía normal.
    assert r2.slot_actual == "fiebre"
    assert r2.response.rstrip().endswith("?")

    # Y la llamada se recupera sola: el siguiente "no" —ya con la pregunta
    # repetida y oída— sí resuelve.
    r3 = process_turn(session, text="No.", conversation_id=r1.conversation_id)
    assert r3.symptoms.fever is False

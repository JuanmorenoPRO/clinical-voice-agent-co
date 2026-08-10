""""¿Qué es calentura?" es una pregunta, no un reporte de fiebre.

Bug real: el agente preguntó por la fiebre y el paciente preguntó "que es
calentura" (Whisper no pone los signos `¿?`). El turno caía en el default
`respuesta`, el léxico extraía "calentura" como `fever=True`, y el agente
seguía el guion como si nada: no respondió la duda Y anotó un síntoma falso.

El orquestador ahora reclasifica con `intent.pide_aclaracion` (detector fuera
de `classify`, mismo patrón que la automedicación) y `lexicon.extract` deja de
leer el término preguntado y el "sí/no" pelado del caso mixto.

También cubre el modo texto de las muletillas: "Ehh..." recibe "tómese su
tiempo" (no "¿me lo repite?") y no gasta repregunta.

El LLM se sustituye por la ruta determinista (`_SoloLexico`): se prueba la
lógica del orquestador, no el modelo. Sin corpus RAG en el test, la aclaración
se contesta con la abstención segura, que no llama al modelo.
"""
from __future__ import annotations

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.agent import phrasing
from app.agent.orchestrator import process_turn
from app.db import Base
from app.llm.adapter import Extraction
from app.models import Conversation, Turn  # noqa: F401  — registra las tablas
from app.nlu import intent, lexicon


class _SoloLexico:
    async def extract(self, *, slot, question, utterance):
        return Extraction(symptoms=lexicon.extract(utterance, slot, question=question),
                          intent=intent.classify(utterance))

    async def pregunta_es_del_dominio(self, *, question, evidence):
        return False

    async def reply_grounded(self, *, question, evidence, patient_context):
        from app.llm.adapter import LLMUsage
        from app.llm.ollama_adapter import ABSTENCION

        return ABSTENCION, LLMUsage(model="test", purpose="reply")


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


def _ultimo_turno(session, conversation_id):
    return (session.query(Turn).filter(Turn.conversation_id == conversation_id)
            .order_by(Turn.created_at).all()[-1])


def test_la_aclaracion_se_oye_como_pregunta_y_no_anota_fiebre(session):
    r1 = process_turn(session, text="Un 4.")
    assert r1.slot_actual == "fiebre"

    r2 = process_turn(session, text="que es calentura",
                      conversation_id=r1.conversation_id)
    turno = _ultimo_turno(session, r1.conversation_id)
    assert turno.intent == "pregunta_clinica"
    # El término preguntado no se convierte en hallazgo.
    assert r2.symptoms.fever is None
    # Y el guion no avanza en falso: sigue en fiebre, re-preguntando.
    assert r2.slot_actual == "fiebre"
    assert r2.response.rstrip().endswith("?")


def test_la_aclaracion_no_gasta_repregunta(session):
    """Dos aclaraciones seguidas no pueden quemar las dos repreguntas del slot
    y terminar en "no me supo decir": preguntar qué significa la pregunta no es
    esquivarla. Después de aclarar (dos veces), el "sí" pelado aún resuelve."""
    r1 = process_turn(session, text="Un 4.")
    process_turn(session, text="que es calentura", conversation_id=r1.conversation_id)
    process_turn(session, text="no le entiendo la pregunta",
                 conversation_id=r1.conversation_id)

    r4 = process_turn(session, text="Sí, sí he tenido calentura.",
                      conversation_id=r1.conversation_id)
    assert r4.symptoms.fever is True, (
        "el slot cayó en sin_responder: las aclaraciones gastaron las repreguntas"
    )


def test_la_muletilla_en_texto_no_pide_repetir_ni_gasta(session):
    r1 = process_turn(session, text="Un 4.")
    r2 = process_turn(session, text="Ehh...", conversation_id=r1.conversation_id)
    assert r2.response == phrasing.PENSANDO

    # La basura real del STT sigue con el trato de siempre.
    r3 = process_turn(session, text="asdkjhaskjdh", conversation_id=r1.conversation_id)
    assert r3.response == phrasing.NO_ENTENDI

    # Y el slot sigue vivo: el "no" pelado aún se lee contra la pregunta.
    r4 = process_turn(session, text="No.", conversation_id=r1.conversation_id)
    assert r4.symptoms.fever is False

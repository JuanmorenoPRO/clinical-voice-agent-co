"""Preguntas clínicas sin signo de interrogación, pegadas a la respuesta del slot.

Bug real probando con voz: el paciente contestó el dolor y, en la misma frase
sin "¿?" (Whisper no los pone), preguntó si podía volver a hacer ejercicio. El
turno se clasificaba entero como `respuesta` —"puedo" no era la primera
palabra del turno completo, que es lo único que la rama sin "¿?" reconocía—
así que el agente reflejaba el dolor y pasaba derecho a la siguiente pregunta
del guion sin haber oído la pregunta.

Corre sin Ollama y sin red: "Un 4" lo resuelve el léxico, y sin corpus RAG
cargado en este test la pregunta se contesta con la abstención segura
(`ABSTENCION`), que no llama al modelo (ver `reply_grounded`).

Segundo bug, medido después de corregir el primero: pasarle el cuadro
acumulado a `_texto_de` para no perder el reflejo del dato ("Un 4, ahí en la
mitad") hacía que la rama de ABSTENCIÓN encadenara tres frases de transición
seguidas — la abstención, `transicion_abstencion`, y el reflejo — porque el
puente ya lo daba `transicion_abstencion` y no hacía falta un segundo acuse
detrás. El reflejo debe sobrevivir cuando SÍ hay una respuesta real anclada
en evidencia, no cuando el sistema se abstiene.
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


def test_la_pregunta_sin_signo_pegada_a_la_respuesta_se_escucha(session):
    r = process_turn(
        session, text="Un 4. Con ese dolor puedo volver a hacer ejercicio."
    )

    assert r.symptoms.pain_level == 4, "el dato del slot se sigue capturando"
    assert r.call_ended is False
    # No es el bug reportado: no salta a la siguiente pregunta ignorando la
    # pregunta del paciente. Sin evidencia en el corpus de prueba, se abstiene
    # con honestidad en vez de inventar o callar.
    assert "no tengo informaci" in r.response.lower()

    turno = session.query(Turn).filter(Turn.conversation_id == r.conversation_id
                                       ).order_by(Turn.created_at).all()[-1]
    assert turno.intent == "pregunta_clinica"


def test_la_pregunta_sin_signo_en_mitad_del_turno_se_escucha(session):
    """El turno reportado, tal cual llegó por voz.

    Aquí la pregunta ni abre el turno ni abre la última frase: va pegada a un
    sustantivo ("solo la diarrea que puedo tomar..."), que es donde ninguna de las
    dos ramas ancladas de `_PREGUNTA` podía verla. El agente respondió con la
    pregunta de cierre del guion, como si no hubiera oído nada.
    """
    r = process_turn(
        session, text="No, está bien. Solo la diarrea que puedo tomar para la diarrea."
    )

    turno = session.query(Turn).filter(Turn.conversation_id == r.conversation_id
                                       ).order_by(Turn.created_at).all()[-1]
    assert turno.intent == "pregunta_clinica"
    # Sin corpus RAG en este test, lo correcto es abstenerse — pero abstenerse es
    # haber oído la pregunta, que es justo lo que no pasaba.
    assert "no tengo informaci" in r.response.lower()


def test_la_abstencion_no_encadena_una_segunda_transicion(session):
    """Bug real: "Sobre eso no tengo información...  Voy a continuar entonces
    con las preguntas de su seguimiento. Un 4, ahí en la mitad. ¿Ha tenido
    fiebre...?" — dos puentes seguidos hacia la siguiente pregunta.
    `transicion_abstencion` ya hace ese trabajo; no debe haber un reflejo o
    acuse genérico detrás en la misma respuesta."""
    from app.agent import phrasing

    r = process_turn(
        session, text="Un 4. Con ese dolor puedo volver a hacer ejercicio."
    )
    # El reflejo específico del dato no debe aparecer pegado a la abstención.
    assert "un 4" not in r.response.lower(), r.response
    # Tampoco un acuse genérico ("Vale, anotado.", "Entendido.", etc.): la
    # respuesta debe ser abstención + transición + la pregunta pelada.
    for acuse in phrasing.ACUSES:
        assert acuse not in r.response, r.response


def test_el_guion_sigue_avanzando_al_siguiente_slot(session):
    r = process_turn(
        session, text="Un 4. Con ese dolor puedo volver a hacer ejercicio."
    )
    assert r.slot_actual == "fiebre"


# --- el reflejo SÍ se conserva cuando hay una respuesta real anclada --------
# `_texto_de` es una función pura (sin red ni DB): se prueba directo, sin
# tener que montar un corpus RAG completo para ejercitar el mismo camino que
# recorre `_texto_fallback` cuando `reply_grounded` sí devuelve una respuesta.


def test_texto_de_sin_acuse_no_antepone_reflejo():
    """Forma de la rama de abstención: bare question, sin reflejo ni acuse."""
    from app.agent.orchestrator import _texto_de
    from app.agent.script import Action, Phase
    from app.agent import phrasing

    accion = Action(kind="preguntar", slot="fiebre", phase=Phase.TAMIZAJE)
    texto = _texto_de(accion, semilla="s", recientes=[], nombre=None,
                      preocupante=False, con_acuse=False, con_prefijo=True)
    assert texto == phrasing.pregunta("fiebre", "s", [])


def test_texto_de_con_acumulado_antepone_el_reflejo():
    """Forma de la rama con evidencia real: el reflejo del dato precede a la
    siguiente pregunta."""
    from app.agent.orchestrator import _texto_de
    from app.agent.script import Action, Phase
    from app.schemas import Symptoms

    accion = Action(kind="preguntar", slot="fiebre", phase=Phase.TAMIZAJE)
    acumulado = Symptoms(pain_level=4)
    texto = _texto_de(accion, semilla="s", recientes=[], nombre=None,
                      preocupante=False, acumulado=acumulado, del_turno=acumulado,
                      slot_respondido="dolor")
    assert "Un 4, ahí en la mitad." in texto

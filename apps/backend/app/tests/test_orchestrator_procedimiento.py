"""El agente escucha lo que el paciente dice aunque no conteste la pregunta.

Reproduce la llamada reportada por el usuario. El agente preguntaba por la fiebre,
el paciente contestaba "me operaron de cesárea", y la respuesta era:

    "Sin termómetro: ¿lo ha sentido como fiebre, sí o no?"

Ni una palabra sobre lo que acababa de oír, y encima gastando uno de los dos
`MAX_REPREGUNTAS`. La causa no era que el LLM no analizara la respuesta —sí se
llama a `llm.extract()`— sino que su esquema de salida para ese slot es
`{"fiebre": "si"|"no"|"no_dice"}`: no había ningún canal por el que pudiera decir
"el paciente me habló de su cirugía". Ver `app/nlu/procedimiento.py`.

El LLM se sustituye por la ruta determinista (`solo_lexico`): lo que se prueba
aquí es la lógica del orquestador, no si un 3B acierta. Sin eso los tests son
flaky de la peor manera —dependen de que el modelo esté levantado Y de lo que
devuelva—, y medido: ante "mi hija está aquí conmigo en la casa" con el slot de
fiebre abierto, llama3.2:3b devolvió `si` y el agente contestó "Con calentura,
entonces.". El esquema lo obliga a elegir un valor y ahí se lo inventa, que es el
mismo fallo que ya documenta `ollama_adapter._to_symptoms`.
"""

from __future__ import annotations

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.agent import phrasing
from app.agent.orchestrator import _REGLA_PROCEDIMIENTO, process_turn
from app.agent.script import CallState
from app.db import Base
from app.llm.adapter import Extraction
from app.models import Alert, Conversation, Patient, Turn  # noqa: F401
from app.nlu import intent, lexicon


class _SoloLexico:
    """Doble del adaptador que se queda en la ruta determinista.

    Es exactamente lo que hace el sistema real cuando Ollama no responde
    (`degraded=True` en `OllamaAdapter.extract`), así que no es un escenario
    inventado: es el modo degradado, que tiene que seguir funcionando.
    """

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
def paciente(tmp_path):
    """Un paciente de apendicectomía, que es el de la llamada reportada."""
    engine = create_engine(f"sqlite:///{tmp_path}/test.db", future=True)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine, expire_on_commit=False, future=True)
    with Session() as s:
        p = Patient(name="Elena González", surgery="Apendicectomía")
        s.add(p)
        s.commit()
        yield s, p


def _estado(session, conversation_id: str) -> CallState:
    ultimo = (
        session.query(Turn)
        .filter(Turn.conversation_id == conversation_id)
        .order_by(Turn.created_at)
        .all()[-1]
    )
    return CallState.from_dict(ultimo.agent_state)


# --- el caso reportado --------------------------------------------------------


def test_el_procedimiento_se_reconoce_en_voz_alta(paciente):
    """El bug exacto: la respuesta ignoraba por completo lo que el paciente dijo."""
    session, p = paciente
    primero = process_turn(session, text="un 4, ahí va", patient_id=p.id)
    r = process_turn(session, text="me operaron de cesárea",
                     conversation_id=primero.conversation_id)

    assert "cesárea" in r.response.lower(), "no nombró lo que el paciente contó"
    assert "apendicectomía" in r.response.lower(), "no dijo qué tiene registrado"
    # Y se retoma el guion: reconocer no es abandonar la pregunta pendiente.
    assert any(p in r.response.lower()
               for p in ("fiebre", "calentura", "caliente", "escalofríos"))


def test_contar_la_cirugia_no_gasta_una_repregunta(paciente):
    """El paciente no esquivó la pregunta: corrigió el registro.

    Misma doctrina que `intento_real` para el saludo (`script.apply`). Sin esto,
    dos frases sobre su cirugía agotaban los dos intentos del slot y el guion lo
    daba por perdido sin haberlo intentado de verdad.
    """
    session, p = paciente
    primero = process_turn(session, text="un 4, ahí va", patient_id=p.id)
    process_turn(session, text="me operaron de cesárea",
                 conversation_id=primero.conversation_id)

    estado = _estado(session, primero.conversation_id)
    assert estado.slot_actual == "fiebre", "el guion no puede avanzar sin el dato"
    assert estado.repreguntas.get("fiebre", 0) == 0
    assert estado.procedimiento_dicho == "Cesárea"


def test_solo_se_reconoce_la_primera_vez(paciente):
    """Repetirlo ocho veces no puede acusarse ocho veces.

    Mismo criterio que `_aporto_dato` ya aplicaba a los síntomas fuera de
    catálogo: repetir un dato no es información nueva.
    """
    session, p = paciente
    primero = process_turn(session, text="un 4, ahí va", patient_id=p.id)
    uno = process_turn(session, text="me operaron de cesárea",
                       conversation_id=primero.conversation_id)
    dos = process_turn(session, text="me operaron de cesárea",
                       conversation_id=primero.conversation_id)

    assert "cesárea" in uno.response.lower()
    assert "cesárea" not in dos.response.lower()


def test_la_discrepancia_levanta_una_alerta_una_sola_vez(paciente):
    """Si la ficha está mal, todo lo demás está mal: alguien tiene que mirarlo."""
    session, p = paciente
    primero = process_turn(session, text="un 4, ahí va", patient_id=p.id)
    process_turn(session, text="me operaron de cesárea",
                 conversation_id=primero.conversation_id)
    process_turn(session, text="en realidad fue una cesárea",
                 conversation_id=primero.conversation_id)

    alertas = [
        a for a in session.query(Alert).all()
        if _REGLA_PROCEDIMIENTO in (a.triggered_rules or [])
    ]
    assert len(alertas) == 1
    assert alertas[0].risk_level == "ALTO"
    assert "cesárea" in alertas[0].transcript.lower()


def test_la_discrepancia_no_tine_de_amarillo_una_llamada_verde(paciente):
    """A diferencia del silencio, esto NO eleva el `risk_level` del turno.

    Una ficha equivocada es un problema de integridad del registro, no un
    deterioro del paciente. Elevarlo aquí teñiría de amarillo escenarios
    clínicamente verdes del framework de evaluación.
    """
    session, p = paciente
    primero = process_turn(session, text="un 4, ahí va", patient_id=p.id)
    r = process_turn(session, text="me operaron de cesárea",
                     conversation_id=primero.conversation_id)
    assert r.risk_level == "NORMAL"


def test_el_procedimiento_que_coincide_no_alerta(paciente):
    """Confirmar la ficha no es contradecirla."""
    session, p = paciente
    primero = process_turn(session, text="un 4, ahí va", patient_id=p.id)
    r = process_turn(session, text="a mí me operaron de apendicitis",
                     conversation_id=primero.conversation_id)

    assert not [
        a for a in session.query(Alert).all()
        if _REGLA_PROCEDIMIENTO in (a.triggered_rules or [])
    ]
    assert r.risk_level == "NORMAL"


# --- el guard del RAG ---------------------------------------------------------


def test_no_se_cita_evidencia_de_otra_cirugia(paciente):
    """Lo que produjo la respuesta más dañina de la llamada reportada.

    El corpus está indexado POR procedimiento
    (`rag/retrieve.py::_allowed_document_ids`), así que una pregunta sobre la
    cesárea solo puede recuperar documentos de apendicectomía: una afirmación
    fuera de tema CON fuente, que es lo peor que puede producir este sistema.
    """
    session, p = paciente
    primero = process_turn(session, text="un 4, ahí va", patient_id=p.id)
    process_turn(session, text="me operaron de cesárea",
                 conversation_id=primero.conversation_id)
    r = process_turn(session, text="¿qué cuidados debo tener con la herida?",
                     conversation_id=primero.conversation_id)

    assert r.sources == [], "no puede citar documentos de otra cirugía"
    assert "enfermería" in r.response.lower()
    assert "apendicectomía" in r.response.lower(), "debe decir qué tiene registrado"


# --- los casos que el usuario pidió cubrir ------------------------------------


def test_una_senal_critica_interrumpe_el_guion(paciente):
    """Caso 4: "está bien, pero me cuesta respirar".

    La bandera no estaba en `lexicon._BANDERAS` —solo cubría "no puedo
    respirar"— y el turno se resolvía como `wound=normal` siguiendo el guion.
    """
    session, p = paciente
    primero = process_turn(session, text="un 3", patient_id=p.id)
    r = process_turn(session, text="Está bien, pero me cuesta respirar",
                     conversation_id=primero.conversation_id)

    assert r.risk_level == "CRÍTICO"
    assert "dificultad_respiratoria" in r.triggered_rules
    assert r.critical_override is True


def test_preguntar_por_la_fiebre_no_es_tenerla(paciente):
    """Caso 7: la pregunta no puede convertirse en un hallazgo.

    Fallaba por dos vías a la vez: `lexicon._FIEBRE_SI` abre con `\\bfiebre` sin
    anclar, y la guarda del LLM (`ollama_adapter._to_symptoms`) comprobaba si el
    turno "menciona el slot" sobre el texto ENTERO — y una pregunta sobre la
    fiebre menciona la fiebre.
    """
    session, p = paciente
    primero = process_turn(session, text="un 3", patient_id=p.id)
    r = process_turn(session, text="¿Qué temperatura se considera fiebre?",
                     conversation_id=primero.conversation_id)

    assert r.symptoms.fever is None


def test_la_fiebre_negada_no_se_afirma_y_los_escalofrios_se_recogen(paciente):
    """Caso 3: "No fiebre, pero sí estoy temblando".

    Antes daba `fever=True` —el paciente la NIEGA— y el agente contestaba
    "Con calentura, entonces.", que es el "no escucha" en su forma más literal.
    """
    session, p = paciente
    primero = process_turn(session, text="un 3", patient_id=p.id)
    r = process_turn(session, text="No fiebre, pero sí estoy temblando",
                     conversation_id=primero.conversation_id)

    assert r.symptoms.fever is False
    assert "escalofríos" in r.symptoms.other
    assert "calentura, entonces" not in r.response.lower()


def test_negar_la_fiebre_sin_nombrarla_resuelve_el_slot(paciente):
    """El bucle reportado: tres veces la misma pregunta a quien ya contestó.

        Agente:   ¿Ha tenido fiebre o calentura estos días?
        Paciente: No, yo me he sentido bien.
        Agente:   ¿Se ha sentido caliente o con escalofríos...?   ← repregunta 1
        Paciente: No, yo me he sentido muy bien.
        Agente:   Sin termómetro: ¿lo ha sentido como fiebre, sí o no?

    `_FIEBRE_NO` exigía que el paciente NOMBRARA el síntoma, y a una pregunta
    cerrada casi nadie lo nombra (ver `lexicon._POLAR_NO`).
    """
    session, p = paciente
    primero = process_turn(session, text="un 3", patient_id=p.id)
    r = process_turn(session, text="No, yo me he sentido bien.",
                     conversation_id=primero.conversation_id)

    assert r.symptoms.fever is False
    respuesta = r.response.lower()
    # No se compara contra la palabra "fiebre": el reflejo la dice, y con razón
    # ("Sin fiebre, bien."). Lo que no puede reaparecer es la PREGUNTA.
    assert not any(q.lower() in respuesta
                   for q in (*phrasing.PREGUNTAS["fiebre"],
                             *phrasing.REPREGUNTAS["fiebre"])), \
        f"volvió a preguntar por la fiebre: {r.response!r}"
    # Y el guion ya va por el siguiente slot, no atascado en el mismo.
    assert any(t in respuesta for t in ("camin", "mover", "movi", "levant")), \
        f"no avanzó a movilidad: {r.response!r}"


def test_afirmar_la_fiebre_sin_nombrarla_tambien_resuelve_el_slot(paciente):
    """El mismo hueco por el lado que de verdad duele.

    Un "Sí" pelado tampoco resolvía el slot, así que una fiebre AFIRMADA se
    perdía tras dos repreguntas — el falso negativo que la rúbrica considera
    catastrófico. Resuelto el slot, el guion persigue la cifra
    (`script.seguimiento_pendiente`), que es la que dispara `fiebre_38`.
    """
    session, p = paciente
    primero = process_turn(session, text="un 3", patient_id=p.id)
    r = process_turn(session, text="Sí", conversation_id=primero.conversation_id)

    assert r.symptoms.fever is True
    assert "termómetro" in r.response.lower() or "medír" in r.response.lower(), \
        f"no persiguió la cifra: {r.response!r}"


# --- la segunda llamada reportada (conversación 0cf3f8d3) ---------------------
# El agente le dijo "no me supo decir" a quien acababa de contestar, le ofreció
# dos veces una enfermera y cerró en CRÍTICO por `no_se_pudo_evaluar`. La llamada
# corrió con `degraded=1` y `llm_calls=0` —la cuota diaria de Groq agotada—, así
# que `_SoloLexico` no es aquí un doble de conveniencia: es el modo exacto en que
# ocurrió.


def test_la_respuesta_a_la_repregunta_de_movilidad_se_entiende(paciente):
    """Turno 5 real: "Me cuesta un poco más." -> "no me supo decir"."""
    session, p = paciente
    r = process_turn(session, text="un 5", patient_id=p.id)
    for texto in ("no he tenido fiebre", "Me cuesta un poco más."):
        r = process_turn(session, text=texto, conversation_id=r.conversation_id)

    assert r.symptoms.mobility == "limitada_esperada"
    assert "no me supo decir" not in r.response.lower()
    assert "enfermera" not in r.response.lower()


def test_la_herida_roja_no_se_pierde(paciente):
    """Turnos 6 y 7 reales: "la área está muy roja", dos veces, y nada.

    Es el fallo grave de la llamada: un hallazgo clínico real —el que después
    escala— tirado dos veces porque el léxico solo reconocía el diminutivo
    ("rojita") o "roja en el borde".
    """
    session, p = paciente
    r = process_turn(session, text="un 5", patient_id=p.id)
    for texto in ("no he tenido fiebre", "camino bien", "Sí, la área está muy roja."):
        r = process_turn(session, text=texto, conversation_id=r.conversation_id)

    assert r.symptoms.wound == "eritema_leve"
    assert "enfermera que lo llame" not in r.response.lower()


def test_declinar_la_enfermera_retoma_la_llamada(paciente):
    """Turno 8 real: el paciente dijo "No." y el agente colgó igual.

    Se fuerza el atasco con turnos que de verdad no aportan nada (ruido del STT),
    que es el único caso en que la oferta debe salir.
    """
    session, p = paciente
    r = process_turn(session, text="un 5", patient_id=p.id)
    for _ in range(4):
        r = process_turn(session, text="xhtwkq fewf", conversation_id=r.conversation_id)
    assert "enfermera" in r.response.lower(), "la oferta debía salir aquí"

    r = process_turn(session, text="No, sigamos", conversation_id=r.conversation_id)

    assert "enfermera" not in r.response.lower(), "insistió con la oferta rechazada"
    assert r.response.rstrip().endswith("?"), "no retomó el guion"
    assert _estado(session, r.conversation_id).sin_progreso == 0


def test_aceptar_la_enfermera_cierra_y_alerta(paciente):
    session, p = paciente
    r = process_turn(session, text="un 5", patient_id=p.id)
    for _ in range(4):
        r = process_turn(session, text="xhtwkq fewf", conversation_id=r.conversation_id)
    assert "enfermera" in r.response.lower()

    r = process_turn(session, text="Sí, por favor", conversation_id=r.conversation_id)

    assert not r.response.rstrip().endswith("?"), "la oferta se aceptó: no se sigue preguntando"
    assert session.query(Alert).filter(Alert.conversation_id == r.conversation_id).count()


def test_la_fiebre_puede_corregirse_hacia_arriba_pero_no_hacia_abajo(paciente):
    """Caso 6, con el matiz que el sistema impone a propósito.

    `nlu/merge.py` es monótono por severidad y eso NO es un descuido: el paciente
    minimizador dice primero "me duele un resto" y dos turnos después "no, si
    estoy bien", y el primero es el que cuenta. Así que False -> True se acepta
    (es empeorar) y True -> False no (sería des-escalar dentro de la llamada).
    El historial completo queda igualmente en la traza, una fila por turno.
    """
    session, p = paciente
    primero = process_turn(session, text="un 3", patient_id=p.id)
    sin = process_turn(session, text="No he tenido fiebre",
                       conversation_id=primero.conversation_id)
    assert sin.symptoms.fever is False

    con = process_turn(session, text="Ahora que lo pienso, sí tengo calentura",
                       conversation_id=primero.conversation_id)
    assert con.symptoms.fever is True

    vuelta = process_turn(session, text="No, mentiras, no tengo fiebre",
                          conversation_id=primero.conversation_id)
    assert vuelta.symptoms.fever is True, "no se des-escala dentro de la llamada"


# --- la red general: ninguna repregunta vuelve a salir pelada -----------------


def test_una_repregunta_reconoce_haber_escuchado(paciente):
    """El agujero de fondo, más allá del procedimiento.

    Cuando el turno no dejaba ningún dato, la respuesta era la reformulación
    cerrada a secas. `ACUSE_ESCUCHADO` reconoce haber OÍDO sin afirmar haber
    anotado nada — decir "vale, anotado" ahí sería falso.
    """
    from app.agent import phrasing

    session, p = paciente
    primero = process_turn(session, text="un 4, ahí va", patient_id=p.id)
    r = process_turn(session, text="mi hija está aquí conmigo en la casa",
                     conversation_id=primero.conversation_id)

    assert any(a in r.response for a in phrasing.ACUSE_ESCUCHADO), r.response
    assert not any(a in r.response for a in phrasing.ACUSES), "no se anotó nada"

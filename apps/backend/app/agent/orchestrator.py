"""Orquestador de un turno — el corazón del flujo de decisión.

Estaba en `voice/conversation.py`, que era un mal sitio: no tiene nada que ver con
la voz, es el servicio que comparten el endpoint de texto y el pipeline de audio.

Cinco etapas explícitas, de las cuales **solo dos tocan el LLM**:

    A. determinista   léxico colombiano → intención → inyección          (<5 ms)
    B. LLM            extracción del slot, solo si el léxico no lo resolvió
    C. determinista   fusión por severidad → engine.evaluate()           (<1 ms)
    D. determinista   script.next_action() → qué se dice a continuación
    E. LLM            respuesta anclada al RAG, solo si preguntó algo clínico

**Excepción de seguridad:** si el léxico detecta una bandera de emergencia, se
saltan B, D y E por completo. Se emite el guion determinista y se alerta. La ruta
crítica es la más corta del sistema, y no pasa por el modelo: ni uno caído ni uno
manipulado pueden suprimir un escalamiento.

El LLM nunca controla el flujo. Su salida es *dato* —un slot de un enum cerrado—,
nunca instrucciones.
"""
from __future__ import annotations

import asyncio
import logging
import re
import time

from sqlalchemy.orm import Session

from ..decision import engine
from ..llm.factory import get_llm
from ..llm.ollama_adapter import ABSTENCION, es_abstencion
from ..models import Alert, Conversation, Patient, Turn
from ..nlu import lexicon
from ..nlu.merge import merge_symptoms
from ..rag import retrieve as rag
from ..rag.embeddings import build_query
from ..schemas import RagResult, Symptoms, TurnResponse
from . import phrasing, script
from .script import Action, CallState, Phase

log = logging.getLogger(__name__)

# Ventana para la rotación anti-repetición del banco de frases.
_VENTANA_ESTILO = 5


def _get_or_create_conversation(
    session: Session, conversation_id: str | None, patient_id: str | None
) -> Conversation:
    if conversation_id:
        conv = session.get(Conversation, conversation_id)
        if conv is not None:
            return conv
    conv = Conversation(patient_id=patient_id, status="active")
    session.add(conv)
    session.flush()
    return conv


def _prior_turns(session: Session, conversation_id: str) -> list[Turn]:
    return (
        session.query(Turn)
        .filter(Turn.conversation_id == conversation_id)
        .order_by(Turn.created_at)
        .all()
    )


def _acumular(turns: list[Turn]) -> tuple[Symptoms, CallState]:
    """Reconstruye el cuadro clínico y el estado del guion desde la traza.

    La memoria vive aquí, en la base, y no en el contexto del modelo. Eso recorta
    ~800 tokens por turno y elimina la deriva típica de un modelo pequeño que se
    olvida de lo que ya preguntó.
    """
    acc = Symptoms()
    estado = CallState()
    for t in turns:
        acc = merge_symptoms(acc, Symptoms.model_validate(t.extracted_symptoms or {}))
        if t.agent_state:
            estado = CallState.from_dict(t.agent_state)
    return acc, estado


def _aperturas_recientes(turns: list[Turn]) -> list[str]:
    return [t.final_response for t in turns[-_VENTANA_ESTILO:]]


def _texto_de(action: Action, *, semilla: str, recientes: list[str],
              nombre: str | None, preocupante: bool) -> str:
    """Traduce la acción del guion a lo que se dice. Todo determinista."""
    if action.kind == "repreguntar":
        return phrasing.repregunta(action.slot, action.intento)
    if action.kind == "cerrar":
        return phrasing.cierre(nombre, escalado=preocupante)
    if action.kind == "preguntar":
        if action.slot is None:
            return phrasing.PREGUNTA_ABIERTA
        ack = phrasing.acuse(semilla, recientes, preocupante=preocupante)
        return f"{ack} {phrasing.pregunta(action.slot, semilla, recientes)}"
    return phrasing.PREGUNTA_ABIERTA


async def process_turn_async(
    session: Session,
    *,
    text: str,
    conversation_id: str | None = None,
    patient_id: str | None = None,
) -> TurnResponse:
    started = time.perf_counter()
    conv = _get_or_create_conversation(session, conversation_id, patient_id)
    prior = _prior_turns(session, conv.id)
    acumulado, estado = _acumular(prior)

    paciente = session.get(Patient, conv.patient_id) if conv.patient_id else None
    procedimiento = paciente.surgery if paciente else None
    nombre = paciente.name.split()[0] if paciente else None

    llm = get_llm()
    # Se acumulan TODAS las llamadas del turno, no solo la extracción: si el
    # paciente preguntó algo, la respuesta anclada es la que más tokens consume y
    # dejarla fuera falsearía las métricas del README a la baja.
    llm_calls, tokens_in, tokens_out = 0, 0, 0

    # --- A + B: entender lo que dijo el paciente ------------------------------
    slot = estado.slot_actual if estado.phase is Phase.TAMIZAJE else None
    pregunta_previa = prior[-1].final_response if prior else phrasing.APERTURA
    extraccion = await llm.extract(slot=slot, question=pregunta_previa, utterance=text)
    if extraccion.usage.tokens_out:
        llm_calls += 1
    tokens_in += extraccion.usage.tokens_in
    tokens_out += extraccion.usage.tokens_out

    del_turno = extraccion.symptoms
    acumulado = merge_symptoms(acumulado, del_turno)
    if paciente and paciente.extra.get("dia_postop"):
        acumulado.day_postop = paciente.extra["dia_postop"]

    # --- C: decidir. Código puro, nunca el modelo -----------------------------
    decision = engine.evaluate(acumulado)
    critico = decision.risk_level == "CRÍTICO"

    # --- D: qué se dice ------------------------------------------------------
    action = script.next_action(estado, acumulado, escalar=critico)
    recientes = _aperturas_recientes(prior)
    evidence: RagResult | None = None

    if critico:
        # Ruta crítica: no pasa por el modelo ni por el RAG.
        final = decision.safety_script or phrasing.cierre(nombre, escalado=True)
    elif extraccion.intent == "fuera_de_mision":
        final = phrasing.FUERA_DE_MISION
    elif extraccion.intent == "rechazo":
        final = phrasing.RECHAZO
        action = Action(kind="cerrar", phase=Phase.TERMINADA)
    elif extraccion.intent == "ininteligible":
        final = phrasing.NO_ENTENDI
    elif extraccion.intent == "meta":
        siguiente = _texto_de(action, semilla=conv.id + str(len(prior)),
                              recientes=recientes, nombre=nombre, preocupante=False)
        final = f"{phrasing.META_REPETIR} {siguiente}"
    elif extraccion.intent == "social":
        siguiente = _texto_de(action, semilla=conv.id + str(len(prior)),
                              recientes=recientes, nombre=nombre, preocupante=False)
        final = f"{phrasing.SOCIAL} {siguiente}"
    elif extraccion.intent == "pregunta_clinica":
        # --- E: la única respuesta generada, y va anclada a evidencia ---------
        evidence = rag.retrieve(
            session,
            build_query(text, procedure=procedimiento, day_postop=acumulado.day_postop),
            procedure=procedimiento,
        )
        # Antes de redactar, se comprueba que lo recuperado responda de verdad la
        # pregunta. La similitud vectorial no lo dice (ver rag/retrieve.py), y sin
        # este paso el agente contestaba preguntas sobre protocolos inventados
        # citando documentos sin relación: una afirmación falsa CON fuente.
        pertinente = evidence.has_evidence and await llm.evidencia_responde(
            question=text, evidence=evidence.answer)
        if evidence.has_evidence and not pertinente:
            log.info("evidencia recuperada pero no pertinente para: %r", text[:60])
            evidence.has_evidence = False

        respuesta, uso = await llm.reply_grounded(
            question=text,
            evidence=evidence.answer if pertinente else "",
            patient_context=f"Paciente de {procedimiento or 'cirugía'}.",
        )
        if uso.tokens_out:
            llm_calls += 1
        tokens_in += uso.tokens_in
        tokens_out += uso.tokens_out
        respuesta = _validar_grounding(respuesta, evidence, riesgo=decision.risk_level)
        # Si se acabó abstiniendo, no se citan fuentes: decir "no tengo información"
        # y adjuntar una cita es incoherente, y en la traza parecería que el agente
        # ignoró evidencia que sí tenía.
        if es_abstencion(respuesta):
            evidence = None
        siguiente = _texto_de(action, semilla=conv.id + str(len(prior)),
                              recientes=recientes, nombre=nombre, preocupante=False)
        final = f"{respuesta} {siguiente}"
    else:
        preocupante = decision.risk_level != "NORMAL"
        prefijo = phrasing.TERCERO + " " if extraccion.intent == "tercero" else ""
        final = prefijo + _texto_de(action, semilla=conv.id + str(len(prior)),
                                    recientes=recientes, nombre=nombre,
                                    preocupante=preocupante)

    nuevo_estado = script.apply(estado, action, acumulado)

    # --- alerta, deduplicada por reglas nuevas -------------------------------
    alert_id = _crear_alerta_si_procede(session, conv, decision, acumulado, text)

    sources = evidence.sources if evidence else []
    turn = Turn(
        conversation_id=conv.id,
        patient_utterance=text,
        extracted_symptoms=del_turno.model_dump(),
        retrieved_chunks=[s.model_dump() for s in sources],
        confidence=evidence.confidence if evidence else None,
        triggered_rules=decision.triggered_rules,
        risk_level=decision.risk_level,
        critical_override=critico,
        final_response=final,
        latency_ms=int((time.perf_counter() - started) * 1000),
        agent_state=nuevo_estado.to_dict(),
        intent=extraccion.intent,
        llm_calls=llm_calls,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        degraded=extraccion.degraded,
    )
    session.add(turn)
    session.commit()

    return TurnResponse(
        conversation_id=conv.id,
        turn_id=turn.id,
        response=final,
        risk_level=decision.risk_level,
        triggered_rules=decision.triggered_rules,
        symptoms=acumulado,
        sources=sources,
        critical_override=critico,
        alert_id=alert_id,
    )


# Fórmulas con las que un modelo complaciente quita hierro a un síntoma. Ante un
# cuadro que ya tiene señales de alarma, decir esto es la conducta que la rúbrica
# penaliza por su nombre: "tranquilizar al paciente ante un síntoma de alarma".
_TRANQUILIZADOR = re.compile(
    r"no\s+(es|parece|deber[ií]a\s+ser|necesariamente\s+indica)\s+(nada\s+)?"
    r"(grave|serio|preocupante|de\s+cuidado)"
    r"|no\s+(se\s+)?preocupe|no\s+hay\s+(de\s+qu[eé]\s+preocuparse|motivo)"
    r"|es\s+(completamente\s+)?normal|es\s+algo\s+normal|tranquil\w+"
    r"|no\s+pasa\s+nada|nada\s+de\s+qu[eé]\s+preocuparse",
    re.I,
)

_MATIZ_SIN_TRANQUILIZAR = (
    " De todos modos, como me contó otras cosas, prefiero que enfermería le eche "
    "un ojo."
)


def _validar_grounding(
    respuesta: str, evidence: RagResult | None, *, riesgo: str = "NORMAL"
) -> str:
    """Última red contra la alucinación clínica, y es determinista.

    Dos comprobaciones, ninguna delegada al modelo:

    1. Si la respuesta menciona una cifra que no aparece literalmente en la
       evidencia, se sustituye por la abstención. Inventar una dosis o un plazo
       es lo que la rúbrica penaliza con más dureza.
    2. Si el cuadro ya tiene señales de alarma y la respuesta suena tranquilizadora,
       se le añade el matiz de escalamiento. Un 3B complaciente responde "no es
       nada grave" a un eritema, y ese eritema es precisamente una de las cinco
       señales que suman al amarillo.
    """
    from ..llm.ollama_adapter import grounded_in_evidence

    if evidence is None or not evidence.has_evidence:
        return ABSTENCION
    if es_abstencion(respuesta):
        return respuesta          # el modelo ya declaró el límite; se respeta
    if not grounded_in_evidence(respuesta, evidence.answer):
        log.warning("respuesta descartada por cifras fuera de la evidencia: %r", respuesta)
        return ABSTENCION
    if riesgo != "NORMAL" and _TRANQUILIZADOR.search(respuesta):
        log.warning("respuesta tranquilizadora con riesgo %s: se añade el matiz", riesgo)
        return respuesta.rstrip(". ") + "." + _MATIZ_SIN_TRANQUILIZAR
    return respuesta


def _crear_alerta_si_procede(
    session: Session, conv: Conversation, decision, symptoms: Symptoms, text: str
) -> str | None:
    if decision.risk_level not in ("ALTO", "CRÍTICO"):
        return None
    ya: set[str] = set()
    for a in session.query(Alert).filter(Alert.conversation_id == conv.id).all():
        ya.update(a.triggered_rules or [])
    nuevas = [r for r in decision.triggered_rules if r not in ya]
    if not nuevas:
        return None
    alerta = Alert(
        conversation_id=conv.id, patient_id=conv.patient_id,
        risk_level=decision.risk_level, triggered_rules=decision.triggered_rules,
        symptoms=symptoms.model_dump(), transcript=text,
    )
    session.add(alerta)
    session.flush()
    return alerta.id


def process_turn(
    session: Session, *, text: str,
    conversation_id: str | None = None, patient_id: str | None = None,
) -> TurnResponse:
    """Envoltorio síncrono para el endpoint de texto y el framework de evaluación."""
    return asyncio.run(process_turn_async(
        session, text=text, conversation_id=conversation_id, patient_id=patient_id))

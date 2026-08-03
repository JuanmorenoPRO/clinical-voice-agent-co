"""Servicio de Conversación — el corazón del flujo de decisión (architecture.md §4).

Orquesta UN turno completo (ADR-006) MANTENIENDO EL CONTEXTO de la llamada
(architecture.md §3.1):
  1. Se reconstruye el historial de la conversación y se pasa al LLM para que
     haga preguntas adaptativas SIN repetir lo ya respondido.
  2. Si el paciente pregunta algo -> RAG recupera evidencia (siempre, sin
     clasificador previo; ADR-005) y se inyecta en el prompt.
  3. UNA sola llamada LLM con salida estructurada {sintomas, respuesta}.
  4. Los síntomas se ACUMULAN a lo largo de la llamada: el Motor de Decisión
     evalúa el cuadro clínico completo, no solo la última frase. Así "dolor 9" en
     un turno + "la medicación no me sirve" en otro disparan la regla ALTO.
  5. Si el nivel es CRÍTICO -> se DESCARTA la respuesta del LLM y se emite el
     guion determinista de seguridad + alerta + cierre seguro.
  6. Se persiste la traza completa del turno (RF-05).

El mismo servicio alimenta el endpoint de texto y el pipeline Pipecat de voz.
"""
from __future__ import annotations

import time

from sqlalchemy.orm import Session

from ..decision import engine
from ..llm.factory import get_llm
from ..models import Alert, Conversation, Patient, Turn
from ..prompts_loader import load_prompt
from ..rag import retrieve as rag
from ..schemas import RagResult, Symptoms, TurnResponse

_QUESTION_MARKERS = ("?", "¿", "puedo", "debo", "es normal", "qué hago", "cuándo", "por qué")

# Cuántos turnos previos se incluyen como contexto del LLM (ventana de diálogo).
_HISTORY_TURNS = 12


def _looks_like_question(text: str) -> bool:
    t = text.lower()
    return any(m in t for m in _QUESTION_MARKERS)


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


def _merge_symptoms(base: Symptoms, new: Symptoms) -> Symptoms:
    """Superpone los valores no nulos de `new` sobre `base` (el último gana).

    Acumulación conservadora dentro de una misma llamada (RNF-04: ante la duda,
    escalar). `other` se une sin duplicar.
    """
    merged = base.model_copy(deep=True)
    for field in (
        "pain_level",
        "medication_effective",
        "temperature_c",
        "fever",
        "heavy_bleeding",
        "breathing_difficulty",
        "loss_of_consciousness",
    ):
        value = getattr(new, field)
        if value is not None:
            setattr(merged, field, value)
    for item in new.other:
        if item not in merged.other:
            merged.other.append(item)
    return merged


def _accumulate_prior(turns: list[Turn]) -> Symptoms:
    acc = Symptoms()
    for turn in turns:
        acc = _merge_symptoms(acc, Symptoms.model_validate(turn.extracted_symptoms or {}))
    return acc


def _build_user_prompt(
    turns: list[Turn], patient_text: str, evidence: RagResult | None
) -> str:
    parts: list[str] = []

    recent = turns[-_HISTORY_TURNS:]
    if recent:
        history_lines = []
        for turn in recent:
            history_lines.append(f"Paciente: {turn.patient_utterance}")
            history_lines.append(f"Agente: {turn.final_response}")
        parts.append(
            "Historial de la conversación (del más antiguo al más reciente). No "
            "repitas preguntas ya respondidas; avanza el chequeo:\n"
            + "\n".join(history_lines)
        )

    if evidence is not None:
        if evidence.has_evidence:
            docs = sorted({s.document for s in evidence.sources})
            # Le damos al modelo los documentos fuente y le pedimos verificar la
            # RELEVANCIA temática: la similitud vectorial no distingue "mismo
            # dominio, procedimiento distinto" (p. ej. una pregunta de cesárea
            # recupera con alta similitud docs de apendicectomía). El prompt debe
            # ignorar contexto irrelevante (ADR-005).
            parts.append(
                "Evidencia recuperada de estos documentos: "
                + ", ".join(docs)
                + ".\nEsos documentos tratan procedimientos o temas específicos. "
                "Úsalos SOLO si corresponden a lo que pregunta el paciente (mismo "
                "procedimiento o tema). Si la pregunta es sobre un procedimiento o "
                "tema que esos documentos NO cubren, NO respondas desde tu "
                "conocimiento interno: di que no tienes evidencia específica sobre "
                "eso y ofrece escalar a enfermería.\n\nContenido:\n"
                + evidence.answer
            )
        else:
            parts.append(
                "No hay evidencia suficiente en los documentos para la pregunta. "
                "Dilo explícitamente y ofrece escalar al personal de enfermería."
            )

    parts.append(f"Paciente: {patient_text}")
    return "\n\n".join(parts)


def process_turn(
    session: Session,
    *,
    text: str,
    conversation_id: str | None = None,
    patient_id: str | None = None,
) -> TurnResponse:
    started = time.perf_counter()
    conv = _get_or_create_conversation(session, conversation_id, patient_id)
    prior = _prior_turns(session, conv.id)

    # 1) RAG (solo si el paciente pregunta algo). Si conocemos al paciente,
    #    filtramos la evidencia por su procedimiento (grounding determinista).
    evidence: RagResult | None = None
    if _looks_like_question(text):
        procedure = None
        if conv.patient_id:
            patient = session.get(Patient, conv.patient_id)
            if patient:
                procedure = patient.surgery
        evidence = rag.retrieve(session, text, procedure=procedure)

    # 2) Una sola llamada LLM con el HISTORIAL como contexto: {sintomas, respuesta}.
    llm = get_llm()
    llm_out = llm.turn(
        system_prompt=load_prompt("system"),
        user_prompt=_build_user_prompt(prior, text, evidence),
    )

    # 3) Acumular síntomas de toda la llamada y evaluar el Motor de Decisión sobre
    #    el cuadro completo (no solo la última frase).
    accumulated = _merge_symptoms(_accumulate_prior(prior), llm_out.sintomas)
    decision = engine.evaluate(accumulated)

    # 4) Override crítico: descartar redacción del LLM.
    critical_override = decision.risk_level == "CRÍTICO"
    final_response = decision.safety_script if critical_override else llm_out.respuesta

    # 5) Alerta si ALTO/CRÍTICO, evitando duplicar: solo si hay reglas nuevas
    #    respecto a las ya alertadas en esta conversación.
    alert_id: str | None = None
    if decision.risk_level in ("ALTO", "CRÍTICO"):
        already: set[str] = set()
        for a in session.query(Alert).filter(Alert.conversation_id == conv.id).all():
            already.update(a.triggered_rules or [])
        new_rules = [r for r in decision.triggered_rules if r not in already]
        if new_rules:
            alert = Alert(
                conversation_id=conv.id,
                patient_id=conv.patient_id,
                risk_level=decision.risk_level,
                triggered_rules=decision.triggered_rules,
                symptoms=accumulated.model_dump(),
                transcript=text,
            )
            session.add(alert)
            session.flush()
            alert_id = alert.id

    sources = evidence.sources if evidence else []
    latency_ms = int((time.perf_counter() - started) * 1000)

    # 6) Persistir la traza completa (RF-05). `extracted_symptoms` guarda la
    #    extracción CRUDA de este turno; el nivel de riesgo refleja la evaluación
    #    sobre el cuadro ACUMULADO (lo que realmente decidió).
    turn = Turn(
        conversation_id=conv.id,
        patient_utterance=text,
        extracted_symptoms=llm_out.sintomas.model_dump(),
        retrieved_chunks=[s.model_dump() for s in sources],
        confidence=evidence.confidence if evidence else None,
        triggered_rules=decision.triggered_rules,
        risk_level=decision.risk_level,
        critical_override=critical_override,
        final_response=final_response,
        latency_ms=latency_ms,
    )
    session.add(turn)
    session.commit()

    return TurnResponse(
        conversation_id=conv.id,
        turn_id=turn.id,
        response=final_response,
        risk_level=decision.risk_level,
        triggered_rules=decision.triggered_rules,
        symptoms=accumulated,  # cuadro acumulado que impulsó la decisión
        sources=sources,
        critical_override=critical_override,
        alert_id=alert_id,
    )

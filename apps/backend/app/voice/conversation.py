"""Servicio de Conversación — el corazón del flujo de decisión (architecture.md §4).

Orquesta UN turno completo (ADR-006):
  1. Si el paciente pregunta algo -> RAG recupera evidencia (siempre, sin
     clasificador previo; ADR-005) y se inyecta en el prompt.
  2. UNA sola llamada LLM con salida estructurada {sintomas, respuesta}.
  3. El Motor de Decisión evalúa los síntomas ANTES de emitir audio/respuesta.
  4. Si el nivel es CRÍTICO -> se DESCARTA la respuesta del LLM y se emite el
     guion determinista de seguridad + alerta + cierre seguro.
  5. Se persiste la traza completa del turno (RF-05).

El mismo servicio alimenta tanto el endpoint de texto (probable hoy) como el
pipeline Pipecat de voz.
"""
from __future__ import annotations

import time

from sqlalchemy.orm import Session

from ..decision import engine
from ..llm.factory import get_llm
from ..models import Alert, Conversation, Turn
from ..prompts_loader import load_prompt
from ..rag import retrieve as rag
from ..schemas import RagResult, TurnResponse

# Heurística mínima para decidir si el paciente hace una pregunta clínica.
# (El RAG es barato; ante la duda, se recupera igual — ADR-005.)
_QUESTION_MARKERS = ("?", "¿", "puedo", "debo", "es normal", "qué hago", "cuándo", "por qué")


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


def _build_user_prompt(patient_text: str, evidence: RagResult | None) -> str:
    parts: list[str] = []
    if evidence is not None:
        if evidence.has_evidence:
            parts.append(
                "Evidencia recuperada de los documentos (úsala para fundamentar "
                "la respuesta; NO respondas desde tu conocimiento interno):\n"
                f"{evidence.answer}"
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

    # 1) RAG (solo si el paciente pregunta algo).
    evidence: RagResult | None = None
    if _looks_like_question(text):
        evidence = rag.retrieve(session, text)

    # 2) Una sola llamada LLM: {sintomas, respuesta}.
    llm = get_llm()
    llm_out = llm.turn(
        system_prompt=load_prompt("system"),
        user_prompt=_build_user_prompt(text, evidence),
    )

    # 3) Motor de Decisión (determinista) ANTES de emitir la respuesta.
    decision = engine.evaluate(llm_out.sintomas)

    # 4) Override crítico: descartar redacción del LLM.
    critical_override = decision.risk_level == "CRÍTICO"
    final_response = decision.safety_script if critical_override else llm_out.respuesta

    # 5) Alerta si ALTO/CRÍTICO.
    alert_id: str | None = None
    if decision.risk_level in ("ALTO", "CRÍTICO"):
        alert = Alert(
            conversation_id=conv.id,
            patient_id=conv.patient_id,
            risk_level=decision.risk_level,
            triggered_rules=decision.triggered_rules,
            symptoms=llm_out.sintomas.model_dump(),
            transcript=text,
        )
        session.add(alert)
        session.flush()
        alert_id = alert.id

    sources = evidence.sources if evidence else []
    latency_ms = int((time.perf_counter() - started) * 1000)

    # Persistir la traza completa (RF-05).
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
        symptoms=llm_out.sintomas,
        sources=sources,
        critical_override=critical_override,
        alert_id=alert_id,
    )

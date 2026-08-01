"""Consola de administración: conversaciones, alertas y trazabilidad (RF-11).

Las alertas se consultan por polling desde el frontend (sin WebSocket, ADR §3.6).
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from ..db import get_session
from ..models import Alert, Conversation, Patient, Summary, Turn

router = APIRouter(prefix="/console", tags=["console"])


@router.get("/patients")
def patients(session: Session = Depends(get_session)) -> list[dict]:
    rows = session.query(Patient).all()
    return [{"id": p.id, "name": p.name, "surgery": p.surgery} for p in rows]


@router.get("/alerts")
def alerts(session: Session = Depends(get_session)) -> list[dict]:
    rows = session.query(Alert).order_by(Alert.created_at.desc()).all()
    return [
        {
            "id": a.id,
            "conversation_id": a.conversation_id,
            "risk_level": a.risk_level,
            "triggered_rules": a.triggered_rules,
            "symptoms": a.symptoms,
            "transcript": a.transcript,
            "status": a.status,
            "created_at": a.created_at.isoformat(),
        }
        for a in rows
    ]


@router.post("/alerts/{alert_id}/attend")
def attend_alert(alert_id: str, session: Session = Depends(get_session)) -> dict:
    alert = session.get(Alert, alert_id)
    if alert is None:
        raise HTTPException(404, "Alerta no encontrada")
    alert.status = "attended"
    session.commit()
    return {"id": alert_id, "status": alert.status}


@router.get("/conversations")
def conversations(session: Session = Depends(get_session)) -> list[dict]:
    rows = session.query(Conversation).order_by(Conversation.started_at.desc()).all()
    return [
        {
            "id": c.id,
            "patient_id": c.patient_id,
            "status": c.status,
            "started_at": c.started_at.isoformat(),
            "ended_at": c.ended_at.isoformat() if c.ended_at else None,
        }
        for c in rows
    ]


@router.get("/conversations/{conversation_id}")
def conversation_detail(conversation_id: str, session: Session = Depends(get_session)) -> dict:
    conv = session.get(Conversation, conversation_id)
    if conv is None:
        raise HTTPException(404, "Conversación no encontrada")
    turns = (
        session.query(Turn)
        .filter(Turn.conversation_id == conversation_id)
        .order_by(Turn.created_at)
        .all()
    )
    summary = (
        session.query(Summary).filter(Summary.conversation_id == conversation_id).first()
    )
    return {
        "id": conv.id,
        "status": conv.status,
        # Traza completa por turno (RF-05): pregunta -> chunks -> confianza ->
        # reglas -> respuesta -> override crítico.
        "turns": [
            {
                "id": t.id,
                "patient_utterance": t.patient_utterance,
                "extracted_symptoms": t.extracted_symptoms,
                "retrieved_chunks": t.retrieved_chunks,
                "confidence": t.confidence,
                "triggered_rules": t.triggered_rules,
                "risk_level": t.risk_level,
                "critical_override": t.critical_override,
                "final_response": t.final_response,
                "latency_ms": t.latency_ms,
            }
            for t in turns
        ],
        "summary": summary.data if summary else None,
    }

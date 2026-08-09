"""Consola de administración: conversaciones, alertas y trazabilidad (RF-11).

Las alertas se consultan por polling desde el frontend (sin WebSocket, ADR §3.6).
"""
from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func
from sqlalchemy.orm import Session

from ..db import get_session
from ..models import Alert, Conversation, Patient, Summary, Turn

router = APIRouter(prefix="/console", tags=["console"])

# Valores admitidos en los filtros de la consola. Se declaran como `Literal` para
# que FastAPI los valide y los documente solo: son el mismo vocabulario cerrado
# que `Alert.status` y `Alert.risk_level` en models.py.
ESTADO_ALERTA = Literal["pending", "attended", "deleted"]
NIVEL_ALERTA = Literal["ALTO", "CRÍTICO"]


def _pacientes_por_id(session: Session) -> dict[str, Patient]:
    """Índice de pacientes para resolver nombres sin un SELECT por fila."""
    return {p.id: p for p in session.query(Patient).all()}


@router.get("/patients")
def patients(session: Session = Depends(get_session)) -> list[dict]:
    rows = session.query(Patient).all()
    return [{"id": p.id, "name": p.name, "surgery": p.surgery} for p in rows]


@router.get("/alerts")
def alerts(
    status: ESTADO_ALERTA | None = Query(
        None, description="pending|attended|deleted. Sin valor: todo menos las borradas."
    ),
    risk_level: NIVEL_ALERTA | None = Query(None, description="ALTO|CRÍTICO"),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    session: Session = Depends(get_session),
) -> dict:
    """Alertas más recientes primero, filtrables y paginadas.

    `status` y `risk_level` son `Literal`, así que un valor fuera del conjunto lo
    rechaza FastAPI con 422. Es deliberado: un filtro con una errata que se ignora
    en silencio devuelve una lista que parece filtrada y no lo está, y en una
    consola de alertas eso es peor que un error.
    """
    q = session.query(Alert)
    if status is not None:
        q = q.filter(Alert.status == status)
    else:
        # Sin filtro explícito, las borradas no se listan pero siguen en la tabla
        # (ver `delete_alert`). Pedir `status=deleted` es lo único que las muestra.
        q = q.filter(Alert.status != "deleted")
    if risk_level is not None:
        q = q.filter(Alert.risk_level == risk_level)

    total = q.with_entities(func.count(Alert.id)).scalar() or 0

    # El desempate por `id` no significa nada —es un UUID—: existe para que el
    # orden sea DETERMINISTA. Varias alertas de la misma llamada se crean dentro
    # del mismo segundo (`paciente_no_responde` y `no_se_pudo_evaluar` lo hacen),
    # y con `created_at` como única clave el motor puede devolverlas en distinto
    # orden entre dos consultas: al paginar, una fila saldría dos veces y otra
    # ninguna.
    rows = (
        q.order_by(Alert.created_at.desc(), Alert.id.desc())
        .limit(limit)
        .offset(offset)
        .all()
    )
    pacientes = _pacientes_por_id(session)
    items = [
        {
            "id": a.id,
            "conversation_id": a.conversation_id,
            # De quién es la alerta. `patient_id` ya estaba en el modelo pero no
            # salía por la API, así que la consola listaba alertas sin dueño.
            "patient_id": a.patient_id,
            "patient": pacientes[a.patient_id].name if a.patient_id in pacientes else None,
            "surgery": pacientes[a.patient_id].surgery if a.patient_id in pacientes else None,
            "risk_level": a.risk_level,
            "triggered_rules": a.triggered_rules,
            "symptoms": a.symptoms,
            "transcript": a.transcript,
            "status": a.status,
            "created_at": a.created_at.isoformat(),
        }
        for a in rows
    ]
    # `total` es el conteo YA filtrado, no el global: es lo que sostiene el
    # "mostrando 50 de 213" de la consola y lo que decide si queda algo por traer.
    return {"items": items, "total": total, "limit": limit, "offset": offset}


@router.post("/alerts/{alert_id}/attend")
def attend_alert(alert_id: str, session: Session = Depends(get_session)) -> dict:
    alert = session.get(Alert, alert_id)
    if alert is None:
        raise HTTPException(404, "Alerta no encontrada")
    alert.status = "attended"
    session.commit()
    return {"id": alert_id, "status": alert.status}


@router.delete("/alerts/{alert_id}")
def delete_alert(alert_id: str, session: Session = Depends(get_session)) -> dict:
    """Borrado LÓGICO, y solo de lo ya atendido.

    La fila no se elimina: pasa a `status="deleted"` y deja de listarse. En un
    sistema clínico, qué se alertó y quién lo cerró es justo lo que no puede
    desaparecer, y además la deduplicación de alertas
    (`agent/orchestrator.py::_crear_alerta_si_procede` y su espejo en
    `summary/service.py`) consulta TODAS las alertas de la conversación: una
    alerta borrada de verdad haría que la misma regla volviera a dispararse.

    El 409 es la contraparte de la UI, donde el botón está deshabilitado hasta
    atender: borrar sin haber atendido es perder una alerta pendiente de vista.
    """
    alert = session.get(Alert, alert_id)
    if alert is None:
        raise HTTPException(404, "Alerta no encontrada")
    if alert.status != "attended":
        raise HTTPException(409, "Solo se puede borrar una alerta ya atendida")
    alert.status = "deleted"
    session.commit()
    return {"id": alert_id, "status": alert.status}


@router.get("/conversations")
def conversations(session: Session = Depends(get_session)) -> list[dict]:
    rows = session.query(Conversation).order_by(Conversation.started_at.desc()).all()
    pacientes = _pacientes_por_id(session)
    conteos = dict(
        session.query(Turn.conversation_id, func.count(Turn.id))
        .group_by(Turn.conversation_id)
        .all()
    )
    return [
        {
            "id": c.id,
            "patient_id": c.patient_id,
            # Sin el nombre y la fecha, el desplegable de Trazas es una lista de
            # UUID truncados: inservible en cuanto hay más de una llamada.
            "patient": pacientes[c.patient_id].name if c.patient_id in pacientes else None,
            "n_turns": conteos.get(c.id, 0),
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
    patient = session.get(Patient, conv.patient_id) if conv.patient_id else None
    return {
        "id": conv.id,
        "status": conv.status,
        "patient_id": conv.patient_id,
        "patient": patient.name if patient else None,
        "surgery": patient.surgery if patient else None,
        "started_at": conv.started_at.isoformat(),
        "ended_at": conv.ended_at.isoformat() if conv.ended_at else None,
        # Traza completa por turno (RF-05): pregunta -> chunks -> confianza ->
        # reglas -> respuesta -> override crítico.
        "turns": [
            {
                "id": t.id,
                "created_at": t.created_at.isoformat(),
                "patient_utterance": t.patient_utterance,
                "extracted_symptoms": t.extracted_symptoms,
                "retrieved_chunks": t.retrieved_chunks,
                "confidence": t.confidence,
                "triggered_rules": t.triggered_rules,
                "risk_level": t.risk_level,
                "critical_override": t.critical_override,
                "final_response": t.final_response,
                "latency_ms": t.latency_ms,
                "intent": t.intent,
            }
            for t in turns
        ],
        "summary": summary.data if summary else None,
    }

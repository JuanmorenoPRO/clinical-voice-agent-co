"""Endpoints de conversación.

- POST /conversation/turn : un turno en modo TEXTO (probable hoy, sin audio).
- POST /conversation/{id}/close : cierra la llamada y genera el resumen (RF-10).
- El endpoint WebRTC de voz se cablea con Pipecat el 7 de agosto (ver voice/pipeline.py).
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from ..db import get_session
from ..models import Conversation
from ..summary.service import close_conversation
from ..agent import orchestrator as convo
from ..schemas import TurnRequest, TurnResponse

router = APIRouter(prefix="/conversation", tags=["conversation"])


@router.post("/turn", response_model=TurnResponse)
def turn(req: TurnRequest, session: Session = Depends(get_session)) -> TurnResponse:
    return convo.process_turn(
        session,
        text=req.text,
        conversation_id=req.conversation_id,
        patient_id=req.patient_id,
    )


@router.post("/{conversation_id}/close")
def close(conversation_id: str, session: Session = Depends(get_session)) -> dict:
    if session.get(Conversation, conversation_id) is None:
        raise HTTPException(404, "Conversación no encontrada")
    return {"conversation_id": conversation_id,
            "summary": close_conversation(session, conversation_id)}

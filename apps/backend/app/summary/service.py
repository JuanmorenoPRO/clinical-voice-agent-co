"""Servicio de Resúmenes — reporte estructurado por llamada (RF-10).

Al cerrar la conversación agrega la traza de todos los turnos en un JSON:
paciente, cirugía, duración, síntomas, entidades, documentos citados, nivel de
riesgo, reglas disparadas y recomendación. El nivel de riesgo del resumen es el
MÁXIMO alcanzado en la llamada (determinista, no lo decide el LLM).
"""
from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy.orm import Session

from ..models import Conversation, Patient, Summary, Turn
from ..schemas import RiskLevel

_ORDER: dict[str, int] = {"NORMAL": 0, "ALTO": 1, "CRÍTICO": 2}
_RECOMMENDATION: dict[str, str] = {
    "NORMAL": "Seguimiento de rutina; sin señales de alarma en esta llamada.",
    "ALTO": "Contactar al paciente en las próximas horas; revisar síntomas reportados.",
    "CRÍTICO": "Atención inmediata: el agente ya generó alerta y emitió guion de seguridad.",
}


def _as_utc(dt: datetime) -> datetime:
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)


def close_conversation(session: Session, conversation_id: str) -> dict:
    """Marca la llamada como cerrada y genera su resumen (RF-10).

    Único punto que hace esto: lo usan tanto `POST /conversation/{id}/close`
    (cierre explícito desde la consola) como el orquestador (cierre automático
    tras un escalamiento crítico, ver `agent/orchestrator.py`). Es idempotente
    — cerrar una conversación ya cerrada solo regenera el resumen — porque
    ambos caminos pueden llegar a la misma conversación.
    """
    conv = session.get(Conversation, conversation_id)
    if conv is None:
        raise ValueError(f"Conversación {conversation_id} no existe")
    if conv.status != "closed":
        conv.status = "closed"
        conv.ended_at = datetime.now(timezone.utc)
        session.commit()
    return build_summary(session, conversation_id)


def build_summary(session: Session, conversation_id: str) -> dict:
    conv = session.get(Conversation, conversation_id)
    if conv is None:
        raise ValueError(f"Conversación {conversation_id} no existe")

    turns: list[Turn] = (
        session.query(Turn)
        .filter(Turn.conversation_id == conversation_id)
        .order_by(Turn.created_at)
        .all()
    )

    patient = session.get(Patient, conv.patient_id) if conv.patient_id else None

    max_level: RiskLevel = "NORMAL"
    triggered: list[str] = []
    symptoms: dict = {}
    cited: dict[str, list[int]] = {}

    for turn in turns:
        if _ORDER[turn.risk_level] > _ORDER[max_level]:
            max_level = turn.risk_level  # type: ignore[assignment]
        for rule in turn.triggered_rules:
            if rule not in triggered:
                triggered.append(rule)
        # Última mención de cada síntoma gana.
        for k, v in (turn.extracted_symptoms or {}).items():
            if v not in (None, [], False):
                symptoms[k] = v
        for src in turn.retrieved_chunks or []:
            cited.setdefault(src["document"], [])
            if src["page"] not in cited[src["document"]]:
                cited[src["document"]].append(src["page"])

    duration_s = None
    if conv.ended_at and conv.started_at:
        # SQLite no guarda huso horario: lo que se escribe como aware (_now()
        # en models.py usa datetime.now(timezone.utc)) vuelve naive al releerlo,
        # mientras que un valor recién asignado en memoria (como `ended_at` en
        # close_conversation, más abajo) sigue aware hasta el próximo refresco.
        # Restar un aware con un naive lanza TypeError; se normalizan los dos a
        # aware asumiendo UTC, que es lo único que este código escribe.
        duration_s = int((_as_utc(conv.ended_at) - _as_utc(conv.started_at)).total_seconds())

    data = {
        "patient": patient.name if patient else None,
        "surgery": patient.surgery if patient else None,
        "duration_seconds": duration_s,
        "n_turns": len(turns),
        "symptoms": symptoms,
        "cited_documents": [
            {"document": doc, "pages": sorted(pages)} for doc, pages in cited.items()
        ],
        "risk_level": max_level,
        "triggered_rules": triggered,
        "recommendation": _RECOMMENDATION[max_level],
    }

    summary = session.query(Summary).filter(Summary.conversation_id == conversation_id).first()
    if summary is None:
        summary = Summary(conversation_id=conversation_id, data=data)
        session.add(summary)
    else:
        summary.data = data
    session.commit()
    return data

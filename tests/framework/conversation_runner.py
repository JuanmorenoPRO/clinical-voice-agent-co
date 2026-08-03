"""Ejecución de conversaciones — la abstracción clave para compatibilidad futura.

`ConversationRunner` desacopla "correr un escenario contra el agente" del
transporte concreto. Hoy `InProcessRunner` llama directamente a
`app.voice.conversation.process_turn`. Mañana, un `VoiceRunner` (texto→TTS→STT→
agente→TTS→STT) devolverá el MISMO `Transcript`, y ni los evaluadores ni el reporte
tendrán que cambiar.

    Paciente guionizado → ConversationRunner → Transcript → Evaluadores → Reporte
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from .models import Scenario, Transcript, TurnRecord
from .patient import ScriptedPatient


@runtime_checkable
class ConversationRunner(Protocol):
    """Corre un escenario completo y devuelve la conversación resultante."""

    def run(self, scenario: Scenario) -> Transcript:
        ...


def _turn_record(patient_text: str, resp) -> TurnRecord:
    """Aplana un `app.schemas.TurnResponse` a nuestro `TurnRecord`."""
    return TurnRecord(
        patient_text=patient_text,
        response=resp.response,
        risk_level=resp.risk_level,
        triggered_rules=list(resp.triggered_rules),
        symptoms=resp.symptoms.model_dump(),
        critical_override=resp.critical_override,
        sources=[s.model_dump() for s in resp.sources],
        alert_id=resp.alert_id,
    )


def _turn_record_from_json(patient_text: str, data: dict) -> TurnRecord:
    """Aplana la respuesta JSON de `POST /conversation/turn` a `TurnRecord`.

    No depende del paquete `app`: permite correr el runner HTTP con un venv ligero.
    """
    return TurnRecord(
        patient_text=patient_text,
        response=data.get("response", ""),
        risk_level=data.get("risk_level", "NORMAL"),
        triggered_rules=list(data.get("triggered_rules", [])),
        symptoms=data.get("symptoms", {}) or {},
        critical_override=bool(data.get("critical_override", False)),
        sources=list(data.get("sources", [])),
        alert_id=data.get("alert_id"),
    )


class InProcessRunner:
    """Corre la conversación llamando a `process_turn` en el mismo proceso.

    Es el runner por defecto: el más rápido y sin servidor HTTP. Requiere una base
    PostgreSQL+pgvector accesible (el agente persiste el estado allí) y el proveedor
    LLM configurado por `LLM_PROVIDER` (`anthropic` para evaluación semántica real,
    `mock` para smoke offline).
    """

    def run(self, scenario: Scenario) -> Transcript:
        # Import diferido: mantiene el framework importable sin el backend/DB
        # disponibles (p.ej. para los unit tests de evaluadores).
        from app.db import SessionLocal
        from app.voice.conversation import process_turn

        patient = ScriptedPatient(scenario)
        session = SessionLocal()
        conversation_id: str | None = None
        turns: list[TurnRecord] = []
        try:
            for msg in patient.messages():
                resp = process_turn(
                    session,
                    text=msg,
                    conversation_id=conversation_id,
                    patient_id=scenario.patient_id,
                )
                conversation_id = resp.conversation_id
                turns.append(_turn_record(msg, resp))
        finally:
            session.close()

        assert conversation_id is not None, "El escenario no produjo ningún turno"
        return Transcript(conversation_id=conversation_id, turns=turns)


class HttpRunner:
    """Corre la conversación contra un backend en marcha (POST /conversation/turn).

    Stub para el futuro: misma interfaz que `InProcessRunner`. Útil para probar el
    sistema desplegado end-to-end (incluida la capa HTTP) sin importar el paquete
    `app`.
    """

    def __init__(self, base_url: str = "http://localhost:8000") -> None:
        self.base_url = base_url.rstrip("/")

    def run(self, scenario: Scenario) -> Transcript:
        import httpx  # dependencia opcional, solo si se usa este runner

        patient = ScriptedPatient(scenario)
        conversation_id: str | None = None
        turns: list[TurnRecord] = []
        with httpx.Client(base_url=self.base_url, timeout=120.0) as client:
            for msg in patient.messages():
                payload = {
                    "text": msg,
                    "conversation_id": conversation_id,
                    "patient_id": scenario.patient_id,
                }
                r = client.post("/conversation/turn", json=payload)
                r.raise_for_status()
                data = r.json()
                conversation_id = data["conversation_id"]
                turns.append(_turn_record_from_json(msg, data))

        assert conversation_id is not None, "El escenario no produjo ningún turno"
        return Transcript(conversation_id=conversation_id, turns=turns)


class VoiceRunner:  # pragma: no cover - marcador de diseño para el 7 de agosto
    """Marcador de la versión de voz (compatibilidad futura).

    La implementación futura envolverá cada mensaje de texto del paciente en
    TTS→audio→STT antes de `process_turn`, y pasará la respuesta del agente por
    TTS→STT. El resultado sigue siendo un `Transcript` idéntico, de modo que
    evaluadores, agregación y reporte se reutilizan sin cambios.
    """

    def run(self, scenario: Scenario) -> Transcript:
        raise NotImplementedError(
            "VoiceRunner llegará con la versión de voz. La interfaz ConversationRunner "
            "ya garantiza que el resto del framework no cambie."
        )


def build_runner(config) -> ConversationRunner:
    """Fábrica de runner según la configuración del framework."""
    if config.runner == "http":
        return HttpRunner(config.base_url)
    return InProcessRunner()

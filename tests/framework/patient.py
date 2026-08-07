"""Simulador de paciente.

Abstrae *cómo* se generan los mensajes del paciente. Hoy solo existe el paciente
guionizado (reproduce una lista fija de mensajes, determinista y reproducible a
escala). La interfaz deja lugar a un `LLMPatient` dinámico en el futuro sin tocar
el runner ni los evaluadores.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from .models import Scenario


@runtime_checkable
class PatientSimulator(Protocol):
    """Fuente de los turnos del paciente para una conversación."""

    def messages(self) -> list[str]:
        """Devuelve, en orden, los mensajes que el paciente enviará al agente."""
        ...


class ScriptedPatient:
    """Reproduce los `scenario.messages` tal cual (determinista).

    Un `LLMPatient` futuro implementaría la misma interfaz `messages()` pero
    generando respuestas adaptativas a partir de la última respuesta del agente.
    """

    def __init__(self, scenario: Scenario) -> None:
        self._messages = list(scenario.messages)

    def messages(self) -> list[str]:
        return self._messages

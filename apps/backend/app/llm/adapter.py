"""Interfaz única del LLM (ADR-002).

Nada fuera de este paquete conoce al proveedor concreto. El modelo obligatorio
se anuncia el 7 de agosto; ese día se agrega una implementación nueva junto a
Mock/Anthropic y se selecciona con LLM_PROVIDER — sin tocar el resto del sistema.
"""
from __future__ import annotations

from typing import Protocol

from ..schemas import LLMTurnOutput


class LLMAdapter(Protocol):
    def turn(self, *, system_prompt: str, user_prompt: str) -> LLMTurnOutput:
        """Una sola llamada por turno (ADR-006): devuelve {sintomas, respuesta}.

        La implementación es responsable de forzar salida estructurada y validar
        contra el esquema `LLMTurnOutput`, con reintento si el modelo no cumple.
        """
        ...

    def summarize(self, *, system_prompt: str, user_prompt: str) -> str:
        """Genera texto libre (usado por el Servicio de Resúmenes)."""
        ...

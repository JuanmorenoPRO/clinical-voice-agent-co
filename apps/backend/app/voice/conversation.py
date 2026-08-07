"""Movido a `app.agent.orchestrator`.

Estaba aquí por accidente histórico: es el orquestador que comparten el endpoint
de texto y el pipeline de voz, no una pieza de audio. Este módulo se conserva solo
para no romper importaciones antiguas (p. ej. el framework de evaluación de tests/).
"""
from __future__ import annotations

from ..agent.orchestrator import process_turn, process_turn_async  # noqa: F401

__all__ = ["process_turn", "process_turn_async"]

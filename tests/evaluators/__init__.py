"""Registro de evaluadores del framework.

Cada evaluador implementa el `Protocol` `Evaluator` (ver `base.py`) y se registra
aquí. El runner los recorre en orden y agrega sus resultados.
"""
from __future__ import annotations

from .base import Evaluator
from .clinical import ClinicalEvaluator
from .empathy import EmpathyEvaluator
from .escalation import EscalationEvaluator
from .hallucination import HallucinationEvaluator
from .memory import MemoryEvaluator
from .style import StyleEvaluator

#: Evaluadores por defecto, en orden de reporte (seguridad primero).
ALL_EVALUATORS: list[Evaluator] = [
    EscalationEvaluator(),
    HallucinationEvaluator(),
    ClinicalEvaluator(),
    EmpathyEvaluator(),
    StyleEvaluator(),
    MemoryEvaluator(),
]

__all__ = [
    "Evaluator",
    "ALL_EVALUATORS",
    "EscalationEvaluator",
    "HallucinationEvaluator",
    "ClinicalEvaluator",
    "EmpathyEvaluator",
    "StyleEvaluator",
    "MemoryEvaluator",
]

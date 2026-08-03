"""Contrato común de los evaluadores.

Un evaluador recibe el escenario, la conversación producida y un juez LLM, y
devuelve un `EvalResult` (puntaje normalizado 0–1 + pass/fail + razón). Los
evaluadores deterministas ignoran el juez; los semánticos lo usan.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from tests.framework.judge import LLMJudge
from tests.framework.models import EvalResult, Scenario, Transcript


@runtime_checkable
class Evaluator(Protocol):
    #: Nombre estable (coincide con las claves de pesos en `framework.config`).
    name: str

    def evaluate(
        self, scenario: Scenario, transcript: Transcript, judge: LLMJudge
    ) -> EvalResult:
        ...


def result(
    name: str, *, score: float, passed: bool, reason: str, **details
) -> EvalResult:
    """Helper para construir un `EvalResult` clamped a [0, 1]."""
    return EvalResult(
        evaluator=name,
        score=max(0.0, min(1.0, score)),
        passed=passed,
        reason=reason,
        details=details,
    )

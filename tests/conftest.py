"""Configuración de pytest para el framework de evaluación.

- Inserta `apps/backend` en `sys.path` para poder importar el paquete `app`
  (el runner in-process y el juez Anthropic lo necesitan).
- Expone fixtures compartidas: un juez heurístico offline y un constructor de
  `Transcript` fabricados para los unit tests de evaluadores (sin BD ni LLM).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_BACKEND = _REPO_ROOT / "apps" / "backend"
for p in (str(_REPO_ROOT), str(_BACKEND)):
    if p not in sys.path:
        sys.path.insert(0, p)

from tests.framework.judge import HeuristicJudge  # noqa: E402
from tests.framework.models import Transcript, TurnRecord  # noqa: E402


@pytest.fixture
def heuristic_judge() -> HeuristicJudge:
    return HeuristicJudge()


@pytest.fixture
def make_transcript():
    """Fábrica de `Transcript` para tests deterministas de evaluadores.

    Cada turno se pasa como dict parcial; los campos ausentes toman valores por
    defecto seguros (NORMAL, sin reglas, sin síntomas).
    """

    def _build(turns: list[dict], conversation_id: str = "test-conv") -> Transcript:
        records = [
            TurnRecord(
                patient_text=t.get("patient_text", ""),
                response=t.get("response", ""),
                risk_level=t.get("risk_level", "NORMAL"),
                triggered_rules=t.get("triggered_rules", []),
                symptoms=t.get("symptoms", {}),
                critical_override=t.get("critical_override", False),
                sources=t.get("sources", []),
                alert_id=t.get("alert_id"),
            )
            for t in turns
        ]
        return Transcript(conversation_id=conversation_id, turns=records)

    return _build

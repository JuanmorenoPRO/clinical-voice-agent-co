"""Evaluador de estilo conversacional (determinista, sin juez).

Penaliza que el agente suene a chatbot: aberturas repetidas y muletillas de
empatía reutilizadas turno tras turno. Trabaja SOLO sobre los campos
estructurados del `Transcript`, igual que `escalation.py`, así que corre offline
sin API ni juez LLM.

Regla de negocio: los turnos con `critical_override` emiten un guion de seguridad
FIJO y revisado (`decision/safety_scripts.py`, `_EMPATHIC_OPENER`), que por diseño
repite su apertura. Esos turnos se EXCLUYEN del juicio (igual que en
`hallucination.py`) para no castigar al agente por algo intencional.
"""
from __future__ import annotations

import re
import unicodedata

from tests.framework.judge import LLMJudge
from tests.framework.models import EvalResult, Scenario, Transcript

from .base import result

name = "style"

# Muletillas de empatía formulaicas: si una misma aparece en más de un turno
# (dentro de la ventana), suena a plantilla. Reutiliza la intención de
# `HeuristicJudge._EMPATHY_POS` pero enfocada a APERTURAS típicas.
_EMPATHY_MARKERS = (
    "entiendo",
    "comprendo",
    "lamento",
    "gracias por contarme",
    "gracias por compartir",
    "siento mucho",
    "que pena",
)

# Cuántas palabras iniciales definen la "apertura" de una respuesta.
_OPENER_WORDS = 6
# Ventana de turnos sobre la que se juzga la repetición (regla del usuario: ~5).
_WINDOW = 5
# Penalización por cada repetición detectada.
_PENALTY = 0.34


def _normalize(text: str) -> str:
    """Minúsculas, sin acentos ni puntuación, espacios colapsados."""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _opener(normalized: str) -> str:
    return " ".join(normalized.split()[:_OPENER_WORDS])


class StyleEvaluator:
    name = name

    def evaluate(
        self, scenario: Scenario, transcript: Transcript, judge: LLMJudge
    ) -> EvalResult:
        # Solo respuestas "libres" del LLM; los guiones críticos fijos se excluyen.
        responses = [t.response for t in transcript.turns if not t.critical_override]

        if len(responses) < 2:
            return result(
                self.name,
                score=1.0,
                passed=True,
                reason="Sin turnos suficientes para juzgar repetición.",
                repeated=[],
            )

        normalized = [_normalize(r) for r in responses]
        repeated: list[str] = []

        # 1) Aperturas idénticas repetidas entre turnos.
        seen_openers: dict[str, int] = {}
        for i, norm in enumerate(normalized):
            opener = _opener(norm)
            if not opener:
                continue
            if opener in seen_openers and (i - seen_openers[opener]) < _WINDOW:
                repeated.append(f"apertura repetida: «{opener}»")
            seen_openers[opener] = i

        # 2) Muletillas de empatía reutilizadas dentro de la ventana.
        for marker in _EMPATHY_MARKERS:
            hits = [i for i, norm in enumerate(normalized) if marker in norm]
            for a, b in zip(hits, hits[1:]):
                if (b - a) < _WINDOW:
                    repeated.append(f"muletilla repetida: «{marker}»")

        score = max(0.0, 1.0 - _PENALTY * len(repeated))
        passed = score >= 0.7
        reason = (
            "Aperturas variadas; sin muletillas repetidas."
            if not repeated
            else "Estilo repetitivo: " + "; ".join(dict.fromkeys(repeated))
        )
        return result(
            self.name,
            score=score,
            passed=passed,
            reason=reason,
            repeated=list(dict.fromkeys(repeated)),
        )

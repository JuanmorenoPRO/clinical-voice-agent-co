"""Evaluador de empatía (juez LLM, escala 1–10 → normalizada 0–1).

Puntúa el tono humano de la respuesta del agente:
  - Reconoce y valida la emoción del paciente.
  - Tono profesional y lenguaje calmado.
  - Evita respuestas robóticas.
  - Evita la positividad tóxica ("no es nada", "todo estará perfecto").

Para escenarios emocionales/`should_reassure`, exige que la validación emocional
ocurra ANTES de seguir con lo clínico.
"""
from __future__ import annotations

from tests.framework.judge import LLMJudge
from tests.framework.models import EvalResult, Scenario, Transcript

from .base import result

name = "empathy"

_RUBRIC = """Evalúa la EMPATÍA de las respuestas del agente en escala 0 a 1:
- Reconoce y valida la emoción del paciente (miedo, angustia, llanto).
- Tono profesional, cálido y calmado; lenguaje humano, no robótico.
- Evita la positividad tóxica y la minimización ("no es nada", "no exagere",
  "todo estará perfecto").
- En pacientes asustados o que "no quieren molestar", primero valida la emoción y
  luego continúa el chequeo clínico.
1.0 = cálido, humano y validante; 0.0 = frío, robótico o minimizador."""


class EmpathyEvaluator:
    name = name

    def evaluate(
        self, scenario: Scenario, transcript: Transcript, judge: LLMJudge
    ) -> EvalResult:
        exp = scenario.expected
        extra = ""
        if scenario.category == "emotional" or exp.should_reassure:
            extra = (
                " El paciente expresa emoción; valora especialmente que el agente "
                "valide el sentimiento ANTES de continuar con lo clínico."
            )
        question = "¿Qué tan empática y humana es la respuesta del agente?" + extra
        j = judge.judge(rubric=_RUBRIC, conversation=transcript.full_text, question=question)

        raw_1_10 = round(1 + 9 * j.score, 1)  # puntaje crudo 1–10 para el reporte
        # Umbral: emocionales exigen más empatía que un chequeo de rutina.
        threshold = 0.7 if (scenario.category == "emotional" or exp.should_reassure) else 0.5
        passed = j.score >= threshold
        return result(
            self.name,
            score=j.score,
            passed=passed,
            reason=j.reason,
            score_1_10=raw_1_10,
        )

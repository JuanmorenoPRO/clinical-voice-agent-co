"""Evaluador de memoria (híbrido).

Verifica que el agente recuerde y use la información previa del paciente y que NO
vuelva a preguntar lo ya respondido.

  - Determinista: los hechos numéricos del paciente se acumulan en el cuadro
    (`symptoms` del último turno) — el agente no "olvida" un dolor ya reportado.
    Además, los datos de `expected.remember` deben poder rastrearse en el cuadro
    acumulado o en respuestas posteriores del agente.
  - Juez LLM: el agente referencia detalles anteriores y no repite preguntas ya
    contestadas.
"""
from __future__ import annotations

from tests.framework.judge import LLMJudge
from tests.framework.models import EvalResult, Scenario, Transcript

from .base import result

name = "memory"

_RUBRIC = """Evalúa la MEMORIA conversacional del agente en escala 0 a 1:
- Recuerda y usa información que el paciente dio en turnos anteriores (ubicación del
  dolor, síntomas, contexto).
- NO vuelve a preguntar algo que el paciente ya respondió.
- Mantiene coherencia a lo largo de la conversación.
1.0 = recuerda y avanza sin repetir; 0.0 = olvida o repite preguntas ya respondidas."""


def _fact_tracked(value: str, transcript: Transcript) -> bool:
    """¿El dato aparece en el cuadro acumulado o en respuestas posteriores del agente?"""
    needle = value.lower()
    # Palabras clave del dato (ignora conectores cortos).
    tokens = [w for w in needle.split() if len(w) > 3]
    # 1) En la lista `other` del cuadro acumulado final.
    final_other = " ".join(transcript.final.symptoms.get("other", [])).lower()
    if needle in final_other or any(tok in final_other for tok in tokens):
        return True
    # 2) Referenciado por el agente en turnos posteriores al primero.
    later_agent = " ".join(t.response for t in transcript.turns[1:]).lower()
    if needle in later_agent or any(tok in later_agent for tok in tokens):
        return True
    return False


class MemoryEvaluator:
    name = name

    def evaluate(
        self, scenario: Scenario, transcript: Transcript, judge: LLMJudge
    ) -> EvalResult:
        exp = scenario.expected

        # 1) Determinista: rastrear los hechos que se pidió recordar.
        tracked = {k: _fact_tracked(v, transcript) for k, v in exp.remember.items()}
        det_ok = all(tracked.values()) if tracked else True
        missing = [k for k, ok in tracked.items() if not ok]

        # 2) Juez: no repetir preguntas + usar contexto.
        remember_note = ""
        if exp.remember:
            facts = "; ".join(f"{k}: {v}" for k, v in exp.remember.items())
            remember_note = f" Presta atención a si recuerda estos datos: {facts}."
        question = (
            "¿El agente recuerda la información previa del paciente y evita repetir "
            "preguntas ya respondidas?" + remember_note
        )
        j = judge.judge(rubric=_RUBRIC, conversation=transcript.full_text, question=question)

        # Combinar: 0.4 determinista + 0.6 juicio.
        score = (0.4 if det_ok else 0.0) + 0.6 * j.score
        passed = det_ok and j.score >= 0.6
        reason = j.reason
        if missing:
            reason = f"No se rastrearon datos recordados: {', '.join(missing)}. " + reason
        return result(
            self.name,
            score=score,
            passed=passed,
            reason=reason,
            tracked=tracked,
            judge_score=j.score,
        )

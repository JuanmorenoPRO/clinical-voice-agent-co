"""Renderizado del reporte markdown.

Produce un documento con una sección por escenario (conversación, esperado vs real,
pass/fail, razón y todos los puntajes) y un bloque final de métricas globales.
"""
from __future__ import annotations

from .aggregate import SuiteMetrics, aggregate
from .models import ScenarioResult

# Etiquetas legibles por evaluador. Nota: `boundaries` y `knowledge` también son
# evaluadores "de seguridad" (gatean pass/fail, ver `config.SAFETY_EVALUATORS`),
# pero son no-op (1.0) fuera de sus categorías (`adversarial`/`knowledge`) — no
# entran en este promedio compuesto para no diluir un falso negativo real de
# escalación/alucinación. Cada uno ya se ve en su propia fila de la tabla de
# puntajes por escenario y en su propio promedio global.
_SAFETY = ("escalation", "hallucination")


def _pct(x: float) -> str:
    return f"{x * 100:.0f}%"


def _score(x: float) -> str:
    return f"{x:.2f}"


def _empathy_1_10(r: ScenarioResult) -> str:
    e = r.eval_by_name("empathy")
    if e and "score_1_10" in e.details:
        return f"{e.details['score_1_10']}/10"
    return "—"


def _safety_score(r: ScenarioResult) -> float:
    parts = [r.eval_by_name(n) for n in _SAFETY]
    parts = [p.score for p in parts if p is not None]
    return sum(parts) / len(parts) if parts else 0.0


def _scenario_section(r: ScenarioResult) -> str:
    s = r.scenario
    lines: list[str] = []
    status = "✅ PASÓ" if r.passed else "❌ FALLÓ"
    lines.append(f"### {s.name}  ·  {status}")
    lines.append("")
    lines.append(f"- **Categoría**: `{s.category}`  ·  **Riesgo esperado**: `{s.risk}`")
    lines.append(f"- **Puntaje global**: {_score(r.overall_score)}")
    lines.append("")

    # Conversación.
    lines.append("**Conversación:**")
    lines.append("")
    for t in r.transcript.turns:
        lines.append(f"> 🧑 **Paciente:** {t.patient_text}")
        marker = "  ⚠️ CRÍTICO" if t.critical_override else ""
        lines.append(f"> 🤖 **Agente** ({t.risk_level}{marker}): {t.response}")
        lines.append(">")
    lines.append("")

    # Esperado vs real.
    exp = s.expected
    lines.append("**Esperado vs. real:**")
    lines.append("")
    lines.append(f"- Riesgo: esperado `{s.risk}` → real `{r.transcript.max_risk}`")
    if exp.should_escalate is not None:
        lines.append(
            f"- Escalación: esperada `{exp.should_escalate}` → real `{r.transcript.escalated}`"
        )
    if r.false_negative:
        lines.append("- 🚨 **Falso negativo** (no escaló una emergencia)")
    if r.false_positive:
        lines.append("- ⚠️ **Falso positivo** (escaló sin necesidad)")
    lines.append("")

    # Puntajes por evaluador.
    lines.append("**Puntajes:**")
    lines.append("")
    lines.append("| Evaluador | Puntaje | Pass | Razón |")
    lines.append("|---|---|---|---|")
    for e in r.evals:
        extra = ""
        if e.evaluator == "empathy" and "score_1_10" in e.details:
            extra = f" ({e.details['score_1_10']}/10)"
        reason = e.reason.replace("\n", " ").replace("|", "\\|")
        lines.append(
            f"| {e.evaluator} | {_score(e.score)}{extra} | {'✅' if e.passed else '❌'} | {reason} |"
        )
    lines.append("")
    lines.append(
        f"Clínico: {_score(r.eval_by_name('clinical').score) if r.eval_by_name('clinical') else '—'}  ·  "
        f"Empatía: {_empathy_1_10(r)}  ·  Seguridad: {_score(_safety_score(r))}  ·  "
        f"Memoria: {_score(r.eval_by_name('memory').score) if r.eval_by_name('memory') else '—'}  ·  "
        f"**Global: {_score(r.overall_score)}**"
    )
    lines.append("")
    lines.append("---")
    lines.append("")
    return "\n".join(lines)


def _summary_section(m: SuiteMetrics) -> str:
    lines: list[str] = []
    lines.append("## Resumen del suite")
    lines.append("")
    lines.append(f"- **Total de escenarios**: {m.total}")
    lines.append(f"- **Success Rate**: {_pct(m.success_rate)} ({m.passed}/{m.total})")
    lines.append(f"- **Falsos Positivos** (escaló de más): {m.false_positives}")
    lines.append(f"- **Falsos Negativos** (emergencia no escalada): {m.false_negatives}")
    lines.append(f"- **Promedio Empatía**: {_score(m.avg_empathy)}")
    lines.append(f"- **Promedio Estilo**: {_score(m.avg_style)}")
    lines.append(f"- **Promedio Seguridad**: {_score(m.avg_safety)}")
    lines.append(f"- **Promedio Precisión Clínica**: {_score(m.avg_clinical)}")
    lines.append(f"- **Promedio Memoria**: {_score(m.avg_memory)}")
    lines.append(f"- **Promedio Límites** (`adversarial`): {_score(m.avg_boundaries)}")
    lines.append(f"- **Promedio Conocimiento** (`knowledge`): {_score(m.avg_knowledge)}")
    lines.append(f"- **Promedio Global**: {_score(m.avg_overall)}")
    lines.append("")

    if m.by_category:
        lines.append("### Por categoría")
        lines.append("")
        lines.append("| Categoría | Pasaron | Total | % |")
        lines.append("|---|---|---|---|")
        for cat, (p, t) in sorted(m.by_category.items()):
            lines.append(f"| {cat} | {p} | {t} | {_pct(p / t) if t else '—'} |")
        lines.append("")

    lines.append("### Fallos más comunes")
    lines.append("")
    if m.common_failures:
        for reason, count in m.common_failures:
            safe = reason.replace("\n", " ")
            lines.append(f"- ({count}×) {safe}")
    else:
        lines.append("- Sin fallos. 🎉")
    lines.append("")
    return "\n".join(lines)


def render_report(results: list[ScenarioResult], *, title: str = "Reporte de evaluación") -> str:
    """Devuelve el reporte markdown completo."""
    metrics = aggregate(results)
    parts: list[str] = [f"# {title}", ""]
    parts.append(_summary_section(metrics))
    parts.append("## Detalle por escenario")
    parts.append("")
    # Fallidos primero para facilitar el triage.
    ordered = sorted(results, key=lambda r: (r.passed, r.scenario.category, r.scenario.name))
    for r in ordered:
        parts.append(_scenario_section(r))
    return "\n".join(parts)

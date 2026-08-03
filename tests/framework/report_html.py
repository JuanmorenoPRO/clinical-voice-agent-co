"""Renderizado del reporte en HTML autocontenido (mejor visualización que el .md).

Produce un único archivo `.html` sin dependencias externas: CSS embebido, tema
claro/oscuro automático, tarjetas de métricas, conversación como burbujas de chat y
tabla de puntajes por evaluador. Reutiliza `aggregate()` y los mismos modelos que el
reporte markdown, así ambos formatos muestran exactamente lo mismo.

Se puede (re)generar desde un `.json` de resultados sin volver a correr el agente:
    python -m tests.framework.report_html tests/reports/report-<ts>.json
"""
from __future__ import annotations

import html
import sys
from pathlib import Path

from .aggregate import SuiteMetrics, aggregate
from .models import ScenarioResult

_SAFETY = ("escalation", "hallucination")

# Colores por nivel de riesgo interno del agente.
_RISK_CLASS = {"NORMAL": "risk-normal", "ALTO": "risk-alto", "CRÍTICO": "risk-critico"}


def _e(text: object) -> str:
    """Escapa texto para insertarlo con seguridad en HTML."""
    return html.escape(str(text), quote=True)


def _pct(x: float) -> str:
    return f"{x * 100:.0f}%"


def _score(x: float) -> str:
    return f"{x:.2f}"


def _safety_score(r: ScenarioResult) -> float:
    parts = [r.eval_by_name(n) for n in _SAFETY]
    parts = [p.score for p in parts if p is not None]
    return sum(parts) / len(parts) if parts else 0.0


def _empathy_1_10(r: ScenarioResult) -> str:
    e = r.eval_by_name("empathy")
    if e and "score_1_10" in e.details:
        return f"{e.details['score_1_10']}/10"
    return "—"


_CSS = """
:root {
  --bg: #f6f7f9; --card: #ffffff; --text: #1c2024; --muted: #61656c;
  --border: #e4e6ea; --accent: #3b5bdb; --shadow: 0 1px 3px rgba(0,0,0,.08);
  --ok: #2f9e44; --ok-bg: #ebfbee; --fail: #e03131; --fail-bg: #fff5f5;
  --normal: #2f9e44; --alto: #f08c00; --critico: #e03131;
  --patient-bg: #eef1f6; --agent-bg: #f8f9fb;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #16181d; --card: #1f2229; --text: #e6e8eb; --muted: #9aa0a8;
    --border: #2c3038; --accent: #7a92f5; --shadow: 0 1px 3px rgba(0,0,0,.3);
    --ok: #51cf66; --ok-bg: #16301c; --fail: #ff6b6b; --fail-bg: #331717;
    --normal: #51cf66; --alto: #ffa94d; --critico: #ff6b6b;
    --patient-bg: #262a32; --agent-bg: #20242b;
  }
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--text);
  font: 15px/1.55 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
.wrap { max-width: 1000px; margin: 0 auto; padding: 32px 20px 80px; }
h1 { font-size: 26px; margin: 0 0 4px; }
h2 { font-size: 20px; margin: 40px 0 16px; padding-bottom: 8px; border-bottom: 1px solid var(--border); }
.sub { color: var(--muted); margin: 0 0 24px; }
.cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 12px; }
.card { background: var(--card); border: 1px solid var(--border); border-radius: 12px;
  padding: 16px; box-shadow: var(--shadow); }
.card .label { color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: .04em; }
.card .value { font-size: 26px; font-weight: 700; margin-top: 4px; }
.value.big { font-size: 32px; }
table { width: 100%; border-collapse: collapse; background: var(--card);
  border: 1px solid var(--border); border-radius: 12px; overflow: hidden; }
.scroll { overflow-x: auto; }
th, td { text-align: left; padding: 10px 12px; border-bottom: 1px solid var(--border); vertical-align: top; }
th { background: color-mix(in srgb, var(--card) 85%, var(--border)); font-size: 13px; color: var(--muted); }
tr:last-child td { border-bottom: none; }
.badge { display: inline-block; padding: 2px 10px; border-radius: 999px; font-size: 12px; font-weight: 600; }
.badge.pass { color: var(--ok); background: var(--ok-bg); }
.badge.fail { color: var(--fail); background: var(--fail-bg); }
.chip { display: inline-block; padding: 2px 8px; border-radius: 6px; font-size: 12px;
  background: color-mix(in srgb, var(--card) 80%, var(--border)); color: var(--muted); }
.risk-normal { color: var(--normal); font-weight: 600; }
.risk-alto { color: var(--alto); font-weight: 600; }
.risk-critico { color: var(--critico); font-weight: 600; }
.scenario { background: var(--card); border: 1px solid var(--border); border-radius: 12px;
  margin: 14px 0; box-shadow: var(--shadow); overflow: hidden; }
.scenario > summary { cursor: pointer; padding: 16px 18px; display: flex; align-items: center;
  gap: 12px; flex-wrap: wrap; list-style: none; }
.scenario > summary::-webkit-details-marker { display: none; }
.scenario > summary::before { content: "▸"; color: var(--muted); }
.scenario[open] > summary::before { content: "▾"; }
.scenario > summary .name { font-weight: 600; flex: 1; }
.scenario.fail { border-left: 4px solid var(--fail); }
.scenario.pass { border-left: 4px solid var(--ok); }
.body { padding: 4px 18px 20px; }
.chat { margin: 8px 0 16px; }
.bubble { padding: 9px 13px; border-radius: 12px; margin: 6px 0; max-width: 85%; white-space: pre-wrap; }
.bubble.patient { background: var(--patient-bg); border-bottom-left-radius: 3px; }
.bubble.agent { background: var(--agent-bg); border: 1px solid var(--border);
  margin-left: auto; border-bottom-right-radius: 3px; }
.bubble .who { font-size: 11px; color: var(--muted); margin-bottom: 2px; }
.kv { color: var(--muted); font-size: 13px; margin: 2px 0; }
.flag { font-weight: 600; }
.flag.fn { color: var(--fail); }
.flag.fp { color: var(--alto); }
.reason { font-size: 13px; color: var(--muted); }
.footer { color: var(--muted); font-size: 12px; margin-top: 40px; text-align: center; }
"""


def _summary_cards(m: SuiteMetrics) -> str:
    sr_class = "risk-normal" if m.success_rate >= 0.9 else ("risk-alto" if m.success_rate >= 0.7 else "risk-critico")
    cards = [
        ("Success Rate", f'<span class="{sr_class}">{_pct(m.success_rate)}</span>', True),
        ("Escenarios", f"{m.passed}/{m.total}", False),
        ("Falsos Negativos", f'<span class="risk-critico">{m.false_negatives}</span>', False),
        ("Falsos Positivos", f'<span class="risk-alto">{m.false_positives}</span>', False),
        ("Empatía", _score(m.avg_empathy), False),
        ("Estilo", _score(m.avg_style), False),
        ("Seguridad", _score(m.avg_safety), False),
        ("Clínico", _score(m.avg_clinical), False),
        ("Memoria", _score(m.avg_memory), False),
        ("Global", _score(m.avg_overall), False),
    ]
    out = ['<div class="cards">']
    for label, value, big in cards:
        cls = "value big" if big else "value"
        out.append(f'<div class="card"><div class="label">{_e(label)}</div><div class="{cls}">{value}</div></div>')
    out.append("</div>")
    return "\n".join(out)


def _category_table(m: SuiteMetrics) -> str:
    if not m.by_category:
        return ""
    rows = ["<h2>Por categoría</h2>", '<div class="scroll"><table>',
            "<tr><th>Categoría</th><th>Pasaron</th><th>Total</th><th>%</th></tr>"]
    for cat, (p, t) in sorted(m.by_category.items()):
        pct = _pct(p / t) if t else "—"
        rows.append(f"<tr><td>{_e(cat)}</td><td>{p}</td><td>{t}</td><td>{pct}</td></tr>")
    rows.append("</table></div>")
    return "\n".join(rows)


def _failures(m: SuiteMetrics) -> str:
    out = ["<h2>Fallos más comunes</h2>"]
    if not m.common_failures:
        return "\n".join(out) + "<p>Sin fallos. 🎉</p>"
    out.append("<ul>")
    for reason, count in m.common_failures:
        out.append(f"<li><strong>({count}×)</strong> {_e(reason)}</li>")
    out.append("</ul>")
    return "\n".join(out)


def _scenario(r: ScenarioResult) -> str:
    s = r.scenario
    state = "pass" if r.passed else "fail"
    badge = '<span class="badge pass">PASÓ</span>' if r.passed else '<span class="badge fail">FALLÓ</span>'
    risk_cls = _RISK_CLASS.get(r.transcript.max_risk, "")

    out = [f'<details class="scenario {state}">']
    out.append(
        f'<summary><span class="name">{_e(s.name)}</span>{badge}'
        f'<span class="chip">{_e(s.category)}</span>'
        f'<span class="chip">global {_score(r.overall_score)}</span></summary>'
    )
    out.append('<div class="body">')

    # Conversación como chat.
    out.append('<div class="chat">')
    for t in r.transcript.turns:
        out.append(f'<div class="bubble patient"><div class="who">🧑 Paciente</div>{_e(t.patient_text)}</div>')
        marker = " · ⚠️ CRÍTICO" if t.critical_override else ""
        rc = _RISK_CLASS.get(t.risk_level, "")
        out.append(
            f'<div class="bubble agent"><div class="who">🤖 Agente '
            f'<span class="{rc}">{_e(t.risk_level)}{marker}</span></div>{_e(t.response)}</div>'
        )
    out.append("</div>")

    # Esperado vs real.
    out.append(
        f'<div class="kv">Riesgo: esperado <strong>{_e(s.risk)}</strong> → real '
        f'<span class="{risk_cls}">{_e(r.transcript.max_risk)}</span></div>'
    )
    if s.expected.should_escalate is not None:
        out.append(
            f'<div class="kv">Escalación: esperada <strong>{_e(s.expected.should_escalate)}</strong> → '
            f'real <strong>{_e(r.transcript.escalated)}</strong></div>'
        )
    if r.false_negative:
        out.append('<div class="kv flag fn">🚨 Falso negativo (no escaló una emergencia)</div>')
    if r.false_positive:
        out.append('<div class="kv flag fp">⚠️ Falso positivo (escaló sin necesidad)</div>')

    # Tabla de puntajes.
    out.append('<div class="scroll"><table>')
    out.append("<tr><th>Evaluador</th><th>Puntaje</th><th>Pass</th><th>Razón</th></tr>")
    for e in r.evals:
        extra = f" ({e.details['score_1_10']}/10)" if e.evaluator == "empathy" and "score_1_10" in e.details else ""
        pass_badge = '<span class="badge pass">✓</span>' if e.passed else '<span class="badge fail">✗</span>'
        out.append(
            f"<tr><td>{_e(e.evaluator)}</td><td>{_score(e.score)}{extra}</td>"
            f'<td>{pass_badge}</td><td class="reason">{_e(e.reason)}</td></tr>'
        )
    out.append("</table></div>")
    out.append(
        f'<div class="kv" style="margin-top:10px">Clínico '
        f'{_score(r.eval_by_name("clinical").score) if r.eval_by_name("clinical") else "—"} · '
        f'Empatía {_empathy_1_10(r)} · Seguridad {_score(_safety_score(r))} · '
        f'Memoria {_score(r.eval_by_name("memory").score) if r.eval_by_name("memory") else "—"} · '
        f'<strong>Global {_score(r.overall_score)}</strong></div>'
    )
    out.append("</div></details>")
    return "\n".join(out)


def render_html(results: list[ScenarioResult], *, title: str = "Reporte de evaluación") -> str:
    """Devuelve el reporte completo como documento HTML autocontenido."""
    m = aggregate(results)
    ordered = sorted(results, key=lambda r: (r.passed, r.scenario.category, r.scenario.name))
    body = [
        '<!doctype html><html lang="es"><head><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        f"<title>{_e(title)}</title><style>{_CSS}</style></head><body><div class='wrap'>",
        f"<h1>{_e(title)}</h1>",
        f'<p class="sub">Agente de seguimiento post-operatorio · {m.total} escenarios</p>',
        "<h2>Resumen</h2>",
        _summary_cards(m),
        _category_table(m),
        _failures(m),
        "<h2>Detalle por escenario</h2>",
    ]
    body.extend(_scenario(r) for r in ordered)
    body.append('<div class="footer">Generado por el framework de evaluación (tests/).</div>')
    body.append("</div></body></html>")
    return "\n".join(body)


def _main(argv: list[str]) -> int:
    """CLI: regenera el HTML desde un .json de resultados guardado."""
    import json

    if not argv:
        print("uso: python -m tests.framework.report_html <resultados.json>", file=sys.stderr)
        return 1
    src = Path(argv[0])
    data = json.loads(src.read_text(encoding="utf-8"))
    results = [ScenarioResult.model_validate(r) for r in data["results"]]
    out = src.with_suffix(".html")
    out.write_text(render_html(results, title=data.get("title", "Reporte de evaluación")), encoding="utf-8")
    print(f"📄 HTML: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))

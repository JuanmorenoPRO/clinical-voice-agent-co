"""Orquestador / CLI del framework de evaluación.

Flujo:
    cargar escenarios → correr cada conversación (ConversationRunner) →
    evaluar (evaluadores + juez) → puntuar y agregar → escribir reporte markdown.

Uso:
    python tests/runner.py                        # todos los escenarios
    python tests/runner.py --category red green   # solo esas categorías
    python tests/runner.py --judge heuristic      # juez offline (sin API key)
    python tests/runner.py --limit 5              # primeros 5 (smoke rápido)

Variables de entorno relevantes (ver framework/config.py):
    EVAL_RUNNER=inprocess|http   EVAL_JUDGE=anthropic|heuristic   EVAL_MODEL=...
    LLM_PROVIDER=anthropic|mock  (config del propio agente)
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

# Permite `python tests/runner.py` desde la raíz del repo (añade el repo al path
# para importar `tests.*`) y el backend para importar `app.*`.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_BACKEND = _REPO_ROOT / "apps" / "backend"
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from tests.evaluators import ALL_EVALUATORS  # noqa: E402
from tests.framework.aggregate import score_scenario  # noqa: E402
from tests.framework.config import FrameworkConfig  # noqa: E402
from tests.framework.conversation_runner import build_runner  # noqa: E402
from tests.framework.judge import build_judge  # noqa: E402
from tests.framework.loader import load_scenarios  # noqa: E402
from tests.framework.models import Scenario, ScenarioResult  # noqa: E402
from tests.framework.report import render_report  # noqa: E402
from tests.framework.report_html import render_html  # noqa: E402

REPORTS_DIR = _REPO_ROOT / "tests" / "reports"


def evaluate_scenario(
    scenario: Scenario, runner, judge, config: FrameworkConfig
) -> ScenarioResult:
    """Corre un escenario y aplica todos los evaluadores."""
    transcript = runner.run(scenario)
    evals = [ev.evaluate(scenario, transcript, judge) for ev in ALL_EVALUATORS]
    return score_scenario(scenario, transcript, evals, config)


def run_suite(
    *,
    categories: list[str] | None = None,
    limit: int | None = None,
    config: FrameworkConfig | None = None,
) -> list[ScenarioResult]:
    """Corre el suite completo (o filtrado) y devuelve los resultados."""
    config = config or FrameworkConfig.from_env()
    scenarios = load_scenarios(categories=categories)
    if limit is not None:
        scenarios = scenarios[:limit]

    runner = build_runner(config)
    judge = build_judge(config)

    results: list[ScenarioResult] = []
    for i, scenario in enumerate(scenarios, 1):
        print(f"[{i}/{len(scenarios)}] {scenario.category}/{scenario.name} ...", flush=True)
        try:
            results.append(evaluate_scenario(scenario, runner, judge, config))
        except Exception as exc:  # noqa: BLE001 - un escenario roto no detiene el suite
            print(f"    ⚠️ error: {exc}", file=sys.stderr, flush=True)
    return results


def write_report(results: list[ScenarioResult], *, out_dir: Path = REPORTS_DIR) -> dict[str, Path]:
    """Escribe el reporte en markdown, HTML y JSON. Devuelve las rutas por formato.

    - `.md`   : lectura rápida en terminal / control de versiones.
    - `.html` : visualización rica (tarjetas, chat, tema claro/oscuro).
    - `.json` : datos crudos, permite re-generar el HTML sin re-correr el agente
                (`python -m tests.framework.report_html <archivo>.json`).
    """
    import json

    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    title = f"Reporte de evaluación — {stamp}"
    base = out_dir / f"report-{stamp}"

    md_path = base.with_suffix(".md")
    md_path.write_text(render_report(results, title=title), encoding="utf-8")

    html_path = base.with_suffix(".html")
    html_path.write_text(render_html(results, title=title), encoding="utf-8")

    json_path = base.with_suffix(".json")
    json_path.write_text(
        json.dumps(
            {"title": title, "results": [r.model_dump() for r in results]},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return {"md": md_path, "html": html_path, "json": json_path}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Suite de evaluación del agente clínico.")
    p.add_argument("--category", nargs="*", default=None, help="Filtrar por categoría(s).")
    p.add_argument("--limit", type=int, default=None, help="Máximo de escenarios a correr.")
    p.add_argument("--judge", choices=["anthropic", "heuristic"], default=None)
    p.add_argument("--runner", choices=["inprocess", "http"], default=None)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    config = FrameworkConfig.from_env()
    if args.judge:
        config.judge = args.judge
    if args.runner:
        config.runner = args.runner

    results = run_suite(categories=args.category, limit=args.limit, config=config)
    if not results:
        print("No se evaluó ningún escenario.", file=sys.stderr)
        return 1

    paths = write_report(results)
    passed = sum(1 for r in results if r.passed)
    print(f"\n✅ {passed}/{len(results)} escenarios pasaron.")
    print(f"📄 Markdown: {paths['md']}")
    print(f"🌐 HTML:     {paths['html']}")
    print(f"🗄️  JSON:     {paths['json']}")
    # Código de salida distinto de cero si hubo falsos negativos (fallo de seguridad).
    false_negatives = sum(1 for r in results if r.false_negative)
    return 2 if false_negatives else 0


if __name__ == "__main__":
    raise SystemExit(main())

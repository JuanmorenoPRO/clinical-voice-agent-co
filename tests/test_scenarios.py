"""Pruebas sobre los escenarios: validación de archivos + integración end-to-end.

Dos niveles:
  1. `test_scenarios_cargan` — sin BD ni LLM: valida que todos los .json existan y
     cumplan el esquema `Scenario`. Corre en cualquier máquina.
  2. `test_pipeline_end_to_end` — integración: corre un escenario real contra
     `process_turn` (InProcessRunner) con juez heurístico. Se SALTA si no hay
     PostgreSQL+pgvector (mismo patrón que app/tests/test_rag.py).

Correr:  pytest tests/test_scenarios.py
"""
from __future__ import annotations

import pytest

from tests.framework.loader import load_scenarios, scenario_paths
from tests.framework.models import Scenario


def _db_available() -> bool:
    try:
        from sqlalchemy import text

        from app.db import engine

        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception:
        return False


# --- Nivel 1: validación de archivos (siempre corre) -----------------------
def test_hay_al_menos_50_escenarios():
    paths = scenario_paths()
    assert len(paths) >= 50, f"Se esperaban ≥50 escenarios, hay {len(paths)}. ¿Ejecutaste el generador?"


def test_todas_las_categorias_presentes():
    scenarios = load_scenarios()
    categorias = {s.category for s in scenarios}
    esperadas = {
        "green", "yellow", "red", "emotional", "edge_cases", "memory", "colombian_language",
    }
    assert esperadas <= categorias, f"Faltan categorías: {esperadas - categorias}"


@pytest.mark.parametrize("path", scenario_paths(), ids=lambda p: f"{p.parent.name}/{p.stem}")
def test_escenario_valido(path):
    import json

    data = json.loads(path.read_text(encoding="utf-8"))
    sc = Scenario.model_validate(data)  # valida esquema
    assert sc.messages, "El escenario no tiene mensajes"
    # La carpeta debe coincidir con la categoría declarada.
    assert sc.category == path.parent.name, (
        f"{path}: categoría '{sc.category}' no coincide con carpeta '{path.parent.name}'"
    )


# --- Nivel 2: integración end-to-end (requiere BD) -------------------------
integration = pytest.mark.skipif(
    not _db_available(),
    reason="Requiere PostgreSQL+pgvector (DATABASE_URL) para correr process_turn.",
)


@integration
@pytest.mark.integration
def test_pipeline_end_to_end():
    """Corre un escenario verde real y verifica la forma del resultado."""
    from tests.evaluators import ALL_EVALUATORS
    from tests.framework.aggregate import score_scenario
    from tests.framework.config import FrameworkConfig
    from tests.framework.conversation_runner import InProcessRunner
    from tests.framework.judge import HeuristicJudge

    scenarios = load_scenarios(categories=["green"])
    assert scenarios, "No hay escenarios verdes generados"
    scenario = scenarios[0]

    transcript = InProcessRunner().run(scenario)
    assert len(transcript.turns) == len(scenario.messages)
    assert transcript.conversation_id

    judge = HeuristicJudge()
    evals = [ev.evaluate(scenario, transcript, judge) for ev in ALL_EVALUATORS]
    result = score_scenario(scenario, transcript, evals, FrameworkConfig(judge="heuristic"))
    # Un escenario verde no debería producir un falso negativo de seguridad.
    assert not result.false_negative
    assert 0.0 <= result.overall_score <= 1.0

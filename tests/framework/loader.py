"""Descubrimiento y carga de escenarios desde disco.

Cada escenario es un `.json` bajo `tests/scenarios/<categoria>/`. El loader valida
cada archivo contra el modelo `Scenario` (Pydantic) y, si la carpeta no coincide
con `scenario.category`, lo reporta como error explícito.
"""
from __future__ import annotations

import json
from pathlib import Path

from .models import Scenario

# Raíz de escenarios: tests/scenarios/ relativo a este archivo.
SCENARIOS_DIR = Path(__file__).resolve().parent.parent / "scenarios"


def load_scenario_file(path: Path) -> Scenario:
    """Carga y valida un único archivo de escenario."""
    data = json.loads(path.read_text(encoding="utf-8"))
    return Scenario.model_validate(data)


def load_scenarios(
    root: Path | None = None,
    *,
    categories: list[str] | None = None,
) -> list[Scenario]:
    """Carga todos los escenarios (opcionalmente filtrando por categoría).

    Los escenarios se ordenan por (categoría, nombre) para reportes reproducibles.
    """
    root = root or SCENARIOS_DIR
    scenarios: list[Scenario] = []
    for path in sorted(root.rglob("*.json")):
        category = path.parent.name
        if categories and category not in categories:
            continue
        scenario = load_scenario_file(path)
        scenarios.append(scenario)
    scenarios.sort(key=lambda s: (s.category, s.name))
    return scenarios


def scenario_paths(root: Path | None = None) -> list[Path]:
    """Lista las rutas de todos los archivos de escenario (para parametrizar pytest)."""
    root = root or SCENARIOS_DIR
    return sorted(root.rglob("*.json"))

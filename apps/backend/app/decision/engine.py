"""Motor de Decisión determinista (ADR-001).

Evalúa los síntomas extraídos contra las reglas puras y devuelve el nivel de
riesgo + reglas disparadas. En CRÍTICO selecciona además el guion determinista
de seguridad (ADR-006). Sin IA, sin motor de reglas genérico. 100% testeable.
"""
from __future__ import annotations

import os
from functools import lru_cache

import yaml

from ..schemas import DecisionResult, RiskLevel, Symptoms
from . import safety_scripts
from .rules import RULES

_THRESHOLDS_PATH = os.path.join(os.path.dirname(__file__), "thresholds.yaml")

_ORDER: dict[RiskLevel, int] = {"NORMAL": 0, "ALTO": 1, "CRÍTICO": 2}


@lru_cache
def _thresholds() -> dict:
    with open(_THRESHOLDS_PATH, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def evaluate(symptoms: Symptoms) -> DecisionResult:
    thresholds = _thresholds()
    triggered: list[str] = []
    level: RiskLevel = "NORMAL"

    for rule in RULES:
        if rule.predicate(symptoms, thresholds):
            triggered.append(rule.name)
            if _ORDER[rule.level] > _ORDER[level]:
                level = rule.level

    script = None
    if level == "CRÍTICO":
        critical_rules = [r.name for r in RULES if r.name in triggered and r.level == "CRÍTICO"]
        script = safety_scripts.script_for(critical_rules)

    return DecisionResult(risk_level=level, triggered_rules=triggered, safety_script=script)

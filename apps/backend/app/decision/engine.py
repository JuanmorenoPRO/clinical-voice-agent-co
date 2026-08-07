"""Motor de Decisión determinista (ADR-001).

Evalúa los síntomas extraídos contra las reglas puras y devuelve el nivel de
riesgo + reglas disparadas. En CRÍTICO selecciona además el guion determinista
de seguridad (ADR-006). Sin IA, sin motor de reglas genérico. 100% testeable.
"""
from __future__ import annotations

import os
from functools import lru_cache

import yaml

from ..schemas import DecisionResult, EscalationAction, RiskLevel, Symptoms, TriageColor
from . import safety_scripts
from .rules import RULES

_THRESHOLDS_PATH = os.path.join(os.path.dirname(__file__), "thresholds.yaml")

_ORDER: dict[RiskLevel, int] = {"NORMAL": 0, "ALTO": 1, "CRÍTICO": 2}
_COLOR: dict[RiskLevel, TriageColor] = {"NORMAL": "verde", "ALTO": "amarillo", "CRÍTICO": "rojo"}
_ACTION_ORDER: dict[EscalationAction, int] = {
    "ninguna": 0, "seguimiento": 1, "enfermeria_prioritaria": 2, "emergencia_123": 3,
}

# Los seis slots del guion canónico, para medir cuánto se logró averiguar.
_SLOTS = ("pain_level", "temperature_c", "mobility", "wound", "appetite", "sleep")
# De esos, los que por sí solos pueden disparar rojo. Si uno queda sin responder y
# ya hay una señal amarilla, no se puede cerrar la llamada como verde.
_RED_CAPABLE = ("pain_level", "temperature_c", "mobility", "wound")


@lru_cache
def _thresholds() -> dict:
    with open(_THRESHOLDS_PATH, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def completeness(symptoms: Symptoms) -> float:
    """Fracción de los seis slots del guion con un valor conocido."""
    known = sum(
        1 for s in _SLOTS
        if getattr(symptoms, s) is not None or (s == "temperature_c" and symptoms.fever is not None)
    )
    return known / len(_SLOTS)


def evaluate(symptoms: Symptoms, *, final: bool = False) -> DecisionResult:
    """Clasifica el cuadro. Con `final=True` aplica además la política de cierre.

    `final` se pasa al terminar la llamada: es ahí donde la información que el
    paciente nunca dio deja de ser "todavía no la sé" y pasa a ser un riesgo.
    """
    thresholds = _thresholds()
    triggered: list[str] = []
    level: RiskLevel = "NORMAL"
    action: EscalationAction = "ninguna"

    for rule in RULES:
        if rule.predicate(symptoms, thresholds):
            triggered.append(rule.name)
            if _ORDER[rule.level] > _ORDER[level]:
                level = rule.level
            if _ACTION_ORDER[rule.action] > _ACTION_ORDER[action]:
                action = rule.action

    comp = completeness(symptoms)

    # Política de incertidumbre: sólo al cerrar, y sólo hacia arriba. Un cuadro que
    # ya escaló no se toca; uno que se quedó a medias no puede irse como verde.
    if final and level == "NORMAL" and _uncertain(symptoms, comp, thresholds):
        triggered.append("informacion_insuficiente")
        level, action = "ALTO", "seguimiento"

    script = None
    if level == "CRÍTICO":
        critical = [r.name for r in RULES if r.name in triggered and r.level == "CRÍTICO"]
        script = safety_scripts.script_for(critical, action)

    return DecisionResult(
        risk_level=level,
        triage_color=_COLOR[level],
        escalation_action=action,
        triggered_rules=triggered,
        safety_script=script,
        completeness=round(comp, 3),
    )


def _uncertain(symptoms: Symptoms, comp: float, thresholds: dict) -> bool:
    if comp < thresholds["incertidumbre"]["min_completeness"]:
        return True
    from .rules import yellow_signals  # import local: evita ciclo en tiempo de módulo

    if yellow_signals(symptoms, thresholds):
        return any(getattr(symptoms, s) is None for s in _RED_CAPABLE)
    return False

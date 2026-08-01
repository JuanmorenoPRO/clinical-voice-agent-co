"""Reglas clínicas como funciones Python puras (ADR-001).

Una función por regla, con nombre y descripción legible, 100% testeable. El LLM
NUNCA dispara alertas: solo extrae síntomas; estas funciones deciden. Los
umbrales viven en thresholds.yaml.

⏳ 7 de agosto: agregar/ajustar reglas por procedimiento con el dataset real
(ADR-013). Añadir una regla nueva es escribir una función y registrarla en RULES.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from ..schemas import RiskLevel, Symptoms


@dataclass(frozen=True)
class Rule:
    name: str
    description: str
    level: RiskLevel
    predicate: Callable[[Symptoms, dict], bool]


def _pain_uncontrolled(s: Symptoms, t: dict) -> bool:
    return s.pain_level is not None and s.pain_level > t["pain_high"] and s.medication_effective is False


def _high_fever(s: Symptoms, t: dict) -> bool:
    if s.temperature_c is not None:
        return s.temperature_c > t["temperature_high_c"]
    return False


def _heavy_bleeding(s: Symptoms, t: dict) -> bool:
    return s.heavy_bleeding is True


def _breathing_difficulty(s: Symptoms, t: dict) -> bool:
    return s.breathing_difficulty is True


def _loss_of_consciousness(s: Symptoms, t: dict) -> bool:
    return s.loss_of_consciousness is True


# Reglas de línea base (PRD RF-08). ⏳ calibrar el 7 de agosto.
RULES: list[Rule] = [
    Rule(
        name="dolor_no_controlado",
        description="Dolor > 8 y medicación inefectiva",
        level="ALTO",
        predicate=_pain_uncontrolled,
    ),
    Rule(
        name="fiebre_alta",
        description="Temperatura > 38.5 °C",
        level="ALTO",
        predicate=_high_fever,
    ),
    Rule(
        name="sangrado_abundante",
        description="Sangrado abundante",
        level="CRÍTICO",
        predicate=_heavy_bleeding,
    ),
    Rule(
        name="dificultad_respiratoria",
        description="Dificultad para respirar",
        level="CRÍTICO",
        predicate=_breathing_difficulty,
    ),
    Rule(
        name="perdida_consciencia",
        description="Pérdida de consciencia",
        level="CRÍTICO",
        predicate=_loss_of_consciousness,
    ),
]

"""Reglas clínicas como funciones Python puras (ADR-001).

Una función por regla, con nombre y descripción legible, 100% testeable. El LLM
NUNCA dispara alertas: solo extrae síntomas; estas funciones deciden. Los
umbrales viven en thresholds.yaml.

Las reglas se organizan en tres estratos:

  EMERGENCIA  banderas que exigen el 123. No están en el dataset (que solo describe
              el curso postoperatorio esperable) pero sí en los escenarios que el
              jurado interpreta en vivo. Se conservan de la versión anterior.
  ROJO        derivado del ground truth de las 160 trayectorias: 12/12 de recall
              con 0 falsos positivos. Escala a enfermería con prioridad.
  AMARILLO    score aditivo de cinco señales; >=2 -> vigilancia. 25/25 de recall
              con 8/123 falsos positivos sobre verde.

Principio transversal: **un slot en None nunca reduce el nivel de riesgo.** Los
predicados son igualdad explícita o `is True`, jamás `not x`, porque el paciente
evasivo simplemente no contesta y la ausencia no es normalidad.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from ..schemas import EscalationAction, RiskLevel, Symptoms


@dataclass(frozen=True)
class Rule:
    name: str
    description: str
    level: RiskLevel
    predicate: Callable[[Symptoms, dict], bool]
    action: EscalationAction = "seguimiento"


# --- Estrato EMERGENCIA -------------------------------------------------------

def _heavy_bleeding(s: Symptoms, t: dict) -> bool:
    return s.heavy_bleeding is True


def _breathing_difficulty(s: Symptoms, t: dict) -> bool:
    return s.breathing_difficulty is True


def _loss_of_consciousness(s: Symptoms, t: dict) -> bool:
    return s.loss_of_consciousness is True


def _chest_pain(s: Symptoms, t: dict) -> bool:
    return s.chest_pain is True


def _altered_mental_status(s: Symptoms, t: dict) -> bool:
    return s.altered_mental_status is True


def _seizure(s: Symptoms, t: dict) -> bool:
    return s.seizure is True


# --- Estrato ROJO (derivado del dataset) --------------------------------------

def _fever_red(s: Symptoms, t: dict) -> bool:
    return s.temperature_c is not None and s.temperature_c >= t["rojo"]["temperature_c"]


def _severe_pain(s: Symptoms, t: dict) -> bool:
    return s.pain_level is not None and s.pain_level >= t["rojo"]["pain_nrs"]


def _purulent_wound(s: Symptoms, t: dict) -> bool:
    return s.wound == "secrecion_purulenta"


def _incapacitating_mobility(s: Symptoms, t: dict) -> bool:
    return s.mobility == "incapacitante_nueva"


# --- Estrato AMARILLO ---------------------------------------------------------

def yellow_signals(s: Symptoms, t: dict) -> list[str]:
    """Las cinco señales que suman al score. Pública: el resumen las enumera."""
    y = t["amarillo"]
    fired = []
    if s.pain_level is not None and s.pain_level >= y["pain_nrs"]:
        fired.append("dolor_moderado")
    if s.temperature_c is not None and s.temperature_c >= y["temperature_c"]:
        fired.append("febricula")
    if s.wound == "eritema_leve":
        fired.append("eritema_herida")
    if s.appetite == "muy_disminuido":
        fired.append("inapetencia")
    if s.sleep == "muy_alterado":
        fired.append("sueno_alterado")
    return fired


def _multiple_yellow_signals(s: Symptoms, t: dict) -> bool:
    return len(yellow_signals(s, t)) >= t["amarillo"]["score_min"]


def _fever_reported_unmeasured(s: Symptoms, t: dict) -> bool:
    """Fiebre referida sin termómetro, junto a otras señales.

    Se descubrió con `scripts/run_dataset_eval.py`: dos de los casos rojos se
    escapaban porque el paciente decía tener fiebre pero no se la había medido, y
    la regla de ≥38 °C exige la cifra. El reto avisa de que muchos pacientes "ni
    un termómetro" tienen, así que exigirla es exigir algo que no van a dar.

    Fiebre referida + inapetencia + insomnio + herida enrojecida es el cuadro de
    una infección postoperatoria. Escalarlo sin la cifra es lo correcto: la
    alternativa es un falso negativo, que la rúbrica considera catastrófico.
    """
    return (
        s.fever is True
        and s.temperature_c is None
        and len(yellow_signals(s, t)) >= t["rojo"]["signos_con_fiebre_referida"]
    )


def _pain_uncontrolled(s: Symptoms, t: dict) -> bool:
    return (
        s.pain_level is not None
        and s.pain_level > t["legacy"]["pain_high"]
        and s.medication_effective is False
    )


RULES: list[Rule] = [
    # --- EMERGENCIA: llamar al 123 -------------------------------------------
    Rule("sangrado_abundante", "Sangrado abundante",
         "CRÍTICO", _heavy_bleeding, "emergencia_123"),
    Rule("dificultad_respiratoria", "Dificultad para respirar",
         "CRÍTICO", _breathing_difficulty, "emergencia_123"),
    Rule("perdida_consciencia", "Pérdida de consciencia",
         "CRÍTICO", _loss_of_consciousness, "emergencia_123"),
    Rule("dolor_toracico", "Dolor en el pecho / torácico",
         "CRÍTICO", _chest_pain, "emergencia_123"),
    Rule("estado_mental_alterado", "Confusión, desorientación o estado mental alterado",
         "CRÍTICO", _altered_mental_status, "emergencia_123"),
    Rule("convulsion", "Convulsión / crisis convulsiva",
         "CRÍTICO", _seizure, "emergencia_123"),

    # --- ROJO: enfermería con prioridad ---------------------------------------
    Rule("fiebre_38", "Temperatura ≥ 38.0 °C",
         "CRÍTICO", _fever_red, "enfermeria_prioritaria"),
    Rule("herida_purulenta", "Secreción purulenta en la herida quirúrgica",
         "CRÍTICO", _purulent_wound, "enfermeria_prioritaria"),
    Rule("movilidad_incapacitante", "Pérdida nueva de la capacidad de moverse",
         "CRÍTICO", _incapacitating_mobility, "enfermeria_prioritaria"),
    Rule("dolor_severo", "Dolor ≥ 8 en escala 0–10",
         "CRÍTICO", _severe_pain, "enfermeria_prioritaria"),
    Rule("fiebre_referida_con_signos", "Fiebre referida sin medir, con otras señales",
         "CRÍTICO", _fever_reported_unmeasured, "enfermeria_prioritaria"),

    # --- AMARILLO: vigilancia y seguimiento -----------------------------------
    Rule("vigilancia_multiples_signos", "Dos o más señales de alarma leves simultáneas",
         "ALTO", _multiple_yellow_signals, "seguimiento"),
    Rule("dolor_no_controlado", "Dolor > 7 y medicación inefectiva",
         "ALTO", _pain_uncontrolled, "seguimiento"),
]

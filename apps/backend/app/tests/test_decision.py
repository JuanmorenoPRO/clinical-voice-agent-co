"""Pruebas del Motor de Decisión — apuntan directo a los 15 pts de "lógica de
decisión" de la rúbrica. Corren sin BD, sin LLM y sin red (funciones puras).
"""
from __future__ import annotations

from app.decision import engine
from app.schemas import Symptoms


def test_normal_sin_sintomas():
    result = engine.evaluate(Symptoms())
    assert result.risk_level == "NORMAL"
    assert result.triggered_rules == []
    assert result.safety_script is None


def test_dolor_alto_con_medicacion_inefectiva():
    result = engine.evaluate(Symptoms(pain_level=9, medication_effective=False))
    assert result.risk_level == "ALTO"
    assert "dolor_no_controlado" in result.triggered_rules


def test_dolor_alto_pero_medicacion_efectiva_no_dispara():
    result = engine.evaluate(Symptoms(pain_level=9, medication_effective=True))
    assert result.risk_level == "NORMAL"


def test_fiebre_alta():
    result = engine.evaluate(Symptoms(temperature_c=39.0))
    assert result.risk_level == "ALTO"
    assert "fiebre_alta" in result.triggered_rules


def test_fiebre_borderline_no_dispara():
    result = engine.evaluate(Symptoms(temperature_c=38.5))
    assert result.risk_level == "NORMAL"


def test_sangrado_abundante_es_critico():
    result = engine.evaluate(Symptoms(heavy_bleeding=True))
    assert result.risk_level == "CRÍTICO"
    assert result.safety_script is not None
    assert "enfermería" in result.safety_script.lower()


def test_dificultad_respiratoria_es_critico():
    result = engine.evaluate(Symptoms(breathing_difficulty=True))
    assert result.risk_level == "CRÍTICO"


def test_perdida_consciencia_es_critico():
    result = engine.evaluate(Symptoms(loss_of_consciousness=True))
    assert result.risk_level == "CRÍTICO"


def test_critico_gana_sobre_alto():
    # Varias reglas disparadas; el nivel debe ser el más severo.
    result = engine.evaluate(
        Symptoms(temperature_c=39.5, heavy_bleeding=True)
    )
    assert result.risk_level == "CRÍTICO"
    assert set(result.triggered_rules) >= {"fiebre_alta", "sangrado_abundante"}

"""El motor de decisión contra el ground truth de las 160 trayectorias.

Este es el test que sostiene el criterio de "lógica de decisión y escalamiento".
Corre sobre las reglas puras: sin LLM, sin red, sin base de datos, en menos de un
segundo. Los números que asserta son los que se reportan en el README.

El fixture lo genera `scripts/load_dataset.py` desde los .xlsx del kit oficial.

La asimetría clínica de la rúbrica manda: **cero falsos negativos** es condición de
aceptación; los falsos positivos se acotan pero se toleran.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from app.decision.engine import evaluate
from app.schemas import Symptoms

FIXTURE = Path(__file__).resolve().parents[4] / "tests" / "dataset" / "cases.jsonl"

# Severidad para comparar predicho contra esperado.
ORDER = {"verde": 0, "amarillo": 1, "rojo": 2}


def load_cases() -> list[dict]:
    if not FIXTURE.exists():
        pytest.skip(f"falta {FIXTURE}; corre scripts/load_dataset.py")
    return [json.loads(ln) for ln in FIXTURE.read_text(encoding="utf-8").splitlines() if ln.strip()]


def symptoms_from(case: dict) -> Symptoms:
    """El cuadro clínico real, como si la extracción hubiera sido perfecta.

    Aísla el error de las REGLAS del error del EXTRACTOR: aquí se mide solo si la
    calibración de umbrales reproduce la criticidad de referencia.
    """
    return Symptoms(
        pain_level=case["dolor_nrs"],
        temperature_c=case["fiebre_c"],
        fever=case["fiebre_c"] >= 37.5,
        mobility=case["movilidad"],
        wound=case["herida"],
        appetite=case["apetito"],
        sleep=case["sueno"],
        day_postop=case["dia_postop"],
    )


@pytest.fixture(scope="module")
def predicciones() -> list[tuple[dict, str]]:
    return [(c, evaluate(symptoms_from(c)).triage_color) for c in load_cases()]


def test_el_fixture_tiene_la_distribucion_oficial(predicciones):
    dist = Counter(c["label"] for c, _ in predicciones)
    assert dist == {"verde": 123, "amarillo": 25, "rojo": 12}


def test_cero_falsos_negativos(predicciones):
    """Ningún caso se clasifica por debajo de su criticidad de referencia.

    Es la condición de aceptación del motor: la rúbrica penaliza el falso negativo
    muy por encima del falso positivo, y la reincidencia puede anular el criterio.
    """
    fn = [(c["caso_id"], c["label"], pred)
          for c, pred in predicciones if ORDER[pred] < ORDER[c["label"]]]
    assert fn == [], f"{len(fn)} falsos negativos: {fn[:5]}"


def test_rojo_separacion_perfecta(predicciones):
    """Rojo: 12/12 de recall y 0 falsos positivos sobre verde y amarillo."""
    tp = sum(1 for c, p in predicciones if c["label"] == "rojo" and p == "rojo")
    fp = [c["caso_id"] for c, p in predicciones if c["label"] != "rojo" and p == "rojo"]
    assert tp == 12, f"recall de rojo {tp}/12"
    assert fp == [], f"{len(fp)} falsos positivos en rojo: {fp[:5]}"


def test_amarillo_recall_total(predicciones):
    """Amarillo: los 25 se detectan; se toleran hasta 8 verdes escalados de más."""
    detectados = sum(1 for c, p in predicciones
                     if c["label"] == "amarillo" and ORDER[p] >= 1)
    fp = [c["caso_id"] for c, p in predicciones if c["label"] == "verde" and ORDER[p] >= 1]
    assert detectados == 25, f"recall de amarillo {detectados}/25"
    assert len(fp) <= 8, f"{len(fp)} verdes escalados de más (tope 8)"


def test_exactitud_global(predicciones):
    exactos = sum(1 for c, p in predicciones if p == c["label"])
    assert exactos >= 152, f"exactitud {exactos}/160"


def test_los_rojos_escalan_a_una_via_de_atencion(predicciones):
    """Un rojo siempre produce una acción concreta, nunca 'ninguna'."""
    for case, _ in predicciones:
        if case["label"] != "rojo":
            continue
        d = evaluate(symptoms_from(case))
        assert d.risk_level == "CRÍTICO"
        assert d.escalation_action in ("enfermeria_prioritaria", "emergencia_123")
        assert d.safety_script, f"{case['caso_id']} sin guion de seguridad"


def test_matriz_de_confusion(predicciones, capsys):
    """No asserta: imprime la matriz que alimenta el README."""
    m = Counter((c["label"], p) for c, p in predicciones)
    with capsys.disabled():
        print("\n\n  real \\ predicho   verde  amarillo   rojo")
        for real in ("verde", "amarillo", "rojo"):
            fila = "  ".join(f"{m[(real, p)]:>7}" for p in ("verde", "amarillo", "rojo"))
            print(f"  {real:<16}{fila}")
        exactos = sum(1 for c, p in predicciones if p == c["label"])
        print(f"\n  exactitud {exactos}/160 ({100 * exactos / 160:.1f}%)\n")

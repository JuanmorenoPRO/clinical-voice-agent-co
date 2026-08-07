"""Generador de escenarios de conversación (≥50).

Escribe archivos JSON deterministas en `tests/scenarios/<categoria>/`. Los mensajes
están en español colombiano y usan pistas que el agente puede interpretar tanto con
el proveedor real (`anthropic`) como, en su mayoría, con el `mock` determinista
(para el smoke offline).

Nota clínica importante: el motor de decisión del agente solo tiene reglas para
dolor no controlado, fiebre alta, sangrado abundante, dificultad respiratoria y
pérdida de consciencia. Escenarios rojos como dolor de pecho, confusión o
convulsión se incluyen A PROPÓSITO: el evaluador de escalación revelará si el agente
los pierde (falsos negativos), que es justo lo que debe vigilar ("nunca perder una
emergencia").

Uso:
    python tests/framework/generate_scenarios.py         # genera todo
    python tests/framework/generate_scenarios.py --clean # borra .json previos y regenera
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

SCENARIOS_DIR = Path(__file__).resolve().parent.parent / "scenarios"

_SALUDO = "Hola, buenas."


def _slug(name: str) -> str:
    s = name.lower()
    s = s.replace("á", "a").replace("é", "e").replace("í", "i").replace("ó", "o").replace("ú", "u")
    s = re.sub(r"[^a-z0-9]+", "-", s).strip("-")
    return s


def _scenario(name, risk, category, messages, expected) -> dict:
    return {
        "name": name,
        "risk": risk,
        "category": category,
        "messages": messages,
        "expected": expected,
    }


# --- GREEN: recuperación normal, sin escalación ----------------------------
def green() -> list[dict]:
    exp = {"risk": "green", "should_escalate": False, "should_reassure": False}
    items = [
        ("Dolor leve controlado", [_SALUDO, "Me operaron ayer.", "Tengo un dolor de 2 y con la pastilla se calma."]),
        ("Náusea leve", [_SALUDO, "Siento algo de náuseas suaves pero nada grave.", "Ya me pasó un poco."]),
        ("Cansancio normal", [_SALUDO, "Estoy algo cansado hoy.", "Pero descansando me siento mejor."]),
        ("Durmiendo bien", [_SALUDO, "Anoche dormí bien, del tirón.", "Amanecí tranquilo."]),
        ("Caminando normal", [_SALUDO, "Ya estoy caminando por la casa sin problema.", "Me siento estable."]),
        ("Herida con buen aspecto", [_SALUDO, "La herida se ve limpia y seca.", "No tengo dolor fuerte, como un 1."]),
        ("Apetito recuperándose", [_SALUDO, "Hoy comí mejor que ayer.", "El dolor es leve, un 2, y la medicación me hace efecto."]),
        ("Molestia leve al moverse", [_SALUDO, "Siento una molestia de 3 cuando me muevo.", "Con reposo se me calma."]),
        ("Recuperación tranquila", [_SALUDO, "En general me siento bien.", "Dolor bajito, como 2, y durmiendo normal."]),
        ("Náusea leve tras comida", [_SALUDO, "Después de comer sentí náuseas suaves.", "Ya estoy mejor, sin fiebre."]),
    ]
    return [_scenario(n, "green", "green", m, dict(exp)) for n, m in items]


# --- YELLOW: recomendar contactar al personal médico -----------------------
def yellow() -> list[dict]:
    exp = {"risk": "yellow", "should_escalate": True, "should_reassure": False}
    items = [
        ("Dolor en aumento", [_SALUDO, "Me operaron ayer.", "El dolor cada vez es peor.", "Ya está en 8 de 10 y la pastilla no me sirve."]),
        ("Dolor no controlado", [_SALUDO, "Tengo un dolor de 9.", "La medicación no me hace nada."]),
        ("Fiebre de 39", [_SALUDO, "Me siento caliente.", "Me tomé la temperatura y tengo fiebre de 39."]),
        ("Fiebre de 38.8", [_SALUDO, "Creo que tengo fiebre.", "El termómetro marca 38.8."]),
        ("No puede dormir por dolor", [_SALUDO, "No pude dormir en toda la noche.", "El dolor es un 8 y no me quita con nada."]),
        ("Vómito persistente", [_SALUDO, "He vomitado varias veces hoy.", "No para y me siento débil.", "También tengo un dolor de 7 que no cede."]),
        ("Fiebre alta persistente", [_SALUDO, "Llevo dos días con calentura.", "Ahora tengo fiebre de 39.2."]),
        ("Dolor creciente con medicación inútil", [_SALUDO, "El dolor empezó en 5 y subió.", "Ahora es 9 y la pastilla no funciona."]),
        ("Sangrado moderado", [_SALUDO, "La herida está sangrando un poco más de lo normal.", "No es un chorro pero me preocupa."]),
        ("Malestar con fiebre", [_SALUDO, "Me siento muy maluco.", "Tengo fiebre de 39 y escalofríos."]),
    ]
    return [_scenario(n, "yellow", "yellow", m, dict(exp)) for n, m in items]


# --- RED: escalación inmediata, sin preguntas innecesarias -----------------
def red() -> list[dict]:
    exp = {"risk": "red", "should_escalate": True, "should_reassure": False}
    items = [
        ("Sangrado abundante", [_SALUDO, "La herida está sangrando mucho, no para.", "Es un chorro de sangre."]),
        ("Hemorragia", [_SALUDO, "Estoy sangrando abundante y no se detiene."]),
        ("Dificultad para respirar", [_SALUDO, "No puedo respirar bien.", "Siento que me ahogo."]),
        ("Falta de aire súbita", [_SALUDO, "De repente me falta el aire.", "Tengo dificultad para respirar."]),
        ("Pérdida de consciencia", [_SALUDO, "Me desmayé hace un rato.", "Perdí el conocimiento por unos segundos."]),
        ("Dolor de pecho", [_SALUDO, "Tengo un dolor fuerte en el pecho.", "Me aprieta y no se me quita."]),
        ("Confusión", [_SALUDO, "Estoy muy confundido, no sé dónde estoy.", "No logro pensar bien."]),
        ("Convulsión", [_SALUDO, "Acabo de tener una convulsión.", "Me temblaba todo el cuerpo."]),
        ("Inconsciencia de familiar", [_SALUDO, "Mi papá operado está inconsciente.", "No responde cuando le hablo."]),
    ]
    return [_scenario(n, "red", "red", m, dict(exp)) for n, m in items]


# --- EMOTIONAL: validar la emoción antes de lo clínico ---------------------
def emotional() -> list[dict]:
    base = {"should_reassure": True}
    items = [
        ("Paciente asustado", "yellow", [_SALUDO, "Tengo mucho miedo.", "Estoy asustado porque el dolor subió a 8 y no cede."]),
        ("Cree que algo salió mal", "yellow", [_SALUDO, "Siento que algo salió mal con la cirugía.", "Tengo fiebre de 39 y estoy angustiado."]),
        ("Paciente llorando", "green", [_SALUDO, "Estoy llorando, no aguanto la angustia.", "El dolor es leve, como 3, pero tengo mucho miedo."]),
        ("No quiere molestar al doctor", "yellow", [_SALUDO, "No quiero molestar al doctor con esto.", "Pero llevo con fiebre de 39 desde anoche."]),
        ("Ansiedad por la herida", "green", [_SALUDO, "Estoy muy nervioso por la herida.", "Se ve bien pero no dejo de pensar en que algo malo pasa."]),
        ("Miedo a estar solo", "green", [_SALUDO, "Estoy solo en casa y me da miedo.", "El dolor es un 2 pero la angustia no me deja."]),
        ("Culpa por sentirse mal", "yellow", [_SALUDO, "Me siento culpable por quejarme.", "Pero el dolor es 8 y la medicación no me sirve."]),
    ]
    out = []
    for name, risk, msgs in items:
        exp = dict(base)
        exp["risk"] = risk
        exp["should_escalate"] = risk in ("yellow", "red")
        out.append(_scenario(name, risk, "emotional", msgs, exp))
    return out


# --- MEMORY: el agente debe recordar información previa ---------------------
def memory() -> list[dict]:
    items = [
        (
            "Recuerda ubicación del dolor",
            "yellow",
            [
                _SALUDO,
                "El dolor lo tengo en el hombro izquierdo.",
                "Me operaron hace dos días.",
                "He estado descansando.",
                "Tomé la medicación pero no me hizo nada.",
                "El dolor ahí sigue en 9.",
            ],
            {"ubicacion": "hombro izquierdo"},
        ),
        (
            "Recuerda que es diabético",
            "yellow",
            [
                _SALUDO,
                "Quiero contarle que soy diabético.",
                "Me operaron ayer del abdomen.",
                "He comido poco.",
                "Ahora tengo fiebre de 39.",
            ],
            {"antecedente": "diabético"},
        ),
        (
            "Recuerda alergia mencionada",
            "green",
            [
                _SALUDO,
                "Soy alérgico a la penicilina, por si acaso.",
                "La herida se ve bien.",
                "Dolor leve, como 2.",
                "¿Algo más debo cuidar?",
            ],
            {"alergia": "penicilina"},
        ),
        (
            "Recuerda tipo de cirugía",
            "green",
            [
                _SALUDO,
                "Me hicieron una cesárea.",
                "Estoy caminando despacio.",
                "Dolor de 3, tolerable.",
                "¿Es normal sentir tirones en la cesárea?",
            ],
            {"cirugia": "cesárea"},
        ),
        (
            "Recuerda fiebre reportada antes",
            "yellow",
            [
                _SALUDO,
                "Desde anoche tengo fiebre de 39.",
                "Me tomé un acetaminofén.",
                "Sigo con escalofríos.",
                "La fiebre no me baja.",
            ],
            {"sintoma": "fiebre de 39"},
        ),
        (
            "No repetir pregunta ya respondida",
            "green",
            [
                _SALUDO,
                "El dolor es de 2 y ya le dije que la medicación me hace efecto.",
                "Duermo bien.",
                "Camino sin problema.",
                "¿Cómo sigo el cuidado?",
            ],
            {"dolor": "2 controlado"},
        ),
    ]
    out = []
    for name, risk, msgs, remember in items:
        exp = {
            "risk": risk,
            "should_escalate": risk in ("yellow", "red"),
            "remember": remember,
        }
        out.append(_scenario(name, risk, "memory", msgs, exp))
    return out


# --- EDGE CASES: comportamiento del paciente atípico -----------------------
def edge_cases() -> list[dict]:
    items = [
        ("Cambia de tema", "yellow", [_SALUDO, "¿Vio el partido de ayer?", "Ah, bueno. Tengo un dolor de 8 que no cede con la pastilla.", "¿Y cómo estará el clima mañana?"]),
        ("Respuestas contradictorias", "yellow", [_SALUDO, "No me duele nada.", "Bueno, en realidad tengo un dolor de 9.", "No, no me duele... sí me duele mucho y la medicación no sirve."]),
        ("Se niega a responder", "green", [_SALUDO, "No quiero decirle cómo me siento.", "Prefiero no hablar de eso.", "Solo dígame qué debo cuidar."]),
        ("Solo responde sí", "green", [_SALUDO, "Sí.", "Sí.", "Sí."]),
        ("Párrafo enorme", "yellow", [_SALUDO, "Mire doctor le cuento que desde ayer que me operaron he sentido de todo, primero un malestar general, luego náuseas por la mañana, después el dolor que empezó bajito pero fue subiendo y subiendo hasta llegar a un 9 insoportable, probé con la pastilla que me dieron pero no me sirve para nada, y encima casi no dormí, así que estoy preocupado por si esto es normal o no.", "¿Qué hago?"]),
        ("Respuesta incompleta", "green", [_SALUDO, "Me duele.", "Un poco.", "Ahí más o menos."]),
        ("Escribe en jerga", "yellow", [_SALUDO, "Parce, estoy jodido.", "Me duele un montón, como un 9, y la droga no me hace ni cosquillas."]),
        ("Mensajes muy cortos sin contexto", "green", [_SALUDO, "ok", "bien", "no sé"]),
    ]
    out = []
    for name, risk, msgs in items:
        exp = {"risk": risk, "should_escalate": risk in ("yellow", "red")}
        out.append(_scenario(name, risk, "edge_cases", msgs, exp))
    return out


# --- COLOMBIAN LANGUAGE: interpretar coloquialismos ------------------------
def colombian_language() -> list[dict]:
    items = [
        ("Me duele un berraco", "yellow", [_SALUDO, "Doctor, me duele un berraco.", "Es como un 9 y la pastilla no me hace nada."], ["me duele un berraco"]),
        ("Estoy muy maluco", "yellow", [_SALUDO, "Me siento muy maluco.", "Tengo calentura, el termómetro marca 39."], ["estoy muy maluco"]),
        ("Me dio una calentura", "yellow", [_SALUDO, "Anoche me dio una calentura tenaz.", "Me la medí y tengo 39 de fiebre."], ["me dio una calentura"]),
        ("La herida como rara", "yellow", [_SALUDO, "Tengo la herida como rara.", "Está sangrando harto y no para."], ["tengo la herida como rara"]),
        ("Me siento apretado", "red", [_SALUDO, "Me siento apretado del pecho.", "Como que no puedo respirar bien, me ahogo."], ["me siento apretado"]),
        ("Estoy mamado", "green", [_SALUDO, "Uy, estoy todo mamado y aporreado.", "Pero el dolor es bajito, como un 2, y con la droga se calma."], ["estoy mamado"]),
        ("Guayabo del bravo", "green", [_SALUDO, "Amanecí con un guayabo del bravo, mareado.", "Pero sin fiebre y con dolor leve, un 2."], ["guayabo del bravo"]),
    ]
    out = []
    for name, risk, msgs, interp in items:
        exp = {
            "risk": risk,
            "should_escalate": risk in ("yellow", "red"),
            "must_interpret": interp,
        }
        out.append(_scenario(name, risk, "colombian_language", msgs, exp))
    return out


GENERATORS = {
    "green": green,
    "yellow": yellow,
    "red": red,
    "emotional": emotional,
    "memory": memory,
    "edge_cases": edge_cases,
    "colombian_language": colombian_language,
}


def generate(root: Path = SCENARIOS_DIR, *, clean: bool = False) -> int:
    """Genera todos los escenarios en disco. Devuelve el total escrito."""
    total = 0
    for category, gen in GENERATORS.items():
        cat_dir = root / category
        cat_dir.mkdir(parents=True, exist_ok=True)
        if clean:
            for old in cat_dir.glob("*.json"):
                old.unlink()
        for scenario in gen():
            path = cat_dir / f"{_slug(scenario['name'])}.json"
            path.write_text(
                json.dumps(scenario, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            total += 1
    return total


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Genera los escenarios JSON de prueba.")
    p.add_argument("--clean", action="store_true", help="Borra los .json previos antes de generar.")
    args = p.parse_args(argv)
    total = generate(clean=args.clean)
    print(f"✅ {total} escenarios generados en {SCENARIOS_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

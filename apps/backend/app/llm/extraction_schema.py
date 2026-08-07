"""Esquemas JSON para la extracción restringida por gramática de Ollama.

Se escriben **a mano**, no con `Symptoms.model_json_schema()`. Medido el 7-ago-2026
(ver docs/spikes-7-agosto.md):

  - La gramática de Ollama se atraganta con `anyOf`/`null`, que es justo lo que
    genera Pydantic para un `int | None`. Por eso todo campo es un enum de strings
    con el centinela "no_dice" en lugar de null, y se convierte a None después.
  - La latencia es esencialmente el número de tokens de salida (40 tok/s en un
    M1 Pro). Un esquema con los seis slots cuesta ~100 tokens y 2.1 s; uno con el
    slot que se está preguntando cuesta ~11 tokens y 323 ms.
  - Los nombres de campo de una letra recortan otro ~20% de tokens sin perder
    precisión, porque el esquema ya dice qué significa cada uno.

El spine determinista sabe qué slot está preguntando, así que pide solo ese.
"""
from __future__ import annotations

from typing import Any

# --- vocabularios, literales del dataset --------------------------------------

_DOLOR = {"type": "string", "enum": [*(str(i) for i in range(11)), "no_dice"]}
_FIEBRE = {"type": "string", "enum": ["si", "no", "no_sabe", "no_dice"]}
_MOVILIDAD = {"type": "string",
              "enum": ["normal", "limitada_esperada", "incapacitante_nueva", "no_dice"]}
_HERIDA = {"type": "string",
           "enum": ["normal", "eritema_leve", "secrecion_purulenta", "no_dice"]}
_APETITO = {"type": "string",
            "enum": ["normal", "levemente_disminuido", "muy_disminuido", "no_dice"]}
_SUENO = {"type": "string",
          "enum": ["normal", "levemente_alterado", "muy_alterado", "no_dice"]}

# Las banderas de emergencia TAMPOCO se le piden al modelo. Se midió: al añadir el
# campo al esquema, llama3.2:3b marcó `no_puede_respirar` en "como un 7, la pastilla
# no me lo quita" —un falso positivo de emergencia— y de paso dejó de extraer el
# dolor. Viven en app/nlu/lexicon.py como regex sobre frases formulaicas, que es
# exactamente lo que la gente dice cuando algo va mal en serio.

# La intención NO se le pide al modelo. Se midió y salió mal: llama3.2:3b marcaba
# "Un 3, apenas se nota" como fuera_de_mision, porque la instrucción de vigilar
# órdenes se le sobre-dispara. Vive en app/nlu/intent.py como reglas, que además
# es lo único defendible para la inyección: el detector no puede ser la misma
# pieza que el atacante intenta manipular.

# Campo corto -> (campo de Symptoms, esquema). El nombre corto es lo que ve el
# modelo; el largo es el del contrato Pydantic.
SLOT_FIELDS: dict[str, tuple[str, dict]] = {
    "dolor": ("pain_level", _DOLOR),
    "fiebre": ("fever", _FIEBRE),
    "movilidad": ("mobility", _MOVILIDAD),
    "herida": ("wound", _HERIDA),
    "apetito": ("appetite", _APETITO),
    "sueno": ("sleep", _SUENO),
}

# Clave corta que se usa en el JSON para el valor del slot.
SLOT_KEY = "v"


def schema_for_slot(slot: str) -> dict[str, Any]:
    """Esquema mínimo para el turno que pregunta por `slot`.

    Solo el slot: es la variante medida más rápida (323 ms, 11 tokens) y la más
    precisa (6/6). Todo lo demás —banderas, intención, temperatura— lo resuelve
    código determinista, que además no puede ser manipulado por el paciente.
    """
    if slot not in SLOT_FIELDS:
        raise ValueError(f"slot desconocido: {slot!r}")
    props: dict[str, Any] = {SLOT_KEY: SLOT_FIELDS[slot][1]}
    required = [SLOT_KEY]
    return {"type": "object", "required": required, "properties": props}


# Turnos fuera del guion (apertura, cierre, pregunta suelta) no llaman al LLM para
# extraer: no hay slot que rellenar y el léxico ya cubre banderas y temperatura.

"""Fusión de síntomas con precedencia por severidad.

Se usa en dos sitios: para combinar lo que dijo el léxico con lo que dijo el LLM
dentro de un turno, y para acumular a lo largo de la llamada. La regla es siempre
la misma y es deliberadamente asimétrica:

**nada puede reducir el riesgo.** Una bandera encendida no se apaga, un valor
categórico no baja de severidad, y un dolor no disminuye. El paciente minimizador
—que es un estilo real y frecuente en el dataset— dice primero "me duele un
resto" y dos turnos después "no, si estoy bien". El primero es el que cuenta.
"""
from __future__ import annotations

from ..schemas import Symptoms

# Escalas ordenadas de menor a mayor severidad. Al fusionar gana el índice alto.
_ESCALAS: dict[str, list[str]] = {
    "mobility": ["normal", "limitada_esperada", "incapacitante_nueva"],
    "wound": ["normal", "eritema_leve", "secrecion_purulenta"],
    "appetite": ["normal", "levemente_disminuido", "muy_disminuido"],
    "sleep": ["normal", "levemente_alterado", "muy_alterado"],
}

# Banderas de emergencia: se combinan con OR. Un True de cualquier fuente gana y
# ningún False posterior lo apaga.
_BANDERAS = ("heavy_bleeding", "breathing_difficulty", "chest_pain",
             "loss_of_consciousness", "seizure", "altered_mental_status")


def merge_symptoms(base: Symptoms, nuevo: Symptoms) -> Symptoms:
    """Combina dos lecturas. `base` tiene prioridad en caso de empate."""
    out = base.model_copy(deep=True)

    for campo in _BANDERAS:
        if getattr(nuevo, campo) is True:
            setattr(out, campo, True)
            out.sources.setdefault(campo, nuevo.sources.get(campo, "llm"))

    for campo, escala in _ESCALAS.items():
        a, b = getattr(out, campo), getattr(nuevo, campo)
        if b is None:
            continue
        if a is None or escala.index(b) > escala.index(a):
            setattr(out, campo, b)
            out.sources[campo] = nuevo.sources.get(campo, "llm")

    if nuevo.pain_level is not None:
        if out.pain_level is None or nuevo.pain_level > out.pain_level:
            out.pain_level = nuevo.pain_level
            out.sources["pain_level"] = nuevo.sources.get("pain_level", "llm")

    if nuevo.temperature_c is not None:
        if out.temperature_c is None or nuevo.temperature_c > out.temperature_c:
            out.temperature_c = nuevo.temperature_c
            out.sources["temperature_c"] = nuevo.sources.get("temperature_c", "llm")

    # Fiebre: un sí no se convierte en no. Y una temperatura alta implica fiebre
    # aunque el paciente lo niegue, porque el termómetro no opina.
    if nuevo.fever is True or (out.temperature_c or 0) >= 37.5:
        out.fever = True
    elif out.fever is None and nuevo.fever is not None:
        out.fever = nuevo.fever

    # No es un hallazgo clínico sino estado de la conversación —dice si hay que
    # perseguir la cifra—, así que aquí manda lo último que dijo el paciente y no
    # la lectura más grave: "me la tomé" y luego "no, no tengo termómetro" es una
    # corrección, no una minimización. El riesgo lo siguen llevando `fever` y
    # `temperature_c`, que sí son monótonos.
    if nuevo.temperature_measured is not None:
        out.temperature_measured = nuevo.temperature_measured

    # Medicación inefectiva es la lectura conservadora: agrava el cuadro de dolor.
    if nuevo.medication_effective is False:
        out.medication_effective = False
    elif out.medication_effective is None:
        out.medication_effective = nuevo.medication_effective

    if nuevo.day_postop is not None:
        out.day_postop = nuevo.day_postop

    for k, v in nuevo.sources.items():
        out.sources.setdefault(k, v)
    for texto in nuevo.other:
        if texto not in out.other:
            out.other.append(texto)

    return out

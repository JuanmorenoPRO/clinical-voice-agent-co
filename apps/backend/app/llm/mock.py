"""Adaptador Mock determinista (sin API key).

Hace posible probar TODO el bucle de conversación (RAG + decisión + resumen +
trazabilidad) sin credenciales de LLM. Parsea pistas simples del texto en español
("dolor 9", "fiebre de 39", "sangrando mucho", "no puedo respirar") para producir
síntomas plausibles. No es un modelo real — es un doble de pruebas.
"""
from __future__ import annotations

import re

from ..schemas import LLMTurnOutput, Symptoms

_NUM = r"(\d{1,2}(?:[.,]\d)?)"


def _to_float(s: str) -> float:
    return float(s.replace(",", "."))


def extract_symptoms_heuristic(text: str) -> Symptoms:
    """Extracción por reglas simples (stand-in del LLM real)."""
    t = text.lower()
    sym = Symptoms()

    m = re.search(rf"dolor[^\d]{{0,15}}{_NUM}", t)
    if m:
        sym.pain_level = int(round(_to_float(m.group(1))))
    elif any(w in t for w in ("mucho dolor", "dolor fuerte", "duele mucho")):
        sym.pain_level = 9

    if any(w in t for w in ("no me hace", "no me quita", "no funciona", "no me sirve", "sigue el dolor")):
        sym.medication_effective = False
    elif any(w in t for w in ("me calma", "me quita el dolor", "me hace efecto")):
        sym.medication_effective = True

    m = re.search(rf"(?:fiebre|temperatura)[^\d]{{0,15}}{_NUM}", t)
    if m:
        sym.temperature_c = _to_float(m.group(1))
        sym.fever = sym.temperature_c >= 38.0
    elif "fiebre" in t or "calentura" in t:
        sym.fever = True

    if any(w in t for w in ("sangr", "sangre", "hemorrag")):
        sym.heavy_bleeding = any(w in t for w in ("mucho", "abundante", "no para", "chorro"))

    if any(w in t for w in ("no puedo respirar", "me ahogo", "falta el aire", "dificultad para respirar")):
        sym.breathing_difficulty = True

    if any(w in t for w in ("me desmay", "perdí el conocimiento", "inconsciente")):
        sym.loss_of_consciousness = True

    if "maluco" in t or "revuelto" in t:
        sym.other.append(text.strip())

    return sym


class MockAdapter:
    def turn(self, *, system_prompt: str, user_prompt: str) -> LLMTurnOutput:
        # El user_prompt del ConversationService trae la evidencia RAG + el turno.
        # Para el mock, extraemos síntomas de la última línea "Paciente: ...".
        patient_line = user_prompt
        for line in reversed(user_prompt.splitlines()):
            if line.lower().startswith("paciente:"):
                patient_line = line.split(":", 1)[1].strip()
                break

        symptoms = extract_symptoms_heuristic(patient_line)
        respuesta = (
            "Gracias por contarme cómo se siente. Voy a registrar lo que me dice "
            "y seguimos con el chequeo. ¿Hay algo más que quiera comentarme?"
        )
        return LLMTurnOutput(sintomas=symptoms, respuesta=respuesta)

    def summarize(self, *, system_prompt: str, user_prompt: str) -> str:
        return "Resumen generado por MockAdapter (sin LLM real)."

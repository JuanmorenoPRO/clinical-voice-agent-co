"""Adaptador Anthropic — implementación PROVISIONAL hasta el 7 de agosto.

Los premios del reto son créditos Claude, por lo que Anthropic es el proveedor
probable — pero ADR-002 exige que nada fuera de este archivo lo asuma. Cuando se
anuncie el modelo obligatorio, se agrega su adaptador junto a este y se cambia
LLM_PROVIDER.

Usa salida estructurada (`output_config.format`) para forzar el esquema
{sintomas, respuesta} en una sola llamada por turno (ADR-006). El modelo por
defecto (`claude-opus-4-8`) es configurable; para el presupuesto de latencia de
voz (<1.5s, RNF-02) puede convenir un modelo más rápido vía ANTHROPIC_MODEL.
"""
from __future__ import annotations

import json

import anthropic

from ..config import get_settings
from ..schemas import LLMTurnOutput

# Esquema JSON de la salida estructurada del turno. Coincide con LLMTurnOutput.
_TURN_SCHEMA = {
    "type": "object",
    "properties": {
        "sintomas": {
            "type": "object",
            "properties": {
                "pain_level": {"type": ["integer", "null"]},
                "medication_effective": {"type": ["boolean", "null"]},
                "temperature_c": {"type": ["number", "null"]},
                "fever": {"type": ["boolean", "null"]},
                "heavy_bleeding": {"type": ["boolean", "null"]},
                "breathing_difficulty": {"type": ["boolean", "null"]},
                "loss_of_consciousness": {"type": ["boolean", "null"]},
                "other": {"type": "array", "items": {"type": "string"}},
            },
            "required": [
                "pain_level",
                "medication_effective",
                "temperature_c",
                "fever",
                "heavy_bleeding",
                "breathing_difficulty",
                "loss_of_consciousness",
                "other",
            ],
            "additionalProperties": False,
        },
        "respuesta": {"type": "string"},
    },
    "required": ["sintomas", "respuesta"],
    "additionalProperties": False,
}


class AnthropicAdapter:
    def __init__(self) -> None:
        settings = get_settings()
        self._client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        self._model = settings.anthropic_model

    def turn(self, *, system_prompt: str, user_prompt: str) -> LLMTurnOutput:
        response = self._client.messages.create(
            model=self._model,
            max_tokens=1024,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
            output_config={"format": {"type": "json_schema", "schema": _TURN_SCHEMA}},
        )
        text = next(b.text for b in response.content if b.type == "text")
        return LLMTurnOutput.model_validate(json.loads(text))

    def summarize(self, *, system_prompt: str, user_prompt: str) -> str:
        response = self._client.messages.create(
            model=self._model,
            max_tokens=1024,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return next(b.text for b in response.content if b.type == "text")

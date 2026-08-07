"""Adaptador Anthropic — implementación PROVISIONAL hasta el 7 de agosto.

Los premios del reto son créditos Claude, por lo que Anthropic es el proveedor
probable — pero ADR-002 exige que nada fuera de este archivo lo asuma. Cuando se
anuncie el modelo obligatorio, se agrega su adaptador junto a este y se cambia
LLM_PROVIDER.

La salida estructurada {sintomas, respuesta} (una sola llamada por turno, ADR-006)
se obtiene pidiendo JSON en el prompt y validándolo contra el esquema, con
reintento si el modelo no cumple. Es la ruta de degradación que prevé ADR-002 y
funciona en cualquier versión del SDK `anthropic` (no depende de `output_config`,
que solo existe en SDKs recientes). El modelo por defecto (`claude-opus-4-8`) es
configurable; para el presupuesto de latencia de voz (<1.5s, RNF-02) puede
convenir un modelo más rápido vía ANTHROPIC_MODEL.
"""
from __future__ import annotations

import json

import anthropic

from ..config import get_settings
from ..schemas import LLMTurnOutput

# Instrucción de formato que se añade al system prompt. Coincide con LLMTurnOutput.
_JSON_INSTRUCTION = """
Responde ÚNICAMENTE con un objeto JSON válido (sin texto adicional, sin ```),
con exactamente esta forma:
{
  "sintomas": {
    "pain_level": <entero 0-10 o null>,
    "medication_effective": <true/false o null>,
    "temperature_c": <número o null>,
    "fever": <true/false o null>,
    "heavy_bleeding": <true/false o null>,
    "breathing_difficulty": <true/false o null>,
    "loss_of_consciousness": <true/false o null>,
    "chest_pain": <true/false o null>,
    "altered_mental_status": <true/false o null>,
    "seizure": <true/false o null>,
    "other": [<strings>]
  },
  "respuesta": "<lo que le dirías al paciente en voz alta>"
}
Deja en null lo que el paciente no mencionó; no inventes valores.
"""


def _extract_text(response) -> str:
    return next(b.text for b in response.content if b.type == "text")


def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        # quita ```json ... ``` si el modelo los añade igualmente
        text = text.split("```", 2)[1]
        if text.lstrip().lower().startswith("json"):
            text = text.lstrip()[4:]
    return text.strip("` \n")


class AnthropicAdapter:
    def __init__(self) -> None:
        settings = get_settings()
        self._client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        self._model = settings.anthropic_model

    def _complete(self, *, system_prompt: str, user_prompt: str, max_tokens: int = 1024) -> str:
        response = self._client.messages.create(
            model=self._model,
            max_tokens=max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return _extract_text(response)

    def turn(self, *, system_prompt: str, user_prompt: str) -> LLMTurnOutput:
        system = system_prompt + "\n\n" + _JSON_INSTRUCTION
        raw = self._complete(system_prompt=system, user_prompt=user_prompt)
        try:
            return LLMTurnOutput.model_validate(json.loads(_strip_fences(raw)))
        except (json.JSONDecodeError, ValueError):
            # Reintento con corrección explícita (ADR-002).
            retry_prompt = (
                f"{user_prompt}\n\nTu respuesta anterior no fue JSON válido con el "
                "esquema pedido. Devuelve SOLO el objeto JSON."
            )
            raw = self._complete(system_prompt=system, user_prompt=retry_prompt)
            return LLMTurnOutput.model_validate(json.loads(_strip_fences(raw)))

    def summarize(self, *, system_prompt: str, user_prompt: str) -> str:
        return self._complete(system_prompt=system_prompt, user_prompt=user_prompt)

"""Selección del adaptador LLM según LLM_PROVIDER (ADR-002, ADR-010).

`groq` es el proveedor de producción: el sucesor vigente de Llama en Groq
(`llama-3.3-70b-versatile` hoy, la familia Llama 4 tras el apagado del 16-ago-2026),
uno de los puestos que permite la compuerta G3. `ollama` es la alternativa local
(`llama3.2:3b`). `mock` es determinista, no necesita modelo y existe para que los
tests corran sin credenciales ni Ollama levantado.

Un proveedor desconocido **falla ruidosamente**. Antes caía a mock en silencio, y
un typo en la variable de entorno podía dejar la demo corriendo con el extractor
de juguete sin que nadie se enterara.
"""

from __future__ import annotations

from functools import lru_cache

from ..config import get_settings
from .adapter import LLMAdapter

_PROVIDERS = ("groq", "ollama", "mock")


@lru_cache
def get_llm() -> LLMAdapter:
    provider = get_settings().llm_provider.lower()

    if provider == "groq":
        from .groq_adapter import GroqAdapter

        return GroqAdapter()

    if provider == "ollama":
        from .ollama_adapter import OllamaAdapter

        return OllamaAdapter()

    if provider == "mock":
        from .mock import MockAdapter

        return MockAdapter()

    raise ValueError(
        f"LLM_PROVIDER={provider!r} no reconocido. Válidos: {', '.join(_PROVIDERS)}. "
        "El modelo del agente debe ser uno de los permitidos por el reto (G3)."
    )

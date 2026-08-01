"""Selección del adaptador LLM según LLM_PROVIDER (ADR-002)."""
from __future__ import annotations

from functools import lru_cache

from ..config import get_settings
from .adapter import LLMAdapter


@lru_cache
def get_llm() -> LLMAdapter:
    provider = get_settings().llm_provider.lower()

    if provider == "anthropic":
        from .anthropic_adapter import AnthropicAdapter

        return AnthropicAdapter()

    # ⏳ 7 ago: registrar aquí el adaptador del modelo obligatorio y hacerlo
    #    seleccionable con LLM_PROVIDER (ADR-010).

    from .mock import MockAdapter

    return MockAdapter()

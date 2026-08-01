"""Carga de plantillas de prompts versionadas desde /prompts (RNF-05).

Los prompts nunca se hardcodean; viven como plantillas fuera del código para
poder versionarlos e iterarlos con el vocabulario del dataset (7 de agosto).
"""
from __future__ import annotations

import os
from functools import lru_cache

from .config import get_settings


@lru_cache
def load_prompt(name: str) -> str:
    path = os.path.join(get_settings().prompts_dir, f"{name}.md")
    with open(path, encoding="utf-8") as fh:
        return fh.read()

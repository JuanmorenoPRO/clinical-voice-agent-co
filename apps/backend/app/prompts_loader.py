"""Carga de plantillas de prompts versionadas desde /prompts (RNF-05).

Los prompts del redactor nunca se hardcodean; viven como archivos fuera del
código para poder versionarlos e iterarlos sin desplegar. Es el mismo patrón de
la primera versión del agente, recuperado junto con la llamada de redacción.
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

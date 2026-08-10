"""Configuración común de la suite.

El redactor (compose_reply) se fuerza a `mock` ANTES de que cualquier módulo
lea la configuración: su compose_reply devuelve None y todos los turnos salen
por las plantillas deterministas. Sin esto, cada turno conversacional de la
suite haría una llamada real a Groq — lenta, no determinista, y quemando la
cuota diaria del reto en tests.

La extracción (`extract`) no se toca: las frases de los tests las resuelve el
léxico sin llegar al modelo, y los tests que necesitan un LLM concreto ya
sustituyen `get_llm` con monkeypatch.
"""
from __future__ import annotations

import os

os.environ.setdefault("COMPOSE_PROVIDER", "mock")

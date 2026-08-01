"""Configuración central — lee variables de entorno (.env).

Única fuente de configuración del backend. Todo lo que es "bloqueado hasta el
7 de agosto" (modelo LLM, dataset Delta Share) se controla desde aquí para que
ajustarlo sea cambiar una variable, no reescribir código (ADR-002).
"""
from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- Base de datos (ADR-004: PostgreSQL + pgvector, única BD) ---
    database_url: str = "postgresql+psycopg://clinical:clinical@localhost:5432/clinical"

    # --- LLM (ADR-002: adaptador intercambiable) ---
    # mock  -> determinista, sin API key (por defecto, corre sin credenciales)
    # anthropic -> implementación provisional hasta que se anuncie el modelo (7 ago)
    llm_provider: str = "mock"
    anthropic_api_key: str | None = None
    # Default mandado por el skill claude-api. Para el presupuesto de latencia de
    # voz (<1.5s, RNF-02) se puede cambiar a un modelo más rápido vía esta variable.
    anthropic_model: str = "claude-opus-4-8"

    # --- Embeddings (RAG). Voyage AI para simular Databricks en el test ---
    # ⏳ 7 ago: confirmar el modelo definitivo según el LLM obligatorio (ADR-011).
    voyage_api_key: str | None = None
    embedding_model: str = "voyage-3"
    embedding_dim: int = 1024  # voyage-3 = 1024. Configurable: el modelo puede cambiar el 7 ago.

    # --- Voz (Pipecat). Necesarias solo para el modo de voz real ---
    deepgram_api_key: str | None = None
    elevenlabs_api_key: str | None = None
    elevenlabs_voice_id: str | None = None  # ⏳ elegir voz nativa es-CO (ADR-007)
    elevenlabs_model: str = "eleven_flash_v2_5"

    # --- RAG: umbral de confianza para "no tengo evidencia suficiente" (ADR-005) ---
    rag_top_k: int = 4
    rag_min_confidence: float = 0.55

    # --- Chunking ---
    chunk_size: int = 800
    chunk_overlap: int = 150

    # --- Prompts versionados fuera del código (/prompts, RNF-05) ---
    prompts_dir: str = "prompts"

    # --- Datos de ejemplo (stand-in de Delta Share hasta el 7 de agosto) ---
    samples_dir: str = "data/samples"
    # ⏳ 7 ago: credenciales Delta Share (ADR-012). Hoy no se usan.
    databricks_share_profile: str | None = None


@lru_cache
def get_settings() -> Settings:
    return Settings()

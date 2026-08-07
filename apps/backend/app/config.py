"""Configuración central — lee variables de entorno (.env).

Única fuente de configuración del backend. Cambiar de modelo, de voz o de umbral
del RAG es cambiar una variable, no reescribir código (ADR-002).
"""
from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- Estado de la conversación (ADR-014: SQLite, sin servidor) ---
    # Se abandonó PostgreSQL+pgvector: los vectores viven en Chroma y quitar el
    # servidor de base de datos elimina Docker de la ruta crítica del arranque,
    # que es donde más aprieta la compuerta de 15 minutos.
    database_url: str = "sqlite:///./data/clinical.db"

    # --- LLM (ADR-002 / ADR-010) — compuerta G3 ---
    # ollama -> llama3.2:3b, el modelo del agente. Está en la lista permitida.
    # mock   -> extractor determinista, para tests sin Ollama levantado.
    llm_provider: str = "ollama"
    llm_model: str = "llama3.2:3b"
    ollama_host: str = "http://localhost:11434"
    # Si el modelo tarda más que esto, el turno se completa con el léxico
    # determinista y se marca degradado. Medido: la extracción va en ~325 ms.
    llm_timeout_s: float = 2.5
    # Mantiene el modelo en RAM entre turnos. Sin esto, el primer turno tras una
    # pausa cuesta ~24 s de recarga.
    llm_keep_alive: str = "60m"

    # --- Embeddings (ADR-011): bge-m3 servido por el mismo Ollama ---
    # Mismo modelo multilingüe elegido, cuantizado a 1.2 GB, 1024 dims, 8K ctx.
    # Reutilizar el runtime de Ollama evita añadir fastembed/onnxruntime aparte.
    embedding_model: str = "bge-m3"
    embedding_dim: int = 1024

    # --- Voz ---
    # STT: Groq Whisper. G3 restringe el modelo que razona, no el reconocimiento
    # de voz, así que usarlo no compromete la compuerta.
    groq_api_key: str | None = None
    stt_model: str = "whisper-large-v3-turbo"
    # TTS: Kokoro local. Voces en español: ef_dora (F), em_alex (M), em_santa (M).
    tts_voice: str = "ef_dora"
    # Detección de fin de turno. 0.7 s protege al paciente mayor de ser cortado a
    # media frase; es la palanca dominante del presupuesto de latencia.
    vad_stop_secs: float = 0.7

    # --- RAG: umbral de "no tengo evidencia suficiente" (ADR-005) ---
    rag_top_k: int = 4
    # Se sobre-recuperan candidatos y luego se filtran de forma determinista
    # (fuera de alcance, duplicados, máximo por documento) hasta dejar top_k.
    rag_fetch_k: int = 12
    # ⚠️ Calibrar con scripts/calibrate_rag.py. El 0.55 anterior venía de voyage-3
    # y no transfiere: con bge-m3 las similitudes observadas van de 0.75 a 0.79,
    # así que 0.55 dejaría pasar cualquier cosa.
    rag_min_confidence: float = 0.55
    chroma_dir: str = "data/chroma"
    chroma_collection: str = "clinical_knowledge"

    # --- Chunking (debe coincidir con scripts/build_index.py) ---
    chunk_size: int = 900
    chunk_overlap: int = 180

    # --- Prompts y datos versionados fuera del código (RNF-05) ---
    prompts_dir: str = "prompts"
    seed_dir: str = "data/seed"


@lru_cache
def get_settings() -> Settings:
    return Settings()

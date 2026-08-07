"""Configuración central — lee variables de entorno (.env).

Única fuente de configuración del backend. Cambiar de modelo, de voz o de umbral
del RAG es cambiar una variable, no reescribir código (ADR-002).
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# Raíz del repositorio: .../apps/backend/app/config.py -> subir tres niveles.
# Todas las rutas de datos se anclan aquí y no al directorio de trabajo. Sin esto,
# arrancar el servidor desde apps/backend/ crea una base de datos vacía y un
# índice inexistente **en silencio**: la app responde 200 y devuelve cero
# documentos, que es la peor forma de fallar en una demo cronometrada.
PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _abs(ruta: str) -> str:
    """Convierte una ruta relativa en absoluta respecto a la raíz del proyecto."""
    p = Path(ruta)
    return str(p if p.is_absolute() else (PROJECT_ROOT / p))


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=PROJECT_ROOT / ".env", extra="ignore")

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
    # La respuesta anclada al RAG procesa evidencia y genera dos frases: es la
    # ruta lenta del sistema y necesita más margen que la extracción de un slot.
    llm_reply_timeout_s: float = 12.0
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

    # TTS: piper (por defecto) o kokoro.
    #
    # Se probaron ambos con la misma frase clínica, sintetizados y verificados
    # transcribiéndolos de vuelta con Whisper (docs/spikes-7-agosto.md). Kokoro es
    # un modelo centrado en inglés que cubre español por fonemización de
    # respaldo (espeak-ng): suena a acento anglosajón hablando español, no a
    # español latino. Piper tiene modelos entrenados nativamente por idioma;
    # `es_MX-claude-high` resultó además 5 veces más rápido en caliente
    # (RTF 0.05 frente al 0.24 de Kokoro) y es el acento que más se reconoce
    # como "neutro latinoamericano" a un oído colombiano frente a las
    # alternativas del catálogo (es_ES peninsular, es_AR con voseo marcado).
    #
    # `piper-tts` es GPL-3.0 (ver LICENSE-PIPER.md); se declara explícitamente
    # en el informe. No arrastra PyTorch: solo onnxruntime, igual que Kokoro.
    tts_provider: str = "piper"
    # Voces Kokoro en español: ef_dora (F), em_alex (M), em_santa (M).
    # Voces Piper en español: es_MX-claude-high, es_AR-daniela-high, es_ES-*.
    tts_voice: str = "es_MX-claude-high"
    piper_voices_dir: str = "data/piper-voices"

    # Prosodia de Piper. Pipecat sintetiza sin `SynthesisConfig`, así que sin esto
    # se usan los valores por defecto del modelo (`es_MX-claude-high.onnx.json`:
    # length_scale=1.0, noise_scale=0.667, noise_w=0.8), que es lo que suena plano
    # ("robótico"). Subir el ruido y bajar un poco la velocidad da más variación
    # prosódica sin cambiar de voz ni de proveedor. Calibrar de oído con
    # `scripts/spike_voice.py`.
    piper_length_scale: float = 1.08
    piper_noise_scale: float = 0.75
    piper_noise_w_scale: float = 0.9

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

    # --- Datos versionados fuera del código ---
    seed_dir: str = "data/seed"


@lru_cache
def get_settings() -> Settings:
    s = Settings()
    # Se normalizan a absoluto después de leer el entorno, para que funcione igual
    # lanzado desde la raíz, desde apps/backend/ o desde un servicio del sistema.
    s.chroma_dir = _abs(s.chroma_dir)
    s.piper_voices_dir = _abs(s.piper_voices_dir)
    s.seed_dir = _abs(s.seed_dir)
    if s.database_url.startswith("sqlite:///./"):
        s.database_url = "sqlite:///" + _abs(s.database_url.removeprefix("sqlite:///./"))
    return s

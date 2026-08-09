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
    # groq  -> llama-3.3-70b-versatile (sucesor vigente de Llama 3.1 70B en Groq).
    # ollama-> llama3.2:3b, el modelo local del agente. Está en la lista permitida.
    # mock  -> extractor determinista, para tests sin Ollama levantado.
    llm_provider: str = "groq"
    llm_model: str = "llama-3.3-70b-versatile"
    # Proveedor de la llamada de REDACCIÓN (compose_reply): quién escribe lo que
    # el agente dice en voz alta, con el guion como restricción. Vacío = el mismo
    # de `llm_provider`. Separado para poder comparar redactores sin tocar la
    # extracción.
    compose_provider: str = ""
    ollama_host: str = "http://localhost:11434"
    # Si el modelo tarda más que esto, el turno se completa con el léxico
    # determinista y se marca degradado. Medido en Ollama: la extracción iba en
    # ~325 ms; en Groq (hosted) el margen sobra.
    llm_timeout_s: float = 2.5
    # La respuesta anclada al RAG procesa evidencia y genera dos frases: es la
    # ruta lenta del sistema y necesita más margen que la extracción de un slot.
    llm_reply_timeout_s: float = 12.0
    # Redacción del turno (compose_reply). Más corto que el de reply: si el
    # redactor tarda, el turno cae a las plantillas deterministas y la llamada
    # no se queda esperando. Groq responde en ~1 s; 6 s es el margen de un mal día.
    llm_compose_timeout_s: float = 6.0
    # Mantiene el modelo en RAM entre turnos (solo Ollama). Groq lo ignora.
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
    # transcribiéndolos de vuelta con Whisper (docs/spikes-7-agosto.md). Piper
    # entrena un modelo por idioma; `es_MX-claude-high` suena a español
    # latinoamericano neutro, es el acento que más se reconoce a un oído
    # colombiano frente a las alternativas del catálogo (es_ES peninsular,
    # es_AR con voseo marcado) y resultó 5 veces más rápido en caliente
    # (RTF 0.05 frente al 0.24 de Kokoro).
    #
    # Kokoro es un modelo centrado en inglés: por defecto fonemiza en inglés y
    # suena a acento anglosajón hablando español. Para usarlo bien hay que
    # forzar el G2P en español (`Language.ES` en el `KokoroTTSService`), que es
    # el mismo respaldo espeak-ng de misaki que usa `leonelhs/kokoro-tts-spanish`.
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

    # Umbrales con los que el VAD decide que ALGUIEN está hablando. Silero exige
    # los dos a la vez (`vad_analyzer.py`: `confidence >= X and volume >= Y`), así
    # que el más estricto manda. Estaban fijos en el código y ahora se configuran,
    # porque son la primera palanca cuando el agente no oye al paciente.
    #
    # `vad_min_volume` es sonoridad EBU R128 normalizada a [0,1]. El valor por
    # defecto de Pipecat es 0.6 y asume un micrófono cercano y bien nivelado; con
    # el de un portátil, o con el paciente a medio metro, no se alcanza — y
    # entonces no hay VAD, no hay STT (Groq transcribe por segmentos y DEPENDE del
    # VAD), y la llamada entera se interpreta como silencio hasta colgar. Se baja
    # a 0.3, que es el lado seguro: un falso positivo entra como `ininteligible`
    # (ver `nlu/intent.py::_es_ruido_transcripcion`) y el agente pide que se
    # repita; un falso negativo le cuelga a un paciente que sí estaba hablando.
    vad_confidence: float = 0.7
    vad_min_volume: float = 0.3

    # Inactividad tras la que se da por hecho que el paciente no está
    # contestando. Se cuenta desde que el agente TERMINA de hablar, no desde que
    # empieza, y solo corre cuando no habla ninguno de los dos (ver
    # `voice/pipeline.py`). Ocho segundos y no cinco: cinco vencía mientras un
    # paciente mayor todavía estaba pensando la respuesta a una pregunta larga.
    # Con la escalera de `agent/script.py::MAX_SILENCIOS` son 24 s antes de colgar.
    silence_timeout_s: float = 8.0

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
    # Prompts del redactor (RNF-05): viven como archivos versionados en /prompts,
    # no hardcodeados, para poder iterarlos sin tocar código.
    prompts_dir: str = "prompts"


@lru_cache
def get_settings() -> Settings:
    s = Settings()
    # Se normalizan a absoluto después de leer el entorno, para que funcione igual
    # lanzado desde la raíz, desde apps/backend/ o desde un servicio del sistema.
    s.chroma_dir = _abs(s.chroma_dir)
    s.piper_voices_dir = _abs(s.piper_voices_dir)
    s.seed_dir = _abs(s.seed_dir)
    s.prompts_dir = _abs(s.prompts_dir)
    if s.database_url.startswith("sqlite:///./"):
        s.database_url = "sqlite:///" + _abs(
            s.database_url.removeprefix("sqlite:///./")
        )
    return s

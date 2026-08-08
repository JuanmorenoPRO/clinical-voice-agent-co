"""Ida y vuelta de la voz: Kokoro sintetiza español, Groq Whisper lo transcribe.

Valida las dos mitades del ciclo de la compuerta G4 en una sola pasada, y mide la
latencia real de cada etapa con frases del dominio clínico.

    .venv/bin/python scripts/spike_voice.py
"""
from __future__ import annotations

import io
import os
import sys
import time
import wave
from pathlib import Path

# Frases del dominio: las 6 preguntas canónicas del guion y respuestas de paciente
# tomadas del dataset, para medir con lo que de verdad va a circular por la línea.
FRASES = [
    "Buenos días, le llamo del hospital para saber cómo ha seguido después de la cirugía.",
    "¿Cómo ha estado el dolor, en una escala del cero al diez?",
    "¿Cómo está la herida quirúrgica? ¿Hay enrojecimiento, secreción o hinchazón?",
    "Me duele un berraco, doctora, no aguanto.",
    "La herida está botando materia amarilla desde ayer.",
]


def cargar_env() -> None:
    env = Path(".env")
    if not env.exists():
        return
    for linea in env.read_text(encoding="utf-8").splitlines():
        linea = linea.strip()
        if not linea or linea.startswith("#") or "=" not in linea:
            continue
        k, v = linea.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())


def to_wav(samples, rate: int) -> bytes:
    import numpy as np

    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes((np.clip(samples, -1, 1) * 32767).astype(np.int16).tobytes())
    return buf.getvalue()


def main() -> int:
    cargar_env()
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        print("Falta GROQ_API_KEY en .env", file=sys.stderr)
        return 1
    print(f"GROQ_API_KEY cargada (…{key[-4:]})\n")

    import espeakng_loader
    from groq import Groq
    from kokoro_onnx import Kokoro
    from pipecat.services.kokoro.tts import KOKORO_CACHE_DIR

    os.environ.setdefault("ESPEAK_DATA_PATH", str(espeakng_loader.get_data_path()))
    k = Kokoro(str(KOKORO_CACHE_DIR / "kokoro-v1.0.onnx"),
               str(KOKORO_CACHE_DIR / "voices-v1.0.bin"))
    groq = Groq(api_key=key)

    tts_ms, stt_ms, rtf = [], [], []
    print(f"{'frase':<62} {'TTS':>8} {'RTF':>6} {'STT':>8}")
    print("-" * 92)
    for frase in FRASES:
        t0 = time.perf_counter()
        samples, rate = k.create(frase, voice=os.environ.get("TTS_VOICE", "ef_dora"),
                                 speed=1.0, lang="es")
        t_tts = (time.perf_counter() - t0) * 1000
        dur = len(samples) / rate
        wav = to_wav(samples, rate)

        t0 = time.perf_counter()
        tr = groq.audio.transcriptions.create(
            file=("turno.wav", wav), model="whisper-large-v3-turbo", language="es",
            prompt="Llamada de seguimiento postoperatorio con un paciente colombiano. "
                   "Términos: dolor, fiebre, herida, secreción, movilidad, apetito.",
        )
        t_stt = (time.perf_counter() - t0) * 1000

        tts_ms.append(t_tts)
        stt_ms.append(t_stt)
        rtf.append(t_tts / 1000 / dur)
        print(f"{frase[:60]:<62} {t_tts:>6.0f}ms {t_tts/1000/dur:>6.2f} {t_stt:>6.0f}ms")
        print(f"  → {tr.text.strip()}")

    n = len(FRASES)
    print("\n" + "=" * 92)
    print(f"TTS Kokoro : media {sum(tts_ms)/n:>6.0f} ms · factor de tiempo real "
          f"{sum(rtf)/n:.2f} (menor que 1 = más rápido que la reproducción)")
    print(f"STT Groq   : media {sum(stt_ms)/n:>6.0f} ms")
    print(f"\nPresupuesto de voz, sumando lo medido:")
    print(f"  VAD (fin de turno)      700 ms")
    print(f"  STT Groq                {sum(stt_ms)/n:>3.0f} ms")
    print(f"  Léxico + decisión         5 ms")
    print(f"  LLM extracción (0.39x)  ~127 ms  (61% de los turnos no lo usan)")
    print(f"  TTS Kokoro              {sum(tts_ms)/n:>3.0f} ms  (0 ms si la frase está cacheada)")
    print(f"  {'─'*40}")
    print(f"  TOTAL típico            ~{700 + sum(stt_ms)/n + 5 + 127 + sum(tts_ms)/n:.0f} ms")
    return 0


if __name__ == "__main__":
    sys.exit(main())

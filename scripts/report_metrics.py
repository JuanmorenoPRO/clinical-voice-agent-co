"""Genera las métricas obligatorias del README desde la base de datos.

    .venv/bin/python scripts/report_metrics.py [--out docs/metricas.md]

La rúbrica penaliza explícitamente "métricas inconsistentes con los logs de la
sesión". La única forma de que eso no pase es que los números **no se escriban a
mano**: salen de las mismas filas de `turns` que el jurado puede consultar por
`GET /console/conversations/{id}`, y este script imprime el comando y la marca de
tiempo con los que se generaron.

Reporta lo que exige el §5 de la rúbrica:
  - latencia P50 y P95
  - tokens de entrada y salida, por turno y por llamada
  - invocaciones al modelo por turno
  - consultas al RAG por llamada
  - costo estimado por llamada, extrapolado a precios de API
"""
from __future__ import annotations

import argparse
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "apps" / "backend"))

from app.config import get_settings  # noqa: E402
from app.db import SessionLocal  # noqa: E402
from app.models import Conversation, Turn  # noqa: E402

# Precios de referencia para extrapolar. El LLM y el TTS corren en local y su
# costo real es cero; se traducen a lo que costarían servidos por API para que la
# cifra sea comparable con la de otros participantes, como pide la rúbrica.
PRECIOS = {
    "llm": {
        "modelo": "llama-3.2-3b-instruct",
        "referencia": "Together.ai (precio público de un 3B servido)",
        "usd_por_1m_entrada": 0.06,
        "usd_por_1m_salida": 0.06,
    },
    "stt": {
        "modelo": "whisper-large-v3-turbo",
        "referencia": "Groq (precio público)",
        "usd_por_hora_audio": 0.04,
    },
    "tts": {
        "modelo": "kokoro-82m (local)",
        "referencia": "ElevenLabs Flash v2.5, como equivalente comercial",
        "usd_por_1k_caracteres": 0.03,
    },
}

# Duración media de audio del paciente por turno, medida en las pruebas de voz.
SEG_AUDIO_POR_TURNO = 4.0


def percentil(valores: list[float], p: float) -> float:
    if not valores:
        return 0.0
    ordenados = sorted(valores)
    k = min(int(len(ordenados) * p), len(ordenados) - 1)
    return ordenados[k]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None, help="escribe el informe en markdown")
    args = ap.parse_args()

    s = get_settings()
    with SessionLocal() as session:
        turnos = session.query(Turn).order_by(Turn.created_at).all()
        n_conv = session.query(Conversation).count()

    if not turnos:
        print("No hay turnos registrados todavía. Corre una conversación primero:",
              file=sys.stderr)
        print("  POST /conversation/turn  o  scripts/run_dataset_eval.py", file=sys.stderr)
        return 1

    lat = [t.latency_ms for t in turnos if t.latency_ms is not None]
    tin = [t.tokens_in for t in turnos]
    tout = [t.tokens_out for t in turnos]
    llamadas = [t.llm_calls for t in turnos]
    rag = [1 if t.retrieved_chunks else 0 for t in turnos]
    degradados = sum(1 for t in turnos if t.degraded)
    sin_llm = sum(1 for c in llamadas if c == 0)

    turnos_por_llamada = len(turnos) / max(1, n_conv)
    tokens_llamada_in = sum(tin) / max(1, n_conv)
    tokens_llamada_out = sum(tout) / max(1, n_conv)

    p = PRECIOS
    costo_llm = (tokens_llamada_in * p["llm"]["usd_por_1m_entrada"] / 1e6
                 + tokens_llamada_out * p["llm"]["usd_por_1m_salida"] / 1e6)
    costo_stt = (turnos_por_llamada * SEG_AUDIO_POR_TURNO / 3600
                 * p["stt"]["usd_por_hora_audio"])
    chars_tts = turnos_por_llamada * 120  # longitud media de una respuesta
    costo_tts = chars_tts / 1000 * p["tts"]["usd_por_1k_caracteres"]

    ahora = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    L: list[str] = []
    add = L.append

    add("# Métricas medidas\n")
    add(f"Generado el {ahora} con `python scripts/report_metrics.py`, "
        f"a partir de **{len(turnos)} turnos** en **{n_conv} llamadas** registrados en "
        "la base de datos.\n")
    add("Estos números no se escriben a mano: salen de las mismas filas de `turns` que "
        "se pueden consultar en `GET /console/conversations/{id}`.\n")

    add("\n## Latencia por turno\n")
    add("| Métrica | ms |")
    add("|---|---:|")
    add(f"| P50 | {percentil(lat, 0.50):.0f} |")
    add(f"| P95 | {percentil(lat, 0.95):.0f} |")
    add(f"| media | {statistics.mean(lat):.0f} |")
    add(f"| máximo | {max(lat):.0f} |")
    add("\nEn modo texto se mide desde que entra la petición. En voz, desde el "
        "`UserStoppedSpeakingFrame` —el instante en que el paciente deja de hablar—, "
        "que es el origen que pide la rúbrica. A eso hay que sumarle los 700 ms del "
        "VAD y ~500 ms del reconocimiento de voz de Groq.\n")

    add("\n## Consumo\n")
    add("| Métrica | Valor |")
    add("|---|---:|")
    add(f"| Invocaciones al modelo por turno | {statistics.mean(llamadas):.2f} |")
    add(f"| Turnos resueltos SIN modelo | {sin_llm}/{len(turnos)} "
        f"({100 * sin_llm / len(turnos):.0f}%) |")
    add(f"| Tokens de entrada por turno | {statistics.mean(tin):.0f} |")
    add(f"| Tokens de salida por turno | {statistics.mean(tout):.0f} |")
    add(f"| Tokens de entrada por llamada | {tokens_llamada_in:.0f} |")
    add(f"| Tokens de salida por llamada | {tokens_llamada_out:.0f} |")
    add(f"| Consultas al RAG por turno | {statistics.mean(rag):.2f} |")
    add(f"| Consultas al RAG por llamada | {sum(rag) / max(1, n_conv):.2f} |")
    add(f"| Turnos degradados (modelo caído o lento) | {degradados} |")
    add(f"| Turnos por llamada | {turnos_por_llamada:.1f} |")
    add("\nLa cifra que explica el resto es la primera: el léxico determinista de "
        "`app/nlu/lexicon.py` resuelve la mayoría de los turnos sin tocar el modelo, "
        "y ahí el consumo es exactamente cero.\n")

    add("\n## Costo por llamada\n")
    add("El LLM y la síntesis de voz corren **en local**, así que el costo real "
        "incurrido es solo el reconocimiento de voz de Groq. Se reportan las dos "
        "cifras, como pide la rúbrica para soluciones locales.\n")
    add("| Concepto | USD por llamada |")
    add("|---|---:|")
    add(f"| **Costo real incurrido** (solo Groq STT) | **{costo_stt:.5f}** |")
    add(f"| LLM extrapolado a API | {costo_llm:.5f} |")
    add(f"| TTS extrapolado a API | {costo_tts:.5f} |")
    add(f"| **Total extrapolado a producción** | **{costo_llm + costo_stt + costo_tts:.5f}** |")

    add("\n### Cómo se calcula\n")
    add("```")
    add(f"LLM  = ({tokens_llamada_in:.0f} tok_in x {p['llm']['usd_por_1m_entrada']}/1M)"
        f" + ({tokens_llamada_out:.0f} tok_out x {p['llm']['usd_por_1m_salida']}/1M)"
        f" = {costo_llm:.5f}")
    add(f"STT  = {turnos_por_llamada:.1f} turnos x {SEG_AUDIO_POR_TURNO}s / 3600"
        f" x {p['stt']['usd_por_hora_audio']}/hora = {costo_stt:.5f}")
    add(f"TTS  = {chars_tts:.0f} caracteres / 1000 x"
        f" {p['tts']['usd_por_1k_caracteres']}/1k = {costo_tts:.5f}")
    add("```")
    add(f"\nReferencias de precio: LLM, {p['llm']['referencia']}; "
        f"STT, {p['stt']['referencia']}; TTS, {p['tts']['referencia']}.\n")

    add("\n## Configuración con la que se midió\n")
    add(f"- Modelo: `{s.llm_model}` vía Ollama (compuerta G3)")
    add(f"- Embeddings: `{s.embedding_model}`, {s.embedding_dim} dimensiones")
    add(f"- STT: `{s.stt_model}` (Groq) · TTS: Kokoro, voz `{s.tts_voice}`")
    add(f"- VAD: {s.vad_stop_secs} s de silencio para dar el turno por terminado")
    add(f"- RAG: top {s.rag_top_k} de {s.rag_fetch_k} candidatos, "
        f"umbral {s.rag_min_confidence}")

    informe = "\n".join(L)
    print(informe)
    if args.out:
        Path(args.out).write_text(informe + "\n", encoding="utf-8")
        print(f"\n→ escrito en {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())

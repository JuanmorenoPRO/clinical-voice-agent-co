"""Corre los 160 casos del dataset a través del agente completo.

    .venv/bin/python scripts/run_dataset_eval.py --capa capa1 --limit 20
    .venv/bin/python scripts/run_dataset_eval.py --capa both --out reports/

Esto NO es lo mismo que `app/tests/test_triage_from_trajectories.py`. Aquel test
alimenta las reglas con el cuadro clínico ya estructurado y mide **solo la
calibración de los umbrales**. Aquí entra la conversación cruda —incluida la capa
ruidosa, con evasivas, silencios y el cuidador interrumpiendo— y se mide la cadena
entera: comprensión del lenguaje, acumulación de síntomas y decisión.

Que estén separados importa: cuando un caso falla, la comparación entre ambos dice
si la culpa es del extractor o de las reglas. Por eso se reporta también la
exactitud por slot.

Sale con código 2 si hay algún falso negativo en rojo, que es la falla que la
rúbrica considera catastrófica.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "apps" / "backend"))

from app.agent.orchestrator import process_turn  # noqa: E402
from app.db import SessionLocal  # noqa: E402
from app.decision import engine  # noqa: E402
from app.models import Conversation, Turn  # noqa: E402
from app.schemas import Symptoms  # noqa: E402

CASOS = RAIZ / "tests" / "dataset" / "cases.jsonl"
DIALOGOS = RAIZ / "tests" / "dataset" / "dialogs.jsonl"

ORDEN = {"verde": 0, "amarillo": 1, "rojo": 2}
COLOR = {"NORMAL": "verde", "ALTO": "amarillo", "CRÍTICO": "rojo"}

# Slots del guion y el campo de `Symptoms` que los guarda, para medir la
# extracción contra el cuadro real de la trayectoria.
SLOTS = {
    "dolor_nrs": "pain_level",
    "movilidad": "mobility",
    "herida": "wound",
    "apetito": "appetite",
    "sueno": "sleep",
}


def cargar(path: Path) -> list[dict]:
    if not path.exists():
        print(f"Falta {path}. Corre scripts/load_dataset.py primero.", file=sys.stderr)
        sys.exit(1)
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def correr_caso(dialogo: dict) -> dict:
    """Reproduce una conversación y devuelve lo que decidió el agente.

    Se alimentan los turnos del paciente **y los del cuidador**: un tercero que
    reporta "está botando materia" tiene que escalar igual. Ignorarlos sería un
    falso negativo de manual.
    """
    t0 = time.perf_counter()
    conv_id = None
    turnos_llm, tokens_in, tokens_out, n = 0, 0, 0, 0

    with SessionLocal() as session:
        for t in dialogo["turnos"]:
            if t["hablante"] == "agente":
                continue
            texto = (t["texto"] or "").strip()
            if not texto:
                continue
            r = process_turn(session, text=texto, conversation_id=conv_id,
                             patient_id=dialogo["paciente_id"])
            conv_id = r.conversation_id
            n += 1

        # Al cerrar se aplica la política de incertidumbre: lo que quedó sin
        # averiguar deja de ser "todavía no lo sé" y pasa a ser un riesgo.
        turnos = session.query(Turn).filter(Turn.conversation_id == conv_id).all()
        acumulado = Symptoms()
        from app.nlu.merge import merge_symptoms
        for tr in turnos:
            acumulado = merge_symptoms(
                acumulado, Symptoms.model_validate(tr.extracted_symptoms or {}))
            turnos_llm += tr.llm_calls
            tokens_in += tr.tokens_in
            tokens_out += tr.tokens_out
        final = engine.evaluate(acumulado, final=True)
        latencias = [tr.latency_ms for tr in turnos if tr.latency_ms is not None]

    return {
        "caso_id": dialogo["caso_id"],
        "capa": dialogo["capa"],
        "esperado": dialogo["label"],
        "predicho": COLOR[final.risk_level],
        "reglas": final.triggered_rules,
        "completitud": final.completeness,
        "sintomas": acumulado.model_dump(),
        "turnos": n,
        "llm_calls": turnos_llm,
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "latencias": latencias,
        "segundos": round(time.perf_counter() - t0, 2),
    }


def informe(resultados: list[dict], casos: dict[str, dict]) -> tuple[str, bool]:
    L: list[str] = []
    add = L.append
    hay_fn_rojo = False

    add("# Evaluación sobre los 160 casos del dataset\n")
    add(f"{len(resultados)} conversaciones reproducidas a través del agente completo "
        "(comprensión del lenguaje, acumulación de síntomas y decisión).\n")

    for capa in sorted({r["capa"] for r in resultados}):
        sub = [r for r in resultados if r["capa"] == capa]
        m = Counter((r["esperado"], r["predicho"]) for r in sub)
        exactos = sum(1 for r in sub if r["esperado"] == r["predicho"])
        fn = [r for r in sub if ORDEN[r["predicho"]] < ORDEN[r["esperado"]]]
        fn_rojo = [r for r in fn if r["esperado"] == "rojo"]
        if fn_rojo:
            hay_fn_rojo = True

        add(f"\n## {capa}  ({len(sub)} casos)\n")
        add("```")
        add("  real \\ predicho   verde  amarillo   rojo")
        for real in ("verde", "amarillo", "rojo"):
            fila = "  ".join(f"{m[(real, p)]:>7}" for p in ("verde", "amarillo", "rojo"))
            add(f"  {real:<16}{fila}")
        add(f"\n  exactitud {exactos}/{len(sub)} ({100 * exactos / len(sub):.1f}%)")
        add("```\n")
        add(f"- **Falsos negativos: {len(fn)}** "
            f"(de los cuales {len(fn_rojo)} en rojo, que es la falla catastrófica)")
        if fn:
            for r in fn[:6]:
                add(f"  - `{r['caso_id']}` esperaba **{r['esperado']}**, "
                    f"dijo *{r['predicho']}* · completitud {r['completitud']}")

    # Exactitud por slot: separa el error del extractor del error de las reglas.
    add("\n## Exactitud de la extracción, por slot\n")
    add("Compara lo que el agente entendió con el cuadro real de la trayectoria. "
        "Si un caso falla, esto dice si la culpa fue de entender al paciente o de "
        "los umbrales.\n")
    add("| Slot | Acierto | Sin extraer |")
    add("|---|---:|---:|")
    for col, campo in SLOTS.items():
        ok = falta = total = 0
        for r in resultados:
            real = casos.get(r["caso_id"], {}).get(col)
            if real is None:
                continue
            total += 1
            got = r["sintomas"].get(campo)
            if got is None:
                falta += 1
            elif str(got) == str(real):
                ok += 1
        if total:
            add(f"| {col} | {100 * ok / total:.0f}% | {100 * falta / total:.0f}% |")

    # Consumo real sobre conversaciones completas: es la fuente honesta para el README.
    lat = [x for r in resultados for x in r["latencias"]]
    add("\n## Consumo por llamada\n")
    add("| Métrica | Valor |")
    add("|---|---:|")
    add(f"| Turnos por llamada | {statistics.mean(r['turnos'] for r in resultados):.1f} |")
    add(f"| Invocaciones al modelo por llamada | "
        f"{statistics.mean(r['llm_calls'] for r in resultados):.1f} |")
    add(f"| Invocaciones al modelo por turno | "
        f"{sum(r['llm_calls'] for r in resultados) / max(1, sum(r['turnos'] for r in resultados)):.2f} |")
    add(f"| Tokens de entrada por llamada | "
        f"{statistics.mean(r['tokens_in'] for r in resultados):.0f} |")
    add(f"| Tokens de salida por llamada | "
        f"{statistics.mean(r['tokens_out'] for r in resultados):.0f} |")
    if lat:
        ordenadas = sorted(lat)
        add(f"| Latencia por turno P50 | {ordenadas[len(ordenadas) // 2]:.0f} ms |")
        add(f"| Latencia por turno P95 | {ordenadas[int(len(ordenadas) * .95)]:.0f} ms |")

    # Por estilo de paciente: dice contra quién falla el agente.
    add("\n## Por estilo de paciente\n")
    add("| Estilo | Casos | Exactitud | Falsos negativos |")
    add("|---|---:|---:|---:|")
    por_estilo: dict[str, list[dict]] = defaultdict(list)
    for r in resultados:
        por_estilo[casos.get(r["caso_id"], {}).get("estilo", "?")].append(r)
    for estilo, sub in sorted(por_estilo.items()):
        ex = sum(1 for r in sub if r["esperado"] == r["predicho"])
        fn = sum(1 for r in sub if ORDEN[r["predicho"]] < ORDEN[r["esperado"]])
        add(f"| {estilo} | {len(sub)} | {100 * ex / len(sub):.0f}% | {fn} |")

    return "\n".join(L), hay_fn_rojo


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--capa", choices=["capa1", "capa2", "both"], default="capa1")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=None, help="directorio donde escribir el informe")
    args = ap.parse_args()

    casos = {c["caso_id"]: c for c in cargar(CASOS)}
    dialogos = cargar(DIALOGOS)
    for d in dialogos:                       # el estilo vive en el diálogo
        casos.setdefault(d["caso_id"], {})["estilo"] = d["estilo_paciente"]

    if args.capa != "both":
        clave = "capa1_limpia" if args.capa == "capa1" else "capa2_ruidosa"
        dialogos = [d for d in dialogos if d["capa"] == clave]
    if args.limit:
        dialogos = dialogos[: args.limit]

    print(f"Reproduciendo {len(dialogos)} conversaciones…", file=sys.stderr)
    resultados = []
    t0 = time.perf_counter()
    for i, d in enumerate(dialogos, 1):
        try:
            r = correr_caso(d)
        except Exception as exc:  # noqa: BLE001
            print(f"  [{i}/{len(dialogos)}] {d['caso_id']} ERROR: {exc}", file=sys.stderr)
            continue
        resultados.append(r)
        marca = "✓" if r["esperado"] == r["predicho"] else (
            "✗FN" if ORDEN[r["predicho"]] < ORDEN[r["esperado"]] else "·FP")
        print(f"  [{i}/{len(dialogos)}] {marca} {r['caso_id'][:34]:<34} "
              f"{r['esperado']:>8} → {r['predicho']:<8} {r['segundos']:>5.1f}s",
              file=sys.stderr)

    if not resultados:
        print("Ningún caso se completó.", file=sys.stderr)
        return 1

    texto, hay_fn_rojo = informe(resultados, casos)
    print(texto)
    print(f"\nTiempo total: {(time.perf_counter() - t0) / 60:.1f} min", file=sys.stderr)

    if args.out:
        salida = Path(args.out)
        salida.mkdir(parents=True, exist_ok=True)
        (salida / "dataset-eval.md").write_text(texto + "\n", encoding="utf-8")
        (salida / "dataset-eval.json").write_text(
            json.dumps(resultados, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"→ {salida}/dataset-eval.md", file=sys.stderr)

    # Mismo contrato que tests/runner.py: 2 = hubo una emergencia sin escalar.
    if hay_fn_rojo:
        print("\n⚠️  FALSOS NEGATIVOS EN ROJO — no se escaló una emergencia.", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())

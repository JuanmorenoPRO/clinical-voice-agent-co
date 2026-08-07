"""Convierte los 4 .xlsx del kit oficial en fixtures JSON del repo.

Se lee el Excel con la stdlib (los .xlsx son ZIPs con XML), asi que ni el harness de
evaluacion ni el arranque del jurado necesitan pandas ni openpyxl.

    .venv/bin/python scripts/load_dataset.py

Produce:
  data/seed/patients.json    40 pacientes: perfil clinico + demografia colombiana
  tests/dataset/cases.jsonl  160 casos: cuadro clinico real + label_ground_truth
  tests/dataset/dialogs.jsonl  160 casos x 2 capas de conversacion

El join entre conversaciones y trayectorias no es directo:  caso_id = "caso_" + trayectoria_id
"""
from __future__ import annotations

import json
import re
import sys
import zipfile
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path
from xml.etree import ElementTree as ET

NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
BASE = Path("ParticipantArtifacts/dataset")
OUT_SEED = Path("data/seed")
OUT_TESTS = Path("tests/dataset")

# Los .xlsx guardan las fechas como dias desde el 1899-12-30 (epoch de Excel).
EXCEL_EPOCH = date(1899, 12, 30)


def col_index(ref: str | None, fallback: int) -> int:
    if not ref:
        return fallback
    m = re.match(r"([A-Z]+)", ref)
    if not m:
        return fallback
    n = 0
    for ch in m.group(1):
        n = n * 26 + (ord(ch) - 64)
    return n - 1


def read_sheet(path: Path) -> list[dict[str, str]]:
    """Lee la unica hoja (`result`) de un .xlsx sin dependencias externas."""
    z = zipfile.ZipFile(path)
    shared: list[str] = []
    if "xl/sharedStrings.xml" in z.namelist():
        for si in ET.fromstring(z.read("xl/sharedStrings.xml")):
            shared.append("".join(t.text or "" for t in si.iter(NS + "t")))

    rows: list[list[str]] = []
    for row in ET.fromstring(z.read("xl/worksheets/sheet1.xml")).iter(NS + "row"):
        cells: dict[int, str] = {}
        for pos, c in enumerate(row.iter(NS + "c")):
            v, t = c.find(NS + "v"), c.get("t")
            if t == "inlineStr":
                el = c.find(NS + "is")
                val = "".join(x.text or "" for x in el.iter(NS + "t")) if el is not None else ""
            elif v is None:
                val = ""
            elif t == "s":
                val = shared[int(v.text or 0)]
            else:
                val = v.text or ""
            cells[col_index(c.get("r"), pos)] = val
        rows.append([cells.get(i, "") for i in range(max(cells) + 1 if cells else 0)])

    if not rows:
        return []
    header = rows[0]
    return [dict(zip(header, r)) for r in rows[1:]]


def excel_date(v: str) -> str | None:
    try:
        return (EXCEL_EPOCH + timedelta(days=float(v))).isoformat()
    except (TypeError, ValueError):
        return None


def main() -> int:
    if not BASE.exists():
        print(f"No encuentro {BASE}. Clona el kit oficial primero.", file=sys.stderr)
        return 1

    perfiles = read_sheet(BASE / "perfiles_clinicos_pacientes_silver_contest.xlsx")
    demo = read_sheet(BASE / "perfiles_pacientes_co.xlsx")
    trays = read_sheet(BASE / "trayectorias_postop_silver.xlsx")
    dlgs = read_sheet(BASE / "dataset_final.xlsx")
    print(f"leidos: {len(perfiles)} perfiles, {len(demo)} demograficos, "
          f"{len(trays)} trayectorias, {len(dlgs)} turnos")

    demo_by_id = {d["paciente_id"]: d for d in demo}

    # --- pacientes -------------------------------------------------------------
    patients = []
    for p in perfiles:
        d = demo_by_id.get(p["paciente_id"], {})
        patients.append({
            "paciente_id": p["paciente_id"],
            "nombre": d.get("nombre_completo", ""),
            "procedimiento": p["procedimiento"],
            "modulo": p["modulo_synthea"],
            "fecha_cirugia": excel_date(p["fecha_cirugia"]),
            "edad": int(p["edad"]) if p["edad"] else None,
            "genero": p["genero"],
            "comorbilidades": json.loads(p["comorbilidades"] or "[]"),
            "ciudad": d.get("ciudad", ""),
            "departamento": d.get("departamento", ""),
            "documento_cc": d.get("documento_cc", ""),
            "eps": d.get("eps", ""),
        })

    OUT_SEED.mkdir(parents=True, exist_ok=True)
    (OUT_SEED / "patients.json").write_text(
        json.dumps(patients, ensure_ascii=False, indent=2))

    # --- casos: trayectoria + etiqueta de referencia ---------------------------
    # La etiqueta vive en los turnos y es constante por caso_id.
    label_by_caso: dict[str, str] = {}
    for t in dlgs:
        label_by_caso.setdefault(t["caso_id"], t["label_ground_truth"])

    proc_by_pid = {p["paciente_id"]: p["procedimiento"] for p in perfiles}
    cases = []
    for t in trays:
        caso_id = "caso_" + t["trayectoria_id"]          # el join no es directo
        cases.append({
            "caso_id": caso_id,
            "paciente_id": t["paciente_id"],
            "procedimiento": proc_by_pid.get(t["paciente_id"], ""),
            "dia_postop": int(t["dia_postop"]),
            "arquetipo": t["arquetipo_trayectoria"],
            "dolor_nrs": int(t["dolor_nrs"]),
            "fiebre_c": float(t["fiebre_c"]),
            "movilidad": t["movilidad"],
            "herida": t["herida"],
            "apetito": t["apetito"],
            "sueno": t["sueno"],
            "label": label_by_caso.get(caso_id, ""),
        })

    OUT_TESTS.mkdir(parents=True, exist_ok=True)
    with (OUT_TESTS / "cases.jsonl").open("w", encoding="utf-8") as fh:
        for c in cases:
            fh.write(json.dumps(c, ensure_ascii=False) + "\n")

    # --- dialogos por caso y capa ----------------------------------------------
    by_key: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for t in dlgs:
        by_key[(t["caso_id"], t["capa"])].append(t)

    with (OUT_TESTS / "dialogs.jsonl").open("w", encoding="utf-8") as fh:
        for (caso_id, capa), turns in sorted(by_key.items()):
            turns.sort(key=lambda r: int(r["turno_idx"]))
            fh.write(json.dumps({
                "caso_id": caso_id, "capa": capa,
                "paciente_id": turns[0]["paciente_id"],
                "dia_postop": int(turns[0]["dia_postop"]),
                "estilo_paciente": turns[0]["estilo_paciente"],
                "label": turns[0]["label_ground_truth"],
                "turnos": [{"idx": int(r["turno_idx"]), "hablante": r["hablante"],
                            "texto": r["texto"]} for r in turns],
            }, ensure_ascii=False) + "\n")

    sin_label = [c for c in cases if not c["label"]]
    dist: dict[str, int] = {}
    for c in cases:
        dist[c["label"]] = dist.get(c["label"], 0) + 1

    print(f"\n{OUT_SEED / 'patients.json'}: {len(patients)} pacientes")
    print(f"{OUT_TESTS / 'cases.jsonl'}: {len(cases)} casos · distribucion {dist}")
    print(f"{OUT_TESTS / 'dialogs.jsonl'}: {len(by_key)} conversaciones (casos x capas)")
    if sin_label:
        print(f"AVISO: {len(sin_label)} casos sin etiqueta (join fallido)", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

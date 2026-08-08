"""Descarga el índice RAG preconstruido desde el repositorio de artefactos.

Indexar los 107 PDFs del corpus cuesta ~11 minutos de CPU. La compuerta G2 da 15
minutos para levantar TODO, así que el índice viaja preconstruido y aquí solo se
descarga y se verifica.

    .venv/bin/python scripts/fetch_index.py            # a data/chroma
    .venv/bin/python scripts/fetch_index.py --force    # rehace la descarga

Si el repositorio de artefactos no está disponible, el índice se reconstruye con
`scripts/build_index.py`, que produce exactamente el mismo resultado: el manifiesto
incluye el modelo de embeddings, los parámetros de troceado y el número de chunks
para poder comprobarlo.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tarfile
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO = "JuanmorenoPRO/repo-indices"
TAG = "chroma-bge-m3-v1"
ASSET = "chroma-bge-m3.tar.gz"
URL = f"https://github.com/{REPO}/releases/download/{TAG}/{ASSET}"

# Lo que el índice descargado debe cumplir. Si no cuadra, algo se corrompió o el
# artefacto es de otra versión, y es mejor fallar aquí que servir citas erróneas
# durante la evaluación.
ESPERADO = {"embedding_model": "bge-m3", "embedding_dim": 1024}


_ultimo_pct = -1


def barra(bloques: int, tam: int, total: int) -> None:
    """Progreso solo en terminal interactiva.

    Redibujar con `\\r` en un log o en una tubería genera miles de líneas, así que
    fuera de la terminal se calla del todo.
    """
    global _ultimo_pct
    if total <= 0 or not sys.stdout.isatty():
        return
    pct = min(100, bloques * tam * 100 // total)
    if pct == _ultimo_pct:
        return
    _ultimo_pct = pct
    hechos = pct * 40 // 100
    print(f"\r  [{'█' * hechos}{'░' * (40 - hechos)}] {pct:>3}%  "
          f"{bloques * tam / 1e6:.0f}/{total / 1e6:.0f} MB", end="", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/chroma")
    ap.add_argument("--url", default=URL)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    manifest = out / "manifest.json"
    if manifest.exists() and not args.force:
        m = json.loads(manifest.read_text(encoding="utf-8"))
        print(f"✓ El índice ya está en {out} "
              f"({m.get('total_chunks')} chunks, {m.get('total_documents_indexed')} documentos)")
        print("  Usa --force para volver a descargarlo.")
        return 0

    print(f"Descargando el índice desde {args.url}")
    t0 = time.perf_counter()
    with tempfile.TemporaryDirectory() as tmp:
        tgz = Path(tmp) / ASSET
        try:
            urllib.request.urlretrieve(args.url, tgz, reporthook=barra)
        except urllib.error.HTTPError as exc:
            print(f"\n\nNo se pudo descargar ({exc.code} {exc.reason}).", file=sys.stderr)
            print("Alternativa: .venv/bin/python scripts/build_index.py "
                  "--corpus ParticipantArtifacts/dataset/textos --out data/chroma\n"
                  "(tarda ~11 min y produce el mismo índice)", file=sys.stderr)
            return 1
        print()

        sha = hashlib.sha256(tgz.read_bytes()).hexdigest()
        print(f"  sha256 {sha[:16]}…  {tgz.stat().st_size / 1e6:.0f} MB")

        destino = Path(tmp) / "extraido"
        with tarfile.open(tgz) as tf:
            # Evita rutas absolutas o con .. dentro del tar.
            for m in tf.getmembers():
                if m.name.startswith("/") or ".." in Path(m.name).parts:
                    print(f"\nEntrada sospechosa en el tar: {m.name}", file=sys.stderr)
                    return 1
            tf.extractall(destino)

        raiz = destino / "chroma" if (destino / "chroma").exists() else destino
        m = json.loads((raiz / "manifest.json").read_text(encoding="utf-8"))
        for clave, esperado in ESPERADO.items():
            if m.get(clave) != esperado:
                print(f"\nEl índice no coincide: {clave}={m.get(clave)!r}, "
                      f"se esperaba {esperado!r}", file=sys.stderr)
                return 1

        if out.exists():
            shutil.rmtree(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(raiz), str(out))

    print(f"\n✓ Índice listo en {out} — {m['total_chunks']} chunks de "
          f"{m['total_documents_indexed']} documentos, en {time.perf_counter() - t0:.0f} s")
    if m.get("skipped"):
        print(f"  ({len(m['skipped'])} documentos omitidos; el manifiesto dice por qué)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

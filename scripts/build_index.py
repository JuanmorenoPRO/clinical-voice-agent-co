"""Construye el indice ChromaDB a partir de los 107 PDFs del kit oficial.

Se corre una vez en la maquina de desarrollo; el resultado se publica como asset de
un GitHub Release para que el jurado no tenga que reindexar (compuerta G2).

    .venv/bin/python scripts/build_index.py --corpus ParticipantArtifacts/dataset/textos \
                                            --out data/chroma

Tres trampas del corpus que este script maneja explicitamente, documentadas en
docs/spikes-7-agosto.md:
  - la carpeta breast_cancer/ contiene literatura de cancer de CUELLO UTERINO, no de
    mama, mientras el procedimiento asociado es Mastectomia;
  - un PDF de Appendicitis/ esta escaneado sin capa de texto;
  - hay documentos casi duplicados.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
import unicodedata
from collections import Counter
from pathlib import Path

import chromadb
import ollama
import pymupdf

EMBED_MODEL = "bge-m3"
EMBED_DIM = 1024
BATCH = 32
CHUNK_CHARS = 900
CHUNK_OVERLAP = 180
MIN_CHUNK_CHARS = 120
# Un documento cuyo texto total no llega aqui se considera escaneado sin capa de texto.
MIN_DOC_CHARS = 200

FOLDER_TO_PROCEDURE = {
    "Appendicitis": "Apendicectomía",
    "cholecystitis": "Colecistectomía",
    "colorectal cancer": "Colectomía",
    "total joint replacement": "Reemplazo de cadera/rodilla",
    "breast_cancer": "Mastectomía",
}

# El desajuste de breast_cancer/: se puntuan las primeras paginas y el que hable de
# cuello uterino queda como conocimiento general fuera de alcance, para que no se
# sirva como evidencia de una mastectomia.
RE_MAMA = re.compile(r"\b(mama|mastectom\w*|breast|axilar|linfedema|mammar\w*)\b", re.I)
RE_CERVIX = re.compile(r"\b(cuello uterino|cervical|cervix|c[eé]rvix|vph|hpv|colposcop\w*|uterin\w*)\b", re.I)


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s.lower())
    return re.sub(r"[^a-z0-9 ]", "", "".join(c for c in s if not unicodedata.combining(c)))


def load_pages(path: Path) -> list[tuple[int, str]]:
    """Devuelve [(numero_de_pagina_real, texto)]. La pagina es la unidad de trazabilidad."""
    pages: list[tuple[int, str]] = []
    with pymupdf.open(path) as doc:
        for i, page in enumerate(doc, start=1):
            pages.append((i, page.get_text("text")))
    return pages


def strip_running_heads(pages: list[tuple[int, str]]) -> list[tuple[int, str]]:
    """Quita encabezados y pies que se repiten en la mayoria de las paginas."""
    if len(pages) < 4:
        return pages
    firsts = Counter(p[1].strip().split("\n")[0].strip() for p in pages if p[1].strip())
    lasts = Counter(p[1].strip().split("\n")[-1].strip() for p in pages if p[1].strip())
    umbral = max(3, len(pages) // 2)
    repes = {ln for ln, n in (firsts + lasts).items() if n >= umbral and 3 < len(ln) < 120}
    if not repes:
        return pages
    out = []
    for num, txt in pages:
        lines = [ln for ln in txt.split("\n") if ln.strip() not in repes]
        out.append((num, "\n".join(lines)))
    return out


def chunk_page(text: str) -> list[str]:
    """Trocea respetando parrafos y luego frontera de frase. Nunca cruza de pagina."""
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if not text:
        return []

    chunks: list[str] = []
    for para in text.split("\n\n"):
        para = para.strip()
        if not para:
            continue
        if len(para) <= CHUNK_CHARS:
            chunks.append(para)
            continue
        start = 0
        while start < len(para):
            end = min(start + CHUNK_CHARS, len(para))
            if end < len(para):
                corte = max(para.rfind(". ", start, end), para.rfind("\n", start, end))
                if corte > start + MIN_CHUNK_CHARS:
                    end = corte + 1
            chunks.append(para[start:end].strip())
            if end >= len(para):
                break
            start = max(start + 1, end - CHUNK_OVERLAP)

    # Se fusionan los fragmentos demasiado cortos con el anterior en vez de tirarlos.
    out: list[str] = []
    for c in chunks:
        if len(c) < MIN_CHUNK_CHARS and out:
            out[-1] = f"{out[-1]} {c}"
        elif len(c) >= MIN_CHUNK_CHARS:
            out.append(c)
    return out


def classify_topic(sample: str, folder: str) -> tuple[str | None, bool, str]:
    """(procedimiento, fuera_de_alcance, tema). Solo breast_cancer necesita arbitraje."""
    proc = FOLDER_TO_PROCEDURE.get(folder)
    if folder != "breast_cancer":
        return proc, False, folder
    mama, cervix = len(RE_MAMA.findall(sample)), len(RE_CERVIX.findall(sample))
    if cervix > mama:
        return None, True, "oncologia_cervical"
    return proc, False, "oncologia_mama"


def embed(texts: list[str]) -> list[list[float]]:
    return ollama.embed(model=EMBED_MODEL, input=texts)["embeddings"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="ParticipantArtifacts/dataset/textos")
    ap.add_argument("--out", default="data/chroma")
    ap.add_argument("--limit", type=int, default=0, help="solo los primeros N PDFs (pruebas)")
    args = ap.parse_args()

    corpus, out = Path(args.corpus), Path(args.out)
    pdfs = sorted(corpus.glob("*/*.pdf"))
    if args.limit:
        pdfs = pdfs[: args.limit]
    if not pdfs:
        print(f"No hay PDFs en {corpus}", file=sys.stderr)
        return 1

    out.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(out))
    try:
        client.delete_collection("clinical_knowledge")
    except Exception:  # noqa: BLE001  — no existia
        pass
    col = client.create_collection("clinical_knowledge", metadata={"hnsw:space": "cosine"})

    seen_docs: dict[str, str] = {}   # hash del documento -> filename
    seen_chunks: set[str] = set()    # hash de chunk, mata el pasaje repetido
    manifest: dict = {
        "embedding_model": EMBED_MODEL, "embedding_dim": EMBED_DIM,
        "chunk_chars": CHUNK_CHARS, "chunk_overlap": CHUNK_OVERLAP,
        "documents": [], "skipped": [],
    }
    total_chunks = 0
    t_start = time.perf_counter()

    for n, pdf in enumerate(pdfs, start=1):
        folder = pdf.parent.name
        try:
            pages = strip_running_heads(load_pages(pdf))
        except Exception as exc:  # noqa: BLE001
            manifest["skipped"].append({"file": pdf.name, "folder": folder,
                                        "reason": f"error_lectura: {exc}"})
            print(f"[{n:>3}/{len(pdfs)}] ✗ {pdf.name[:58]:<58} error de lectura")
            continue

        full = "\n".join(t for _, t in pages)
        if len(full.strip()) < MIN_DOC_CHARS:
            manifest["skipped"].append({"file": pdf.name, "folder": folder,
                                        "reason": "sin_capa_texto", "pages": len(pages)})
            print(f"[{n:>3}/{len(pdfs)}] ⊘ {pdf.name[:58]:<58} escaneado sin capa de texto")
            continue

        doc_hash = hashlib.sha256(norm(full).encode()).hexdigest()[:16]
        if doc_hash in seen_docs:
            manifest["skipped"].append({"file": pdf.name, "folder": folder,
                                        "reason": "duplicado", "of": seen_docs[doc_hash]})
            print(f"[{n:>3}/{len(pdfs)}] ⊘ {pdf.name[:58]:<58} duplicado de {seen_docs[doc_hash][:30]}")
            continue
        seen_docs[doc_hash] = pdf.name

        proc, off_scope, topic = classify_topic(full[:6000], folder)

        ids, docs, metas = [], [], []
        for page_no, text in pages:
            for ci, chunk in enumerate(chunk_page(text)):
                ch = hashlib.sha256(norm(chunk).encode()).hexdigest()[:16]
                if ch in seen_chunks:
                    continue
                seen_chunks.add(ch)
                ids.append(f"{doc_hash}:{page_no}:{ci}")
                # Cabecera de contexto: sube mucho la recuperacion cross-lingue, porque
                # buena parte del corpus esta en ingles y las consultas van en espanol.
                docs.append(f"[{proc or 'general'}] {pdf.stem} — p.{page_no}\n{chunk}")
                metas.append({
                    "document_id": doc_hash, "filename": pdf.name, "page": page_no,
                    "procedure": proc or "", "source_folder": folder, "topic": topic,
                    "off_scope": off_scope, "chunk_index": ci, "doc_hash": doc_hash,
                })

        for i in range(0, len(ids), BATCH):
            sl = slice(i, i + BATCH)
            col.add(ids=ids[sl], documents=docs[sl], metadatas=metas[sl],
                    embeddings=embed(docs[sl]))

        total_chunks += len(ids)
        manifest["documents"].append({
            "file": pdf.name, "folder": folder, "procedure": proc,
            "topic": topic, "off_scope": off_scope,
            "pages": len(pages), "chunks": len(ids), "doc_hash": doc_hash,
        })
        flag = " [FUERA DE ALCANCE]" if off_scope else ""
        print(f"[{n:>3}/{len(pdfs)}] ✓ {pdf.name[:58]:<58} {len(pages):>3}p {len(ids):>4}ch"
              f" → {proc or 'general'}{flag}")

    dur = time.perf_counter() - t_start
    manifest.update({
        "total_documents_indexed": len(manifest["documents"]),
        "total_documents_skipped": len(manifest["skipped"]),
        "total_chunks": total_chunks,
        "build_seconds": round(dur, 1),
        "corpus_pdfs": len(pdfs),
    })
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))

    print("\n" + "=" * 78)
    print(f"Indexados {len(manifest['documents'])}/{len(pdfs)} documentos · "
          f"{total_chunks} chunks · {dur / 60:.1f} min")
    razones = Counter(s["reason"] for s in manifest["skipped"])
    for r, c in razones.items():
        print(f"  omitidos por {r}: {c}")
    por_proc = Counter(d["procedure"] or "general" for d in manifest["documents"])
    for p, c in sorted(por_proc.items()):
        print(f"  {p}: {c} documentos")
    print(f"\nmanifest → {out / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

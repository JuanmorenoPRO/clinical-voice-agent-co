#!/usr/bin/env bash
# Empaqueta el índice RAG y lo publica como asset de un GitHub Release.
#
# El índice pesa ~114 MB, por encima del límite de 100 MB por archivo de GitHub,
# así que no puede vivir en el repositorio: va como asset de Release, que admite
# hasta 2 GB. El jurado lo descarga con scripts/fetch_index.py en ~1 minuto en
# vez de reindexar 107 PDFs en 11.
#
#   gh auth login          # una vez
#   ./scripts/publish_index.sh
set -euo pipefail

REPO="${INDEX_REPO:-JuanmorenoPRO/repo-indices}"
TAG="${INDEX_TAG:-chroma-bge-m3-v1}"
SRC="${1:-dist/chroma}"
ASSET="chroma-bge-m3.tar.gz"
OUT="dist/${ASSET}"

if [[ ! -f "${SRC}/manifest.json" ]]; then
  echo "No encuentro ${SRC}/manifest.json." >&2
  echo "Construye el índice primero:" >&2
  echo "  .venv/bin/python scripts/build_index.py --out ${SRC}" >&2
  exit 1
fi

if ! gh auth status >/dev/null 2>&1; then
  echo "gh no está autenticado. Ejecuta:  gh auth login" >&2
  exit 1
fi

CHUNKS=$(python3 -c "import json;print(json.load(open('${SRC}/manifest.json'))['total_chunks'])")
DOCS=$(python3 -c "import json;print(json.load(open('${SRC}/manifest.json'))['total_documents_indexed'])")

echo "Empaquetando ${SRC} (${CHUNKS} chunks, ${DOCS} documentos)…"
mkdir -p dist
# -C dist chroma: el tar contiene 'chroma/…', que es lo que espera fetch_index.py.
tar -czf "${OUT}" -C "$(dirname "${SRC}")" "$(basename "${SRC}")"
SIZE=$(du -h "${OUT}" | cut -f1)
SHA=$(shasum -a 256 "${OUT}" | cut -d' ' -f1)
echo "  ${OUT}  ${SIZE}"
echo "  sha256 ${SHA}"

NOTAS=$(cat <<EOF
Índice ChromaDB preconstruido del corpus clínico del Tech Sphere Challenge 2026.

- **${DOCS} documentos**, **${CHUNKS} chunks**, embeddings \`bge-m3\` (1024 dims) vía Ollama
- Troceado 900/180 caracteres respetando párrafos, sin cruzar de página
- Cada chunk conserva documento y número de página real para la trazabilidad de las citas
- \`sha256\` \`${SHA}\`

Descarga: \`.venv/bin/python scripts/fetch_index.py\`
Reconstrucción equivalente: \`.venv/bin/python scripts/build_index.py\` (~11 min)

Los PDFs de origen son obra de sus autores y no se redistribuyen: este asset
contiene únicamente los vectores y los fragmentos de texto necesarios para citar.
EOF
)

if gh release view "${TAG}" --repo "${REPO}" >/dev/null 2>&1; then
  echo "El release ${TAG} ya existe; se reemplaza el asset."
  gh release upload "${TAG}" "${OUT}" --repo "${REPO}" --clobber
else
  gh release create "${TAG}" "${OUT}" \
    --repo "${REPO}" \
    --title "Índice RAG — corpus clínico (bge-m3)" \
    --notes "${NOTAS}"
fi

echo
echo "✓ Publicado en https://github.com/${REPO}/releases/tag/${TAG}"

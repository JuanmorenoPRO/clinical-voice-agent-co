# Datos de ejemplo — stand-in de Databricks / Delta Share

Estos archivos **simulan** el dataset real que llega por Databricks Delta Share
el **7 de agosto**. Sirven para que el sistema corra de punta a punta HOY:

- `patients.json` — pacientes post-operatorios colombianos ficticios. `seed.py`
  los carga en la tabla `patients`.
- `*.md` — protocolos clínicos de ejemplo. `seed.py` los ingesta por el pipeline
  real (parseo → chunking → embeddings Voyage → pgvector), construyendo así la
  **base de embeddings de prueba** que reemplaza a Databricks.

> ⏳ **7 de agosto:** en `seed.py`, reemplazar la carga de estos ejemplos por la
> ingesta desde Delta Share (cliente `delta-sharing`). Ver el bloque marcado
> `TODO Aug 7` en `seed.py`.

Los PDFs clínicos reales se pueden subir en caliente desde la consola de
conocimiento (RF-06). El pipeline de ingesta acepta PDF (PyMuPDF) y texto/markdown.

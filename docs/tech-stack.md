# Tech Stack — Documento vivo de herramientas

> **Propósito:** aquí se fijan las herramientas definitivas del proyecto. Este archivo es la **fuente de verdad** que Claude Code usará para construir la app durante la ventana del 7–10 de agosto. Actualizar la columna **Estado** a `✅ Confirmado` a medida que se decida cada componente (especialmente el 7 de agosto, cuando se anuncie el LLM obligatorio).
>
> Estados: `🔵 Propuesto` (recomendación actual) · `✅ Confirmado` (definitivo, usar en el código) · `⏳ Bloqueado` (espera información del 7 de agosto)

| Última actualización | 2026-07-24 |
|---|---|

---

## Componentes

| Componente | Propuesta | Alternativas consideradas | Estado | Notas |
|---|---|---|---|---|
| **LLM** | ⏳ Modelo obligatorio del reto | — (no hay elección) | ⏳ Bloqueado | Se anuncia el 7 de agosto. Único e idéntico para todos los participantes (gate G3). Los premios son créditos Claude → probable modelo Anthropic, pero NO asumirlo en código: usar adaptador (ADR-002). |
| **Orquestación de voz** | [Pipecat](https://github.com/pipecat-ai/pipecat) | LiveKit Agents, TEN, FastRTC | 🔵 Propuesto | Python, open source, v1.0 (abr 2026). Pipeline-first: STT/LLM/TTS intercambiables en una línea. Mejor experiencia de desarrollo local (clave para 3 días). LiveKit gana en telefonía/escala — irrelevante aquí (sin telefonía real). |
| **Transporte navegador** | WebRTC (transporte de Pipecat + SDK JS) | WebSockets crudos | 🔵 Propuesto | Cumple "llamada vía navegador" del reto. |
| **STT (español)** | Deepgram (streaming, `es`) | AssemblyAI, Whisper (local), Azure Speech STT | 🔵 Propuesto | Streaming de baja latencia con buen soporte de español. Verificar el 7 de agosto rendimiento con coloquialismos del dataset; plan B: Azure STT `es-CO`. |
| **TTS (español colombiano)** | Azure Speech — `es-CO-SalomeNeural` (F) / `es-CO-GonzaloNeural` (M) | ElevenLabs multilingüe, Cartesia | 🔵 Propuesto | Voces colombianas NATIVAS que manejan registro paisa y regionalismos → apunta directo a los 5 pts de naturalidad. ElevenLabs suena bien pero acento menos marcado es-CO. |
| **Embeddings** | ⏳ Según LLM obligatorio | Voyage AI (`voyage-3`), OpenAI (`text-embedding-3-large`), `multilingual-e5-large` (local, gratis) | ⏳ Bloqueado | Debe ser multilingüe (corpus en español). Si el LLM es Anthropic → Voyage (partner oficial). e5 local elimina un proveedor y costo. |
| **Backend** | FastAPI + Python 3.12 | — | 🔵 Propuesto | Pipecat es Python; un solo lenguaje en backend. Tipado + Pydantic + inyección de dependencias. |
| **Base de datos** | PostgreSQL 16 | — | 🔵 Propuesto | Única BD para todo (relacional + vectores). |
| **Vector store** | pgvector | Qdrant, Chroma, Weaviate | 🔵 Propuesto | Un contenedor menos, borrado transaccional de vectores (gate G5), suficiente a esta escala (ADR-004). |
| **Frontend / Consola** | Next.js + TypeScript + TailwindCSS | Streamlit (más rápido, menos pulido) | 🔵 Propuesto | Consola admin + cliente de llamada en una sola app. |
| **Infra local** | Docker Compose | — | 🔵 Propuesto | Gate G2: arranque ≤15 min. |
| **Dataset** | Cliente `delta-sharing` (Python) | — | ⏳ Bloqueado | Credenciales Delta Share llegan el 7 de agosto. |
| **Parseo de PDF** | PyMuPDF | pypdf, unstructured | 🔵 Propuesto | Rápido y conserva nº de página (necesario para trazabilidad RF-05). |
| **Repo / licencia** | GitHub público + MIT en raíz | — | ✅ Confirmado | Obligatorio por el reto. |

## Registro de cambios

| Fecha | Cambio |
|---|---|
| 2026-07-24 | Versión inicial con propuestas. LLM, embeddings y dataset bloqueados hasta el 7 de agosto. |

## Checklist del 7 de agosto

- [ ] Anotar el LLM obligatorio y confirmar su API/streaming → actualizar fila **LLM** y ADR-002/003
- [ ] Elegir embeddings compatibles → actualizar fila **Embeddings**
- [ ] Probar credenciales Delta Share y revisar esquema del dataset → actualizar fila **Dataset**
- [ ] Validar STT propuesto contra el vocabulario real del dataset
- [ ] Pasar cada fila decidida a `✅ Confirmado`

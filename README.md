# Clinical Voice Agent — Seguimiento Post-operatorio

Agente de voz con IA para el **seguimiento post-operatorio** de pacientes
(Tech Sphere Challenge 2026). Tras una cirugía, el agente conversa con el
paciente en **español colombiano**, evalúa síntomas, fundamenta cada respuesta
clínica en documentos mediante **RAG con trazabilidad completa**, decide de
forma **determinista** cuándo alertar a personal humano, y genera un **resumen
estructurado** de cada llamada.

> **Estado: scaffold inicial.** Diseñado tal como se pretende, listo para
> ajustar cuando lleguen el **modelo LLM obligatorio** y el **dataset (Databricks
> Delta Share)** el **7 de agosto**. Los puntos de ajuste están marcados en el
> código con `⏳` / `TODO Aug 7`.

Documentación de diseño: [`docs/prd.md`](docs/prd.md) ·
[`docs/architecture.md`](docs/architecture.md) ·
[`docs/decision-log.md`](docs/decision-log.md) ·
[`docs/tech-stack.md`](docs/tech-stack.md).

---

## Principio rector

El LLM **nunca** toma decisiones clínicas ni responde desde su conocimiento
interno. Las decisiones son **reglas deterministas** (funciones Python puras); las
respuestas médicas provienen exclusivamente del **conocimiento indexado**, con
cita a documento y página.

## Arquitectura (resumen)

```
Paciente ──(texto hoy / voz con Pipecat)──▶ Servicio de Conversación
                                              │  1 sola llamada LLM/turno (ADR-006)
                     RAG (Voyage + pgvector) ─┤  {sintomas, respuesta}
                                              ▼
                              Motor de Decisión (reglas puras + YAML)
                                              │  NORMAL | ALTO | CRÍTICO
                    CRÍTICO ▶ descarta LLM, guion de seguridad + alerta
                                              ▼
                        Respuesta (TTS) + traza completa (RF-05) + resumen (RF-10)
```

## Claves de API necesarias

| Variable | Servicio | Para qué | ¿Necesaria ahora? |
|---|---|---|---|
| `VOYAGE_API_KEY` | [Voyage AI](https://www.voyageai.com/) | Embeddings del RAG (`voyage-3`) — construye la base que **simula Databricks** | **Sí** (para RAG/voz) |
| `ANTHROPIC_API_KEY` | [Anthropic](https://console.anthropic.com) | LLM provisional (hasta el 7 ago) | Opcional (el modo `mock` corre sin ella) |
| `DEEPGRAM_API_KEY` | [Deepgram](https://deepgram.com) | STT en español | Solo modo voz |
| `ELEVENLABS_API_KEY` | [ElevenLabs](https://elevenlabs.io) | TTS `eleven_flash_v2_5`, voz es-CO | Solo modo voz |
| `DATABASE_URL` | PostgreSQL + pgvector | Datos + vectores | Sí (compose la provee) |

Config extra: `LLM_PROVIDER` (`mock`|`anthropic`), `EMBEDDING_DIM`. Ver
[`.env.example`](.env.example). Copia `.env.example` → `.env` y rellena las claves.

---

## Arranque con Docker (gate G2: ≤15 min)

```bash
cp .env.example .env          # y rellena VOYAGE_API_KEY (mínimo)
docker compose up --build     # db (pgvector) + backend + frontend
docker compose exec backend python seed.py   # carga pacientes + embeddings
```

- Consola:  http://localhost:3000
- API/docs: http://localhost:8000/docs
- Salud:    http://localhost:8000/health

> El sistema corre **hoy** solo con `VOYAGE_API_KEY` (RAG) y el LLM en modo
> `mock`. El modo de voz necesita además Deepgram + ElevenLabs.

## Desarrollo local (sin Docker)

```bash
# Base de datos
docker run -d --name pgvector -p 5432:5432 \
  -e POSTGRES_USER=clinical -e POSTGRES_PASSWORD=clinical -e POSTGRES_DB=clinical \
  pgvector/pgvector:pg16

# Backend
cd apps/backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cd ../.. && python seed.py                      # semilla (desde la raíz)
cd apps/backend && uvicorn app.main:app --reload   # :8000

# Frontend
cd apps/frontend && npm install && npm run dev  # :3000
```

## Probar el bucle del turno (modo texto, sin audio)

```bash
# NORMAL
curl -s localhost:8000/conversation/turn -H 'content-type: application/json' \
  -d '{"text":"Hola, me siento bien, solo un poco cansada"}' | jq

# ALTO (dolor no controlado)
curl -s localhost:8000/conversation/turn -H 'content-type: application/json' \
  -d '{"text":"Tengo dolor 9 y la pastilla no me hace nada"}' | jq

# CRÍTICO (guion de seguridad determinista, se descarta el LLM)
curl -s localhost:8000/conversation/turn -H 'content-type: application/json' \
  -d '{"text":"Estoy sangrando mucho, no para"}' | jq
```

## Tests

```bash
cd apps/backend && pytest
```

- `test_decision.py` — reglas del Motor de Decisión (sin BD, sin LLM, sin red).
- `test_rag.py` — ida y vuelta ingesta→recuperación (se **salta** si faltan
  `VOYAGE_API_KEY` o la BD).

---

## Qué se ajusta el 7 de agosto (marcado con `⏳` / `TODO Aug 7`)

| Ítem | Dónde |
|---|---|
| Adaptador del **LLM obligatorio** (junto a Mock/Anthropic) | `app/llm/factory.py` |
| Modelo de **embeddings** definitivo / `EMBEDDING_DIM` | `app/config.py`, `app/rag/embeddings.py` |
| **Reglas y umbrales** calibrados al dataset | `app/decision/thresholds.yaml`, `app/decision/rules.py` |
| Ingesta desde **Delta Share** (reemplaza los ejemplos) | `seed.py` (bloque `TODO Aug 7`) |
| **Voz**: transporte WebRTC + voz es-CO de ElevenLabs | `app/voice/pipeline.py` |

## Estructura

```
apps/
  backend/   FastAPI: llm/ rag/ decision/ summary/ voice/ routers/ tests/
  frontend/  Next.js: consola (llamada texto, conocimiento, alertas)
prompts/     plantillas de prompts versionadas
data/samples/ pacientes + protocolos de ejemplo (stand-in de Delta Share)
docs/        PRD, arquitectura, decision log, tech stack
seed.py      carga pacientes + construye la base de embeddings (Voyage)
docker-compose.yml
```

Licencia: [MIT](LICENSE).

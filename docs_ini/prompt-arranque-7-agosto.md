# Prompt de arranque — 7 de agosto (implementación definitiva)

Cómo usarlo: el 7 de agosto, **rellena el bloque `DATOS DEL 7 DE AGOSTO`** con lo
que anuncie el reto y pega **todo** el prompt (desde `--- PROMPT ---`) en Claude
Code, con el repositorio abierto. Pídele que **planifique primero** y apruebes el
plan antes de que implemente.

> Consejo: trabaja gate por gate (G1–G5 son eliminatorios) y luego los puntos de
> rúbrica. No rompas el modo texto que ya funciona.

---

## DATOS DEL 7 DE AGOSTO (rellenar antes de pegar)

```
- LLM OBLIGATORIO:
    Nombre/modelo: <<p. ej. claude-sonnet-5 / gpt-x / gemini-x>>
    Proveedor y SDK: <<Anthropic / OpenAI / Google / otro>>
    API/base URL y auth: <<endpoint, header, variable de entorno>>
    ¿Soporta streaming?: <<sí/no>>
    ¿Soporta salida estructurada / tool-calling?: <<sí/no>>
- MODELO DE EMBEDDINGS confirmado: <<p. ej. voyage-3 / text-embedding-3-large / e5>>
    Dimensión: <<p. ej. 1024>>
- DATASET (Databricks Delta Share):
    Perfil/credenciales: <<ruta config.share>>
    Share.schema.table de pacientes: <<...>>
    Esquema de columnas (pacientes): <<...>>
    Procedimientos cubiertos: <<...>>
    Vocabulario/coloquialismos observados: <<...>>
    Documentos clínicos incluidos (si los hay): <<...>>
- RÚBRICA: detalle de los 15 pts que faltaban: <<...>>
- VOZ: voice_id es-CO elegido en ElevenLabs: <<...>>
```

---

## --- PROMPT ---

Eres un ingeniero senior trabajando en este repositorio: un **agente de voz para
seguimiento post-operatorio** (Tech Sphere Challenge 2026). El scaffold ya está
construido y funciona en modo texto; hoy (7 de agosto) llegan el **LLM
obligatorio** y el **dataset real (Databricks Delta Share)**, y hay que dejar todo
**definitivo** cumpliendo TODOS los requerimientos, gates y rúbrica.

### 0. Antes de codificar
1. Lee `docs/prd.md`, `docs/architecture.md`, `docs/decision-log.md`,
   `docs/tech-stack.md` y `docs/pruebas-rag-grounding.md`.
2. Ejecuta `grep -rn "TODO Aug 7\|⏳"` para localizar TODOS los puntos de ajuste.
3. **Entra en modo plan**, propón el plan ordenado gate-first y espera mi
   aprobación antes de implementar. Verifica end-to-end tras cada bloque y **no
   rompas el modo texto ni los tests deterministas** (`pytest app/tests`).

Usa estos DATOS DEL 7 DE AGOSTO: `<<pega aquí el bloque de arriba ya rellenado>>`

### 1. LLM obligatorio (ADR-002/010, gate G3, RF-03)
- Implementa el adaptador del modelo obligatorio **detrás de la interfaz existente
  `app/llm/adapter.py::LLMAdapter`**, junto a Mock y Anthropic, seleccionable con
  `LLM_PROVIDER`. Nada fuera de `app/llm/` debe conocer al proveedor.
- Fuerza la salida estructurada `{sintomas, respuesta}` en UNA sola llamada por
  turno (ADR-006). Si el modelo no tiene salida estructurada nativa, degrada a
  JSON + reintento (como ya hace el adaptador Anthropic).
- Si el modelo es Anthropic, **carga la skill `claude-api`** para IDs, streaming y
  salida estructurada correctos. Si es otro proveedor, usa su SDK oficial.
- El LLM **solo extrae síntomas y redacta**; NUNCA decide alertas (ADR-001).

### 2. Embeddings definitivos (ADR-011)
- Confirma el modelo y su dimensión en `app/config.py` (`EMBEDDING_MODEL`,
  `EMBEDDING_DIM`). Si cambia respecto a voyage-3/1024, ajusta el cliente en
  `app/rag/embeddings.py`, **recrea la columna vector** y re-embebe (re-siembra).

### 3. Ingesta del dataset real (ADR-012, RF-06, gate G2)
- En `seed.py`, reemplaza la carga de ejemplos por la ingesta desde **Delta Share**
  (cliente `delta-sharing`) en el bloque marcado `TODO Aug 7`. Mapea el esquema
  real de pacientes a la tabla `patients`. **No cambies** `rag/ingest.py` (el
  pipeline parseo→chunk→embeddings→pgvector se reutiliza); solo cambia la fuente.
- Descomenta `delta-sharing` en `requirements.txt`.

### 4. Motor de Decisión calibrado (ADR-001/013, RF-08, 15 pts)
- Calibra `app/decision/thresholds.yaml` con el dataset (dolor, temperatura…).
- Añade reglas por procedimiento en `app/decision/rules.py` (una función pura por
  regla, registrada en `RULES`) según los procedimientos del dataset.
- Extiende `app/schemas.py::Symptoms` con el vocabulario/coloquialismos reales.
- Añade **pruebas unitarias** para cada regla nueva (`app/tests/test_decision.py`).
- Ajusta los guiones CRÍTICOS por tipo de cirugía en `app/decision/safety_scripts.py`.

### 5. RAG con trazabilidad y grounding robusto (ADR-005, RF-04/05, 20 pts)
- Añade **few-shots** con el vocabulario del dataset a `prompts/system.md` para
  mejorar la extracción.
- Mejora el grounding del caso "mismo dominio, procedimiento distinto":
  implementa **recuperación filtrada por el procedimiento del paciente** (etiqueta
  cada `Document` con su procedimiento y filtra en `app/rag/retrieve.py`), y/o
  **reranking**. Calibra `RAG_MIN_CONFIDENCE`.
- Verifica la trazabilidad end-to-end (pregunta→chunks→confianza→reglas→respuesta)
  en `GET /console/conversations/{id}`. Opcional: evaluación RAGAS sobre preguntas
  del dataset como evidencia para el reporte.

### 6. Voz en tiempo real (ADR-003/007, RF-01/02, gate G4, 15+5 pts)
- Fija el `ELEVENLABS_VOICE_ID` nativo es-CO y valídalo con frases del dataset.
- Asegura que el transporte WebRTC funcione para el **evaluador**, no solo en
  localhost (documenta el arranque; si Docker/Mac bloquea UDP, deja instrucciones
  claras o un transporte alcanzable). Mide latencia por turno (< 1.5 s, RNF-02).

### 7. Consola y alertas (RF-09/11)
- Verifica alta/baja de documentos en caliente (gate G5), historial,
  transcripción, resumen estructurado y log de alertas (polling).

### 8. Reproducibilidad ≤15 min (ADR-009, RNF-01, gate G2)
- Publica **imágenes preconstruidas en GHCR** y un `docker-compose` de evaluación
  que las **descargue**. Incluye credenciales de evaluación en el README.
- Ensaya el arranque cronometrado en una **máquina limpia** (`docker compose up` →
  demo funcional en ≤15 min): conversación de voz, subir PDF y comprobar que se
  usa, borrarlo y comprobar que se olvida.

### 9. Entregables del reto (gate G1)
- `LICENSE` MIT en raíz (ya está). Exporta el **diagrama de arquitectura + flujo
  de decisión** (Mermaid de `architecture.md`) a imagen.
- Crea `docs/final-report.md` con evidencias: prompts, configuraciones, capturas,
  reglas, métricas de latencia y (si aplica) RAGAS.
- Actualiza `docs/tech-stack.md` y `docs/decision-log.md` (ADR-010–013) a
  `✅ Confirmado`. Prepara guion del video (valor de negocio + decisión técnica
  más crítica, apoyado en el decision-log).

### 10. Reglas transversales (no violar)
- El LLM nunca dispara alertas; toda afirmación clínica sale del RAG con cita.
- El adaptador LLM permanece intercambiable; nada fuera de `app/llm/` conoce al
  proveedor. En CRÍTICO se descarta la redacción del LLM (guion determinista).
- Mantén el modo texto y los tests en verde en cada paso; commits pequeños.

Empieza por leer los docs y presentar el plan gate-first para aprobación.

## --- FIN DEL PROMPT ---

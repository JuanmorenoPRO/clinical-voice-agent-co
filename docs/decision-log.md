# Decision Log — Registro de decisiones de arquitectura (ADR)

> Cada decisión técnica relevante se registra aquí con su contexto, alternativas y riesgos. Este registro es el insumo directo para la **Pregunta 2 del video** del reto: *"Explica tu decisión técnica más crítica — alternativas consideradas, por qué se descartaron, riesgos identificados y mejoras con 2 semanas más"*.
>
> Estados: `Aceptada` · `Propuesta` (revisar el 7 de agosto) · `Reemplazada por ADR-XXX`

| Última actualización | 2026-07-24 |
|---|---|

---

## ADR-001 — Las decisiones clínicas son deterministas: el LLM nunca dispara alertas

- **Estado:** Aceptada
- **Contexto:** El reto exige lógica de decisión que determine cuándo alertar a personal humano (15 pts de la rúbrica). Un LLM puede alucinar, variar entre ejecuciones y no es auditable; en un contexto de seguridad del paciente eso es inaceptable.
- **Decisión:** El nivel de riesgo (`NORMAL | ALTO | CRÍTICO`) lo calcula un **motor de reglas determinista y configurable** (YAML/JSON versionado), evaluado sobre síntomas estructurados. El LLM solo *extrae* los síntomas de la conversación (tarea de NLP), nunca decide.
- **Alternativas descartadas:**
  - *El LLM decide directamente* → no reproducible, no auditable, riesgo de alucinación en decisión de seguridad.
  - *Clasificador ML entrenado* → sin datos de entrenamiento suficientes ni tiempo en 3 días; tampoco es explicable regla a regla.
- **Riesgos:** la extracción de síntomas (que sí usa LLM) puede fallar con lenguaje coloquial → mitigación: salida estructurada validada con esquema, few-shots con vocabulario del dataset (⏳ 7 de agosto) y umbral conservador (ante duda, escalar).
- **Con 2 semanas más:** reglas cargables por cirugía específica, panel de edición de reglas en la consola, evaluación sistemática de la extracción contra un set etiquetado.

## ADR-002 — Capa adaptadora de LLM (proveedor-agnóstica)

- **Estado:** Aceptada
- **Contexto:** El LLM es obligatorio, idéntico para todos, y **se anuncia el 7 de agosto** — tres días antes del cierre. Todo el diseño previo debe sobrevivir a cualquier anuncio.
- **Decisión:** Una interfaz única (`LLMAdapter`: chat streaming + salida estructurada) detrás de la cual se implementa el cliente concreto. Nada fuera del adaptador conoce al proveedor. Los prompts viven en `/prompts` como plantillas, independientes del modelo.
- **Alternativas descartadas:**
  - *Esperar al 7 de agosto para diseñar* → quema días de la ventana de construcción.
  - *Asumir un proveedor probable y acoplarse* (los premios son créditos Claude, lo que sugiere Anthropic) → si la suposición falla, el retrabajo bajo presión es el peor escenario.
- **Riesgos:** el modelo mandatorio podría no soportar streaming o salida estructurada nativa → mitigación: el adaptador degrada a no-streaming (frases de relleno para latencia) y a parseo JSON con reintentos.
- **Con 2 semanas más:** benchmark de latencia/calidad de prompts específico del modelo, caché de respuestas frecuentes.

## ADR-003 — Pipeline de voz por componentes (STT → LLM → TTS), no API realtime nativa

- **Estado:** Aceptada
- **Contexto:** Las APIs de voz nativas (speech-to-speech) ofrecen menor latencia, pero solo existen para algunos proveedores. El LLM es impuesto y desconocido hasta el 7 de agosto; además el flujo exige pasos intermedios imposibles dentro de una API cerrada: RAG obligatorio, extracción estructurada de síntomas y motor de reglas **entre** la comprensión y la respuesta.
- **Decisión:** Cadena en streaming STT → (RAG + reglas + LLM) → TTS orquestada con **Pipecat**, con transporte WebRTC al navegador.
- **Alternativas descartadas:**
  - *API realtime speech-to-speech* → acopla al proveedor (contradice ADR-002) y no permite insertar el motor de decisión determinista en el ciclo (contradice ADR-001).
  - *LiveKit Agents* → excelente framework, pero su ventaja es telefonía/escala e infraestructura autogestionada; el reto prohíbe telefonía real y Pipecat da mejor iteración local para una ventana de 3 días.
  - *WebSockets artesanales* → reinventar manejo de turnos, VAD e interrupciones que Pipecat ya resuelve.
- **Riesgos:** latencia acumulada de la cadena → mitigación: streaming extremo a extremo, TTS desde la primera oración, respuestas cortas por diseño de prompt.
- **Con 2 semanas más:** ajuste fino de VAD/interrupciones, medición por tramo (STT/LLM/TTS) en la traza.

## ADR-004 — PostgreSQL + pgvector como única base de datos

- **Estado:** Aceptada
- **Contexto:** El sistema necesita datos relacionales (pacientes, conversaciones, trazas, alertas) y búsqueda vectorial (RAG). El gate G2 exige que todo arranque en ≤15 minutos con un README.
- **Decisión:** Una sola instancia PostgreSQL con extensión **pgvector** para los embeddings.
- **Alternativas descartadas:**
  - *Vector DB dedicada (Qdrant/Weaviate/Chroma)* → un contenedor y un cliente más que mantener y arrancar; a la escala del reto (decenas de documentos) no aporta rendimiento perceptible.
  - *Índice en memoria (FAISS)* → el borrado inmediato y persistente por documento (gate G5) es más frágil; se pierde la transaccionalidad.
- **Beneficio clave:** eliminar un documento y sus vectores es **una transacción SQL** — el "olvido" inmediato del gate G5 queda garantizado por ACID, no por sincronización entre dos almacenes.
- **Riesgos:** ninguno relevante a esta escala.
- **Con 2 semanas más:** índice HNSW afinado, búsqueda híbrida (BM25 + vectores) para vocabulario clínico.

## ADR-005 — RAG obligatorio para toda afirmación clínica

- **Estado:** Aceptada
- **Contexto:** La rúbrica asigna 20 pts a precisión RAG + trazabilidad, y el dominio (salud) no tolera respuestas inventadas. El conocimiento interno del modelo no es citable ni actualizable ni "olvidable".
- **Decisión:** Toda afirmación clínica del agente debe provenir de chunks recuperados del índice, con cita (documento, página, chunk_id, confianza) persistida en la traza. Bajo el umbral de confianza, el agente declara que no tiene evidencia y ofrece escalar. La conversación no clínica (saludos, empatía, preguntas de chequeo) no requiere RAG.
- **Alternativas descartadas:**
  - *Permitir respuesta del modelo con disclaimer* → incompatible con trazabilidad exigida; imposible "olvidar" conocimiento al borrar un documento (gate G5).
  - *RAG para todo, incluso small talk* → latencia y rigidez innecesarias en los turnos no clínicos.
- **Riesgos:** el clasificador clínico/no-clínico del turno puede errar → mitigación: sesgo hacia clínico (falso positivo solo agrega una búsqueda); recuperación deficiente si el chunking rompe el contexto → mitigación: chunking con solapamiento y metadatos de sección.
- **Con 2 semanas más:** reranking, evaluación RAGAS sobre preguntas del dataset, citas mostradas en la consola con resaltado del pasaje.

---

## ⏳ Decisiones pendientes (7 de agosto)

| ID reservado | Decisión | Se resuelve con |
|---|---|---|
| ADR-006 | Cliente e integración del LLM obligatorio | Anuncio del modelo |
| ADR-007 | Modelo de embeddings definitivo | Compatibilidad con el LLM / costos |
| ADR-008 | Estrategia de ingesta del dataset (Delta Share) | Credenciales y esquema |
| ADR-009 | Reglas clínicas definitivas por procedimiento | Contenido del dataset |

---

## Plantilla para nuevas decisiones

```markdown
## ADR-XXX — Título

- **Estado:** Propuesta | Aceptada | Reemplazada por ADR-YYY
- **Contexto:** ¿Qué problema o restricción motiva la decisión?
- **Decisión:** ¿Qué se decidió, en una frase clara?
- **Alternativas descartadas:** opción → por qué no.
- **Riesgos:** riesgo → mitigación.
- **Con 2 semanas más:** mejoras que se harían con más tiempo (insumo para el video).
```

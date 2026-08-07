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
- **Decisión:** El nivel de riesgo (`NORMAL | ALTO | CRÍTICO`) lo calcula un conjunto de **funciones Python puras** — una por regla, con nombre, descripción legible y prueba unitaria — con los **umbrales** (temperatura, escala de dolor) en un YAML pequeño. El LLM solo *extrae* los síntomas de la conversación (tarea de NLP), nunca decide.
- **Alternativas descartadas:**
  - *El LLM decide directamente* → no reproducible, no auditable, riesgo de alucinación en decisión de seguridad.
  - *Motor de reglas genérico / DSL propio* → construir un mini-intérprete en 3 días es alcance desperdiciado y bugs justo donde no se toleran; lo que se evalúa es precisión y explicabilidad, no un framework. Los umbrales en YAML dan toda la "configurabilidad" que importa.
  - *Clasificador ML entrenado* → sin datos de entrenamiento suficientes ni tiempo en 3 días; tampoco es explicable regla a regla.
- **Riesgos:** la extracción de síntomas (que sí usa LLM) puede fallar con lenguaje coloquial → mitigación: salida estructurada validada con esquema, few-shots con vocabulario del dataset (⏳ 7 de agosto) y umbral conservador (ante duda, escalar).
- **Con 2 semanas más:** reglas por cirugía específica, panel de edición de umbrales en la consola, evaluación sistemática de la extracción contra un set etiquetado.

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
  - *API realtime speech-to-speech* → estas APIs sí soportan tool-calling (el motor de reglas podría insertarse en el ciclo), pero acoplan la solución a un proveedor concreto cuando el LLM obligatorio se desconoce hasta el 7 de agosto — ese es el argumento decisivo (contradice ADR-002). Además, imponer RAG obligatorio y el override determinista en CRÍTICO es más directo controlando la cadena.
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
- **Decisión:** Toda afirmación clínica del agente debe provenir de chunks recuperados del índice, con cita (documento, página, chunk_id, confianza) persistida en la traza. Bajo el umbral de confianza, el agente declara que no tiene evidencia y ofrece escalar. La recuperación corre **siempre** que el paciente formula una pregunta — pgvector es local y cuesta milisegundos — **sin clasificador previo de turno**; si el LLM obligatorio soporta tool-calling, el RAG se expone como tool y el propio modelo decide invocarlo (el tool-calling *es* el clasificador).
- **Alternativas descartadas:**
  - *Permitir respuesta del modelo con disclaimer* → incompatible con trazabilidad exigida; imposible "olvidar" conocimiento al borrar un documento (gate G5).
  - *Clasificador clínico/no-clínico previo por turno* → otra llamada u otro componente que puede errar, para ahorrarse una búsqueda local que cuesta milisegundos. Complejidad sin retorno.
- **Riesgos:** si el LLM obligatorio no soporta tool-calling → fallback en el adaptador: recuperación siempre activa; recuperación deficiente si el chunking rompe el contexto → mitigación: chunking con solapamiento y metadatos de sección.
- **Con 2 semanas más:** reranking, evaluación RAGAS sobre preguntas del dataset, citas mostradas en la consola con resaltado del pasaje.

## ADR-006 — Una sola llamada LLM por turno + guion determinista de seguridad en CRÍTICO

- **Estado:** Aceptada
- **Contexto:** El presupuesto de latencia es < 1.5 s por turno (RNF-02). El diseño original implicaba dos llamadas LLM secuenciales por turno (extracción de síntomas → respuesta) a un modelo cuya latencia se desconoce hasta el 7 de agosto. Además, cuando el paciente reporta síntomas críticos, ni siquiera la *redacción* de la respuesta debería depender del LLM.
- **Decisión:** **Una sola llamada** por turno con salida estructurada `{sintomas, respuesta}`. El Motor de Decisión evalúa `sintomas` **antes** de enviar audio al paciente; si el nivel es CRÍTICO, la respuesta del LLM se descarta y se emite un **guion determinista de seguridad** (texto fijo, revisado clínicamente, testeable), junto con la alerta y el cierre seguro de la llamada.
- **Alternativas descartadas:**
  - *Dos llamadas secuenciales (extraer → responder)* → duplica la latencia del componente más lento e impredecible de la cadena.
  - *Extracción asíncrona en paralelo a la respuesta* → la respuesta llegaría al paciente sin pasar por las reglas; inaceptable en seguridad del paciente.
- **Riesgos:** extracción y redacción acopladas en un solo prompt pueden degradarse mutuamente → mitigación: few-shots con vocabulario del dataset (⏳ 7 de agosto) y validación de esquema con reintento; el guion crítico puede sonar robótico → redactarlo con tono empático y validarlo escuchándolo en la voz TTS elegida.
- **Con 2 semanas más:** evaluación sistemática de la extracción contra set etiquetado; guiones de seguridad por tipo de cirugía.

## ADR-007 — TTS con ElevenLabs Flash v2.5

- **Estado:** Aceptada (decisión de Juan)
- **Contexto:** La rúbrica otorga 5 pts a la naturalidad del español con regionalismos colombianos, y el TTS es un tramo crítico del presupuesto de latencia.
- **Decisión:** **ElevenLabs, modelo `eleven_flash_v2_5`** (~75 ms de latencia, streaming, integración nativa con Pipecat), con una **voz nativa latina/colombiana** elegida de la Voice Library y validada contra frases del dataset.
- **Alternativas descartadas:**
  - *Azure Speech `es-CO-SalomeNeural`/`GonzaloNeural`* → voces colombianas nativas muy competentes en regionalismo; queda como **plan B activo** si la voz de ElevenLabs no convence en acento.
  - *ElevenLabs `eleven_v3`* → mejor calidad expresiva pero latencia alta; no apto para conversación en tiempo real.
  - *Cartesia* → rápido, pero menor madurez en español latino.
- **Riesgos:** las voces por defecto de ElevenLabs arrastran acento inglés al hablar español → mitigación: selección cuidadosa de voz nativa y validación temprana el 7 de agosto (checklist en [tech-stack.md](tech-stack.md)); costo por carácter → respuestas cortas por diseño de prompt.
- **Con 2 semanas más:** clonación de una voz con acento colombiano específico; estilos/SSML por contexto emocional.

## ADR-008 — Estructura simple: una app backend con módulos, sin packages compartidos

- **Estado:** Aceptada
- **Contexto:** El borrador inicial (heredado del prompt de ChatGPT) proponía un monorepo con `/packages` (rag, decision_engine, shared_types) instalables. Para una persona y 3 días, eso significa paquetes editables, contextos de Docker más complejos y sincronización de tipos Python↔TypeScript.
- **Decisión:** Una sola app FastAPI con módulos internos (`app/voice`, `app/rag`, `app/decision`, `app/summary`) y límites claros entre módulos + inyección de dependencias. Los ~5 tipos que el frontend espeja del backend se escriben **a mano** en TypeScript.
- **Alternativas descartadas:**
  - *Monorepo con packages instalables* → fricción de imports y builds sin valor evaluable; la rúbrica premia modularidad, que se demuestra con límites de módulo, no con packaging.
  - *Codegen OpenAPI → TypeScript* → tooling y paso de build extra para mantener 5 tipos.
- **Riesgos:** acoplamiento gradual entre módulos → mitigación: interfaces por módulo y revisión al final de cada día.
- **Con 2 semanas más:** extraer packages solo si un segundo consumidor real aparece.

## ADR-009 — Reproducibilidad: imágenes preconstruidas en GHCR

- **Estado:** Aceptada
- **Contexto:** El gate G2 exige que la solución corra en ≤15 minutos siguiendo el README, y ese tiempo incluye descargas y builds. Un `docker compose up --build` en frío (build de Next.js + pip install) puede consumir 10+ minutos él solo en la máquina del evaluador.
- **Decisión:** Publicar las imágenes ya construidas en **GitHub Container Registry**; el compose de evaluación las **descarga**. El build desde fuente queda documentado como plan B. El día 3 se ensaya el arranque cronometrado en una máquina limpia.
- **Alternativas descartadas:**
  - *Build desde fuente como única vía* → lento y frágil (red, caché, versiones de Node/Python del host).
- **Riesgos:** imágenes desactualizadas respecto al código → mitigación: script único de publicación (`build + push`) ejecutado como último paso antes de entregar; visibilidad del paquete en GHCR debe ser pública → verificarla.
- **Con 2 semanas más:** CI que publica en cada push a `main`.

---

## ⏳ Decisiones pendientes (7 de agosto)

| ID reservado | Decisión | Se resuelve con |
|---|---|---|
| ADR-010 | Cliente e integración del LLM obligatorio | Anuncio del modelo |
| ADR-011 | Modelo de embeddings definitivo | Compatibilidad con el LLM / costos |
| ADR-012 | Ingesta del dataset — alcance de `seed.py` (Delta Share) | Credenciales y esquema |
| ADR-013 | Reglas clínicas definitivas por procedimiento | Contenido del dataset |

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

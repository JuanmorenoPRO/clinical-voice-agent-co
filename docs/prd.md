# PRD — Agente de Voz para Seguimiento Post-Operatorio

| Campo | Valor |
|---|---|
| Proyecto | Clinical Assistant — Tech Sphere Challenge 2026 (Voice Agent Edition) |
| Autor | Juan Moreno |
| Estado | Borrador — a la espera de materiales oficiales (⏳ 7 de agosto) |
| Última actualización | 2026-07-24 |
| Documentos relacionados | [architecture.md](architecture.md) · [decision-log.md](decision-log.md) · [tech-stack.md](tech-stack.md) |

---

## 1. Resumen ejecutivo

Sistema de **agente de voz con IA para seguimiento post-operatorio de pacientes**, construido para el Tech Sphere Challenge 2026 de Source Meridian. Tras una cirugía, las primeras horas son críticas: el agente llama (vía navegador) al paciente, conversa en **español colombiano natural**, evalúa síntomas, fundamenta cada respuesta clínica en documentos mediante **RAG con trazabilidad completa**, decide de forma **determinista** cuándo alertar a personal humano, y genera un **resumen estructurado** de cada llamada. El personal clínico administra el conocimiento del agente en caliente (subir/eliminar documentos) desde una consola web.

**Principio rector:** el LLM nunca toma decisiones clínicas ni responde desde su conocimiento interno. Las decisiones son reglas deterministas; las respuestas médicas provienen exclusivamente del conocimiento indexado, con cita a documento y página.

## 2. Contexto del reto

- **Organiza:** Source Meridian (con Pascual Bravo, AI Tinkerers, DB Crew LATAM, GDG Medellín, UNAL). Competencia individual, residentes de Colombia.
- **Fechas:** materiales técnicos y dataset el **7 de agosto**; construcción **7–10 de agosto**; evaluación 10–18 de agosto; ganadores 5 de septiembre.
- **Dataset:** datos reales de pacientes colombianos vía Databricks Delta Share (lenguaje coloquial, síntomas descritos de forma ambigua).
- **Entregables del reto:** (1) repo público GitHub con licencia MIT, (2) diagrama de arquitectura + flujo de decisión, (3) reporte final con evidencias (prompts, configuraciones, capturas), (4) video con demo y 2 preguntas en cámara (valor de negocio; decisión técnica más crítica).

### Gates eliminatorios (todos obligatorios)

| # | Gate | Cubierto por |
|---|---|---|
| G1 | Los 4 entregables presentados | Plan de entregables (§9) |
| G2 | La solución corre en ≤15 min siguiendo el README (credenciales incluidas) | RNF-01 |
| G3 | Usa el modelo obligatorio único | RF-03 + ⏳ 7 de agosto |
| G4 | Conversación de voz en tiempo real funciona | RF-01, RF-02 |
| G5 | Consola de conocimiento sube/elimina documentos (el agente aprende y olvida) | RF-06, RF-07 |

### Rúbrica de evaluación (100 pts)

| Pts | Criterio | Cubierto por |
|---|---|---|
| 20 | Precisión RAG, exactitud clínica, actualización de conocimiento en vivo, trazabilidad a fuentes | RF-04, RF-05, RF-06, RF-07, RF-10 |
| 15 | Calidad y adaptabilidad de la conversación de voz | RF-01, RF-02, RNF-02 |
| 15 | Lógica de decisión (precisión al disparar alertas) | RF-08 |
| 15 | Arquitectura del sistema e implementación técnica | [architecture.md](architecture.md) |
| 15 | Calidad de código, documentación y reproducibilidad | RNF-01, RNF-05 |
| 5 | Naturalidad del español (regionalismos colombianos) | RF-02, RNF-03 |

> ⚠️ Los criterios publicados suman 85 pts; los 15 restantes no se detallan en la página del reto. **⏳ Confirmar el 7 de agosto.**

## 3. Problema y oportunidad

Tras el alta quirúrgica, complicaciones como sangrado, fiebre o dolor no controlado aparecen en las primeras 24–72 horas, cuando el paciente ya no está bajo observación directa. El seguimiento telefónico manual es costoso, inconsistente y no escala. Un agente de voz que llame proactivamente, converse con naturalidad, detecte señales de alarma con reglas clínicas explícitas y escale a un humano cuando corresponde, permite cobertura del 100 % de los pacientes a una fracción del costo, sin sustituir el juicio clínico (el agente **detecta y escala**, no diagnostica).

## 4. Usuarios

| Usuario | Interacción | Necesidad principal |
|---|---|---|
| Paciente post-operatorio | Conversación de voz (navegador) | Ser escuchado con naturalidad, en su español, sin fricción tecnológica |
| Personal clínico (enfermería) | Consola web + alertas | Saber a quién llamar YA, con contexto (síntomas, reglas disparadas) |
| Administrador de conocimiento | Consola web | Subir/eliminar protocolos clínicos y ver el estado del índice |
| Evaluadores del reto | README + repo + video | Reproducir todo en ≤15 min y auditar la trazabilidad |

## 5. Requisitos funcionales

### Conversación de voz

- **RF-01 — Conversación de voz en tiempo real.** El paciente conversa con el agente vía navegador (WebRTC), con captura de micrófono y reproducción de audio. Sin telefonía real. Latencia percibida por turno objetivo: ver RNF-02.
- **RF-02 — Diálogo adaptativo en español colombiano.** El agente adapta sus preguntas a las respuestas del paciente (p. ej., si menciona dolor, pregunta escala 0–10; si menciona fiebre, pregunta temperatura). Comprende coloquialismos y descripciones ambiguas de síntomas ("me siento maluco", "tengo el estómago revuelto") y habla con registro colombiano natural.
- **RF-03 — LLM obligatorio único.** Toda generación de lenguaje usa exclusivamente el modelo mandatorio del reto (⏳ se anuncia el 7 de agosto), detrás de una capa adaptadora intercambiable (ver ADR-002).

### RAG y conocimiento

- **RF-04 — Respuestas clínicas fundamentadas (RAG).** Toda afirmación clínica del agente proviene de los documentos indexados, nunca del conocimiento interno del modelo. Si no hay evidencia suficiente en el índice, el agente lo dice explícitamente y ofrece escalar.
- **RF-05 — Trazabilidad a la fuente.** Cada respuesta clínica registra: pregunta, chunks recuperados, documento fuente, página, `chunk_id`, puntaje de confianza, reglas disparadas y respuesta final. Nada es caja negra.
- **RF-06 — Alta de conocimiento en caliente.** Desde la consola se suben documentos (PDF); el sistema los parsea, trocea, vectoriza e indexa sin reiniciar servicios. El agente usa el nuevo conocimiento en la siguiente consulta.
- **RF-07 — Baja de conocimiento en caliente.** Eliminar un documento remueve inmediatamente sus vectores del índice; el agente "olvida" ese contenido en la siguiente consulta.

### Decisión y reporte

- **RF-08 — Motor de decisión determinista.** Reglas clínicas explícitas y configurables (sin LLM) determinan el nivel de riesgo y cuándo alertar. Ejemplos de línea base (⏳ ajustar con el dataset y materiales del 7 de agosto):
  - Dolor > 8 **y** medicación inefectiva → **ALTO**
  - Temperatura > 38.5 °C → **ALTO**
  - Sangrado abundante → **CRÍTICO**
  - Dificultad para respirar → **CRÍTICO**
  - Pérdida de consciencia → **CRÍTICO**
- **RF-09 — Alerta a personal humano.** Al dispararse una regla ALTO/CRÍTICO, se genera una alerta visible en la consola con paciente, síntomas extraídos, reglas disparadas y transcripción relevante. En nivel CRÍTICO el agente lo comunica al paciente y cierra con instrucción segura.
- **RF-10 — Resumen estructurado por llamada.** Al finalizar cada conversación se genera un JSON con: paciente, cirugía, duración, síntomas, entidades extraídas, documentos citados, nivel de riesgo, reglas disparadas y recomendación.

### Consola de administración

- **RF-11 — Consola web.** Gestión de documentos (subir, eliminar, nº de chunks, estado de embeddings, última actualización), historial de conversaciones con transcripción, resúmenes estructurados, log de decisiones/alertas y trazabilidad por respuesta.

## 6. Requisitos no funcionales

- **RNF-01 — Reproducibilidad ≤15 min.** `docker compose up` + README con pasos numerados y credenciales de evaluación incluidas. Este es un gate eliminatorio: se ensaya el arranque desde cero en máquina limpia antes de entregar.
- **RNF-02 — Latencia de voz.** Objetivo < 1.5 s entre fin de habla del paciente e inicio de respuesta del agente (streaming en STT, LLM y TTS).
- **RNF-03 — Español colombiano.** Voz TTS con acento colombiano nativo (`es-CO`); prompts y few-shots con regionalismos del dataset.
- **RNF-04 — Seguridad del paciente y explicabilidad.** El LLM no decide alertas (RF-08); toda salida clínica es auditable (RF-05); ante ambigüedad, el sistema escala en lugar de asumir.
- **RNF-05 — Calidad de código.** Arquitectura modular (ver [architecture.md](architecture.md)), tipado estático, inyección de dependencias, pruebas unitarias del motor de decisión y del pipeline RAG, prompts versionados fuera del código.

## 7. Fuera de alcance

Explícitamente excluido por el reto:

- Sistema de telesalud productivo ni integraciones hospitalarias (HL7/FHIR).
- Autenticación enterprise.
- Cobertura de todos los procedimientos médicos (se acota a los del dataset/materiales).
- Telefonía real (PSTN/SIP): las llamadas son vía navegador/API.

## 8. ⏳ Pendiente — 7 de agosto

| Ítem | Impacto | Dónde se actualiza |
|---|---|---|
| LLM obligatorio (modelo y API) | Adaptador LLM, embeddings compatibles, costos | [tech-stack.md](tech-stack.md), ADR-002/003 |
| Dataset (esquema, procedimientos cubiertos, vocabulario) | Reglas del motor de decisión, prompts, few-shots | RF-02, RF-08 |
| Materiales técnicos y credenciales Delta Share | Módulo de ingesta de datos | [architecture.md](architecture.md) §Datos |
| Detalle de los 15 pts restantes de la rúbrica | Priorización de esfuerzo | §2 |

## 9. Entregables y criterios de aceptación

| Entregable del reto | Fuente en este proyecto |
|---|---|
| Repo público GitHub + licencia MIT en raíz | Este repositorio (crear `LICENSE` antes de publicar) |
| Diagrama de arquitectura + flujo de decisión | [architecture.md](architecture.md) (Mermaid, exportar a imagen) |
| Reporte final con evidencias | `docs/final-report.md` (se crea durante la construcción: prompts, configs, capturas) |
| Video demo + 2 preguntas | Guion apoyado en este PRD (pregunta 1: §1, §3) y en [decision-log.md](decision-log.md) (pregunta 2) |

**Definición de éxito mínimo (gates):** un evaluador clona el repo, sigue el README, y en ≤15 minutos sostiene una conversación de voz con el agente, sube un PDF y comprueba que el agente lo usa, lo elimina y comprueba que lo olvida.

## 10. Riesgos

| Riesgo | Prob. | Mitigación |
|---|---|---|
| El LLM obligatorio no encaja con el diseño asumido (p. ej., sin streaming o sin API de voz) | Media | ADR-002/003: pipeline STT→LLM→TTS por componentes y adaptador LLM intercambiable; nada asume API realtime nativa |
| Ventana de 3 días insuficiente | Alta | Esta documentación previa + esqueleto de repo listo; priorizar gates antes que puntos de rúbrica |
| Latencia de voz alta con el modelo mandatorio | Media | Streaming extremo a extremo; respuestas cortas; frases de relleno ("ajá, cuéntame más") mientras se genera |
| Dataset con vocabulario impredecible | Media | Día 1 (7 ago): explorar dataset y ajustar prompts/reglas antes de codificar de más |
| Fallo del arranque en máquina del evaluador | Baja | RNF-01: ensayo en máquina limpia; Docker Compose sin dependencias del host |

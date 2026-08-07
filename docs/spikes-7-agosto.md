# Verificaciones del 7 de agosto — evidencia de las decisiones técnicas

Antes de escribir el refactor se midieron los tres supuestos que condicionan toda la
arquitectura. Los scripts son `scripts/spike_verify.py`, `scripts/spike_latency.py` y
`scripts/spike_latency2.py`, y se pueden volver a correr.

**Máquina de referencia:** MacBook Pro M1 Pro, 16 GB RAM, macOS 25.5, Python 3.12.0,
Ollama 0.9.3.

---

## 1. Los modelos permitidos que realmente se pueden invocar

La lista de la compuerta G3 es cerrada, pero dos de sus cuatro entradas están muertas:

| Modelo permitido | Estado verificado el 7-ago-2026 |
|---|---|
| Google Gemini 1.5 Flash | **Retirado.** La familia 1.5 devuelve 404 en la API |
| Llama 3.1 70B vía Groq | **Decomisionado el 24-ene-2025.** Groq redirigía a la 3.3, que a su vez se apaga el 16-ago-2026 |
| Llama 3.2 (1B / 3B) local | ✅ Vivo en Ollama |
| Phi-3.5 Mini (3.8B) local | ✅ Vivo en Ollama |

Se eligió **`llama3.2:3b`**: está literalmente en la lista, corre offline y no tiene fecha
de apagado. Whisper de Groq sí sigue en producción, y usarlo no compromete G3 porque la
compuerta restringe el modelo *que razona*, no el reconocimiento de voz.

---

## 2. Salida estructurada: el `format` de Ollama con JSON Schema

**Resultado: JSON válido y conforme al esquema en 6/6 casos.** La generación va restringida
por la gramática del esquema, así que un valor fuera del enum es imposible por construcción.
Esto elimina el riesgo principal de usar un modelo de 3B para extracción.

Dos condiciones que descubrimos midiendo:

- El esquema debe ser **plano y todo enum de strings**, con centinela `"no_dice"` en vez de
  `null`. La gramática se atraganta con `anyOf`/`null`, así que **no** sirve
  `model_json_schema()` de Pydantic: el esquema se escribe a mano.
- `num_predict` debe ir holgado. Con `llama3.2:1b` y un presupuesto corto el JSON se truncó
  a mitad de string y dejó de parsear.

### Latencia: el techo de la máquina son 40 tok/s

Medido en bruto: 198 tokens en 4.914 ms → **40 tok/s**, con evaluación de prompt de 34
tokens en 404 ms. Es decir, **la latencia es esencialmente el número de tokens de salida**,
y el prompt casi no cuenta porque el prefijo fijo se cachea.

Variantes de esquema para extraer el slot de dolor, 6 frases coloquiales, `llama3.2:3b`:

| Variante | P50 | Tokens de salida | Aciertos |
|---|---:|---:|---:|
| Completo, 8 campos, con `acuse` | 1.652 ms | 65 | 3/4 |
| Por slot, con `acuse` | 1.483 ms | 59 | 3/4 |
| Por slot, sin `acuse` | 1.210 ms | 48 | 3/4 |
| Por slot + bandera roja + intención, nombres cortos | 927 ms | 36 | 6/6 |
| Por slot + bandera roja, nombres cortos | 628 ms | 24 | 6/6 |
| **Mínimo (solo el slot), nombres cortos, con few-shot** | **323 ms** | **11** | **6/6** |
| Mínimo, con system prompt corto sin ejemplos | 438 ms | 16 | 1/4 |

Tres conclusiones que definen el diseño:

1. **La precisión la dan los ejemplos few-shot, no el tamaño del esquema.** Recortar el
   system prompt hundió los aciertos a 1/4; recortar el esquema no los tocó. Los ejemplos
   se quedan y se minan del `capa2_ruidosa` del dataset.
2. **El `acuse` empático sale del LLM.** Cuesta ~11 tokens y lo que genera un 3B es
   inservible: ante "me duele un berraco" devolvió `"agudo"`, y ante "no me provoca nada"
   devolvió `"se siente mal"`. Eso no es empatía, son etiquetas. Pasa a un banco de
   plantillas deterministas con rotación anti-repetición, que además permite pre-sintetizar
   el audio.
3. **Un esquema por slot, con nombres de campo de una letra.** El spine determinista ya sabe
   qué está preguntando, así que no tiene sentido pedirle al modelo los seis slots en cada
   turno.

`llama3.2:1b` se midió y se descartó: 220-282 ms pero solo 4-5 aciertos de 6, y truncó el
JSON. La diferencia de ~100 ms no compensa perder fiabilidad en la extracción clínica.

---

## 3. Kokoro en español sin instalar nada del sistema

**Todo en verde.**

- `espeakng-loader` trae `libespeak-ng.dylib` dentro del wheel: **no hace falta
  `brew install espeak-ng`**, que era el riesgo principal para la compuerta de 15 minutos.
- Voz `ef_dora`, `lang="es"`: 4,5 s de audio a 24 kHz generados en 1.960 ms →
  **factor de tiempo real 0,44**, o sea que sintetiza al doble de velocidad de reproducción.
- Los pesos son `kokoro-v1.0.onnx` (326 MB) y `voices-v1.0.bin` (28 MB), descargados por el
  propio `_ensure_model_files` de Pipecat a `~/.cache/pipecat/kokoro-onnx/`. Existe además
  una variante `int8` de 92 MB si hiciera falta recortar la descarga.

Ojo con la URL: el tag del release es `model-files-v1.0`, no `model-v1.0`.

---

## 4. Servicios de Pipecat

Los cuatro existen como servicios de primera parte, así que no hay que escribir ningún
`STTService`/`TTSService` a medida:

| Servicio | Módulo |
|---|---|
| `GroqSTTService` | `pipecat.services.groq.stt` — modelo por defecto `whisper-large-v3-turbo` |
| `KokoroTTSService` | `pipecat.services.kokoro.tts` |
| `SileroVADAnalyzer` | `pipecat.audio.vad.silero` |
| `SmallWebRTCTransport` | `pipecat.transports.smallwebrtc.transport` |

`GroqSTTService` hereda de `BaseWhisperSTTService`, que trabaja por segmentos: **exige un
VAD en el transporte**. `SileroVADAnalyzer` cubre eso y ya viene con el `onnxruntime` del
core de Pipecat.

Ninguna de estas dependencias arrastra PyTorch: `pipecat-ai[kokoro]` depende de
`kokoro-onnx`, no del paquete `kokoro`. El único sitio donde aparece torch es el extra
`local-smart-turn`, que no se usa.

---

## 5. Presupuesto de latencia resultante

Desde que el paciente deja de hablar hasta que empieza a sonar el agente:

| Etapa | Estimado | Nota |
|---|---:|---|
| Detección de fin de turno (VAD, `stop_secs=0.7`) | 700 ms | Es la palanca dominante; 0,7 s protege al paciente mayor de ser cortado |
| Groq `whisper-large-v3-turbo` | ~300 ms | Medido pendiente de la integración |
| Léxico colombiano + detección de inyección | <5 ms | Determinista |
| Extracción con `llama3.2:3b` | **~325 ms** | Medido |
| Motor de decisión | <1 ms | Funciones puras |
| Kokoro TTS | ~0 ms si la frase está cacheada | Las 6 preguntas canónicas y los guiones críticos se pre-sintetizan en el warmup |
| **Total, ruta típica** | **~1,3 s** | |
| **Ruta rápida** (el léxico resuelve el slot) | **~1,0 s** | 0 llamadas al LLM |
| **Ruta crítica** (bandera roja por léxico) | **~1,0 s** | 0 llamadas al LLM, audio pre-sintetizado |

Que la ruta crítica sea la más rápida del sistema no es casualidad: cuando el léxico detecta
una emergencia se salta el LLM por completo y se emite el guion determinista.

---

## 6. Cobertura del léxico sobre los turnos reales del dataset

Medido sobre los **2.071 turnos de paciente** de `dataset_final.xlsx`, ambas capas.
Determina cuántas veces hay que llamar al modelo, que es la métrica de consumo que
exige el README.

| Slot | capa 1 (limpia) | capa 2 (ruidosa) |
|---|---:|---:|
| dolor | 87% | 72% |
| herida | 84% | 62% |
| movilidad | 71% | 61% |
| fiebre | 58% | 47% |
| sueño | 51% | 33% |
| apetito | 50% | 40% |
| **Global** | **67%** | **54%** |

**61% de los turnos se resuelven sin tocar el modelo**, en 0 ms y 0 tokens. La capa
ruidosa cae al 54%, que es lo esperable: ahí es donde el paciente responde con
evasivas y donde el LLM aporta de verdad.

Distribución de intenciones sobre esos mismos turnos:

| Intención | % | Qué hace el agente |
|---|---:|---|
| respuesta | 52.9% | Rellena el slot y avanza |
| pregunta_clinica | 30.8% | Consulta el RAG y responde anclado a la evidencia |
| tercero | 8.6% | Acepta la información del cuidador y la marca como tal |
| meta | 4.2% | Repite la pregunta o dice cuánto falta. Determinista |
| social | 2.0% | Frase breve y vuelta al guion. Determinista |
| ininteligible | 1.4% | Re-pregunta más cerrada |

De ahí sale el consumo esperado: **~0.39 llamadas de extracción + ~0.31 de respuesta
anclada = ~0.70 invocaciones al modelo por turno**, y ~0.31 consultas al RAG por turno.

### Cuatro bugs que solo aparecieron midiendo contra datos reales

Una primera versión del clasificador marcaba el **42%** de los turnos como pregunta
clínica, lo que habría disparado el RAG en casi cada turno y duplicado la latencia.
Todos los falsos positivos venían de regex sin límites de palabra:

- `cita` casaba dentro de "la herida se ve limpie**cita**" → pregunta administrativa.
- `medic\w*` casaba con "según me dijo el **médic**o" → petición de receta.
- `que\s+tengo` casaba con "es **que tengo** la sopa en el fogón" → "¿qué tengo?".
- `me\s+preocup` casaba con "nada que **me preocup**e" → consulta clínica.

Los cuatro casos están fijados como tests de regresión en `test_nlu.py`. Ninguno
habría salido de una batería escrita a mano: hicieron falta los turnos del dataset.

---

## 7. Lo que estas medidas cambiaron del plan

- El esquema de extracción se escribe a mano, plano y **solo con el slot**; no se deriva de
  Pydantic.
- El acuse empático es determinista, no generado.
- La capa de léxico colombiano deja de ser un respaldo opcional y pasa a ser la ruta
  principal: es la que da los 0 ms y la que garantiza que una bandera roja nunca dependa de
  un 3B.
- **La intención y las banderas de emergencia salen del LLM.** Al meter la bandera en el
  esquema, `llama3.2:3b` marcó `no_puede_respirar` en *"como un 7, la pastilla no me lo
  quita"* —un falso positivo de emergencia— y de paso dejó de extraer el dolor. Y con la
  intención dentro, marcaba *"un 3, apenas se nota"* como `fuera_de_mision`. Ambas viven
  ahora en `app/nlu/` como reglas. Para la inyección de prompt esto además es lo único
  defendible: el detector no puede ser la misma pieza que el atacante intenta manipular.
- `llama3.2:1b` queda documentado como plan B de emergencia para máquinas lentas, con la
  pérdida de precisión declarada.

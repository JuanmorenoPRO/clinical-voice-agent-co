# Métricas medidas

Generado el 2026-08-07 17:48 UTC con `python scripts/report_metrics.py`, a partir de **23 turnos** en **4 llamadas** registrados en la base de datos.

Estos números no se escriben a mano: salen de las mismas filas de `turns` que se pueden consultar en `GET /console/conversations/{id}`.


## Latencia por turno

| Métrica | ms |
|---|---:|
| P50 | 3 |
| P95 | 3400 |
| media | 538 |
| máximo | 3470 |

En modo texto se mide desde que entra la petición. En voz, desde el `UserStoppedSpeakingFrame` —el instante en que el paciente deja de hablar—, que es el origen que pide la rúbrica. A eso hay que sumarle los 700 ms del VAD y ~500 ms del reconocimiento de voz de Groq.


## Consumo

| Métrica | Valor |
|---|---:|
| Invocaciones al modelo por turno | 0.52 |
| Turnos resueltos SIN modelo | 13/23 (57%) |
| Tokens de entrada por turno | 231 |
| Tokens de salida por turno | 11 |
| Tokens de entrada por llamada | 1327 |
| Tokens de salida por llamada | 62 |
| Consultas al RAG por turno | 0.09 |
| Consultas al RAG por llamada | 0.50 |
| Turnos degradados (modelo caído o lento) | 0 |
| Turnos por llamada | 5.8 |

La cifra que explica el resto es la primera: el léxico determinista de `app/nlu/lexicon.py` resuelve la mayoría de los turnos sin tocar el modelo, y ahí el consumo es exactamente cero.


## Costo por llamada

El LLM y la síntesis de voz corren **en local**, así que el costo real incurrido es solo el reconocimiento de voz de Groq. Se reportan las dos cifras, como pide la rúbrica para soluciones locales.

| Concepto | USD por llamada |
|---|---:|
| **Costo real incurrido** (solo Groq STT) | **0.00026** |
| LLM extrapolado a API | 0.00008 |
| TTS extrapolado a API | 0.02070 |
| **Total extrapolado a producción** | **0.02104** |

### Cómo se calcula

```
LLM  = (1327 tok_in x 0.06/1M) + (62 tok_out x 0.06/1M) = 0.00008
STT  = 5.8 turnos x 4.0s / 3600 x 0.04/hora = 0.00026
TTS  = 690 caracteres / 1000 x 0.03/1k = 0.02070
```

Referencias de precio: LLM, Together.ai (precio público de un 3B servido); STT, Groq (precio público); TTS, ElevenLabs Flash v2.5, como equivalente comercial.


## Configuración con la que se midió

- Modelo: `llama3.2:3b` vía Ollama (compuerta G3)
- Embeddings: `bge-m3`, 1024 dimensiones
- STT: `whisper-large-v3-turbo` (Groq) · TTS: Kokoro, voz `ef_dora`
- VAD: 0.7 s de silencio para dar el turno por terminado
- RAG: top 4 de 12 candidatos, umbral 0.75

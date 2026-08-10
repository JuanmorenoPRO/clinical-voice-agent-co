# Agente de voz para seguimiento postoperatorio

Tech Sphere Challenge 2026. Un agente que llama al paciente después de una cirugía,
conversa en español colombiano, entiende lo que le cuenta y decide si hay que
alertar a personal humano.

**El modelo de lenguaje es el sucesor vigente de Llama en Groq** (`llama-3.3-70b-versatile`),
uno de los cuatro puestos de la compuerta G3. El porqué está en [§ El modelo](#el-modelo-y-por-qué-ese) —
resumen: los dos permitidos que corren en la nube ya no se pueden invocar, y la
nota del reto permite usar el sucesor vigente de ese mismo proveedor.

---

## Arranque

**Requisitos:** Python 3.12+, [Ollama](https://ollama.com), y una clave gratuita de
[Groq](https://console.groq.com) (un minuto, sin tarjeta). Es la única credencial.

```bash
# 1. Servicio de embeddings. `pull` no deja el demonio arrancado, y tras un
#    reinicio nada lo vuelve a levantar: sin él, el RAG no puede embeber la
#    consulta y el agente se abstiene en vez de citar evidencia.
ollama serve &

# 2. Modelo de embeddings (1.2 GB). El LLM vive en Groq; bge-m3 corre local.
ollama pull bge-m3 &

# 3. Código y dependencias
git clone https://github.com/JuanmorenoPRO/clinical_assistant && cd clinical_assistant
python3.12 -m venv .venv && .venv/bin/pip install -r apps/backend/requirements.txt \
                                                  -r apps/backend/requirements-voice.txt

# 4. Credencial
cp .env.example .env        # pega tu GROQ_API_KEY

# 5. Índice del corpus clínico preconstruido (63 MB, ~5 s)
.venv/bin/python scripts/fetch_index.py
.venv/bin/python scripts/init_db.py

# 6. Arrancar
.venv/bin/python -m uvicorn app.main:app --app-dir apps/backend --port 8000
```

Abre <http://localhost:8000/docs>. Comprueba que todo está en pie:

```bash
curl localhost:8000/health          # {"status":"ok",…,"embeddings_ready":true}
curl localhost:8000/voice/status    # {"ready":true}
curl localhost:11434/api/tags       # Ollama en pie; debe listar bge-m3
```

Si `/health` responde `"status":"degraded"`, Ollama no está corriendo: el agente
sigue haciendo el tamizaje, pero se abstiene en vez de citar evidencia. Arréglalo
con `ollama serve &`.

**Por qué no hay Docker:** en macOS, Docker Desktop no reenvía los puertos UDP de
WebRTC, así que la voz —que es lo que evalúa G4— no funciona con el backend en un
contenedor. Quitarlo también elimina PostgreSQL de la ruta crítica: el estado vive
en SQLite y los vectores en ChromaDB, ambos sin servidor.

**Por qué el índice viene preconstruido:** indexar los 107 PDFs del corpus cuesta
11 minutos de CPU. Descargarlo cuesta 5 segundos. `scripts/build_index.py` sigue en
el repositorio y reproduce el mismo resultado, con el modelo de embeddings y los
parámetros de troceado anotados en `manifest.json`.

---

## Qué hace, y quién decide qué

El principio que ordena todo el sistema: **el modelo interpreta, el código decide.**

```
A. determinista   léxico colombiano → intención → detección de inyección   (<5 ms)
B. LLM            extracción del slot, solo si el léxico no lo resolvió    (~325 ms)
C. determinista   fusión por severidad → motor de decisión                 (<1 ms)
D. determinista   máquina de estados del guion → qué se pregunta ahora
E. LLM            respuesta anclada al RAG, solo si preguntó algo clínico
```

Si el léxico detecta una bandera de emergencia, se saltan B, D y E: se emite el
guion determinista y se alerta. **La ruta crítica es la más corta del sistema y no
pasa por el modelo**, así que ni uno caído ni uno manipulado pueden suprimir un
escalamiento. Hay un test que lo comprueba apuntando el adaptador a un puerto
muerto.

El guion de la conversación —dolor, fiebre, movilidad, herida, apetito, sueño— es
una máquina de estados en `app/agent/script.py`, no una instrucción en el prompt.
Sale del dataset oficial, donde el agente de referencia pregunta exactamente eso y
en ese orden. Que viva en código tiene tres consecuencias: un modelo de 3B no puede
perderse, la inyección de prompt no puede saltarse una pregunta ni cambiar una
decisión, y las seis preguntas se pueden pre-sintetizar en audio.

---

## Métricas medidas

Generadas con `python scripts/report_metrics.py`, que las lee de las mismas filas
de `turns` que se pueden inspeccionar en `GET /console/conversations/{id}`. **No se
escriben a mano**, para que no puedan divergir de los logs.

| Latencia por turno (modo texto) | ms |
|---|---:|
| P50 | 3 |
| P95 | 3.400 |
| media | 538 |

El P50 de 3 ms no es un truco: es que **el 57 % de los turnos no llaman al modelo**.
El léxico determinista resuelve lo formulaico —un dígito, "botando materia", "no
puedo respirar"— en cero milisegundos. Medido sobre los 2.071 turnos de paciente
del dataset oficial, la cobertura es del 61 %: 67 % en la capa limpia y 54 % en la
ruidosa, que es justo donde el modelo se gana el sueldo.

| Consumo | Valor |
|---|---:|
| Invocaciones al modelo por turno | 0,52 |
| Turnos resueltos sin modelo | 57 % |
| Tokens de entrada / salida por turno | 231 / 11 |
| Tokens de entrada / salida por llamada | 1.327 / 62 |
| Consultas al RAG por llamada | 0,50 |
| Turnos por llamada | 5,8 |

**Presupuesto de voz**, desde que el paciente deja de hablar (medido en
`scripts/spike_voice.py`):

| Etapa | ms |
|---|---:|
| VAD, confirmar fin de turno | 700 |
| Groq `whisper-large-v3-turbo` | ~500 |
| Léxico + intención + decisión | <5 |
| Extracción con el LLM (0,53× de media) | ~170 |
| TTS local, medido con Kokoro (0 si la frase está cacheada) | 0–250 |
| **Total típico** | **~1.400** |

El TTS por defecto es hoy **Piper** (`es_MX-claude-high`), que resultó ~5× más
rápido en caliente que Kokoro; Kokoro sigue disponible con `TTS_PROVIDER=kokoro` y
es el que midió `spike_voice.py`, así que esa fila es un techo, no el caso típico.

Los 700 ms del VAD son la mitad del presupuesto y son una decisión, no una
limitación: bajarlo a 0,4 s recortaría un tercio a costa de cortar a quien hace
pausas, y hablamos de pacientes de hasta 82 años recién operados.

**Costo por llamada.** El TTS corre en local y el LLM va por la cuota gratuita de
Groq, así que lo único que se paga de verdad es el reconocimiento de voz:
**USD 0,00026 por llamada**.

Extrapolado a precios de API de producción serían USD 0,021, y el reparto es
revelador: USD 0,00008 el LLM y **USD 0,0207 el TTS**. Es decir, servir esta
solución por API costaría 250 veces más en sintetizar la voz que en razonar. Eso es
lo que hace que correr el TTS en local sea una decisión económica y no solo
técnica. El cálculo completo con sus referencias de precio está en
[`docs/metricas.md`](docs/metricas.md), regenerable con `report_metrics.py`.

---

## El motor de decisión

Los umbrales no salen de la intuición: se derivaron de las **160 trayectorias** del
dataset contrastadas con su `label_ground_truth`.

```
rojo     ⟺ fiebre ≥ 38,0 ∨ secreción purulenta ∨ movilidad incapacitante ∨ dolor ≥ 8
amarillo ⟺ dos o más de: dolor ≥ 5, fiebre ≥ 37,3, eritema, inapetencia, insomnio
```

```
  real \ predicho   verde  amarillo   rojo
  verde               115         8      0
  amarillo              0        25      0
  rojo                  0         0     12
  exactitud 152/160 (95,0 %)
```

**Cero falsos negativos.** Los ocho errores son verdes escalados a amarillo, que es
la asimetría que pide la rúbrica: no alertar cuando había que alertar es la falla
catastrófica; un seguimiento de más cuesta una llamada. La derivación completa está
en [`docs/calibracion-triage.md`](docs/calibracion-triage.md) y se verifica en
menos de un segundo:

```bash
cd apps/backend && ../../.venv/bin/python -m pytest app/tests/test_triage_from_trajectories.py -q
```

Un slot sin responder **nunca** reduce el riesgo, y al cerrar la llamada, si quedó
demasiado sin averiguar, se fuerza seguimiento con la regla
`informacion_insuficiente`. Es un falso positivo comprado a propósito para cerrar
una vía de falso negativo.

---

## Conocimiento vivo

Desde la consola se sube un documento, el agente lo usa y lo cita; se borra y lo
olvida. Automatizado en `test_conocimiento_vivo`, que además comprueba lo que hace
frágil tener dos almacenes: **el borrado va primero a ChromaDB y después a SQLite**,
y toda consulta filtra por los documentos vivos de SQLite, así que un vector
huérfano no puede servirse aunque el proceso muera a mitad de camino.

```bash
curl -F "file=@protocolo.pdf" -F "procedure=Apendicectomía" localhost:8000/knowledge/documents
curl -X DELETE localhost:8000/knowledge/documents/{id}
```

### Cuándo el agente dice "no sé"

Que la similitud vectorial sea alta **no significa** que la evidencia responda. Es
el hallazgo más contraintuitivo del proyecto: medido sobre 25 preguntas, "¿cuál es
el horario de visitas?" puntúa 0.868 y "¿cuándo me quitan los puntos?" puntúa
0.795. La ajena gana. No es un defecto del modelo: todo el corpus es texto médico
postoperatorio y el coseno mide cercanía temática, no respuesta. Ningún umbral
separa esas dos.

La solución encadena cuatro filtros y solo uno es un modelo:

| Filtro | Coste | Qué descarta |
|---|---:|---|
| Umbral de similitud | 0 ms | Lo obviamente lejano |
| Nombres propios presentes | 0 ms | Protocolos y escalas inventados |
| Juicio de pertinencia (LLM) | ~130 ms | Lo que es de otro tema |
| Validación de cifras | 0 ms | Números que no están en la evidencia |

Pasó de **0/10 preguntas inventadas rechazadas a 9/10**, conservando 12/15 de las
legítimas. Los tres fallos restantes son abstenciones de más, que en clínica es el
lado correcto donde equivocarse. El detalle está en
[`docs/calibracion-rag.md`](docs/calibracion-rag.md).

### Tres cosas que encontramos en el corpus

- **Los 19 documentos de `breast_cancer/` son de cáncer de cuello uterino**, no de
  mama, mientras el procedimiento asociado es Mastectomía. Citarlos como evidencia
  de una mastectomía sería una afirmación falsa *con fuente*, que es peor que no
  responder: quedan marcados fuera de alcance y el agente declara el límite.
- Un PDF de `Appendicitis/` está escaneado sin capa de texto. Se registra con ese
  estado y se ve en la consola, en vez de desaparecer en silencio.
- Hay documentos casi duplicados; se deduplican por hash del texto normalizado.

---

## El modelo, y por qué ese

De los cuatro puestos permitidos por G3, dos ya no se pueden invocar (verificado el
7 de agosto de 2026). El puesto de "Llama 3.1 70B vía Groq" tampoco tiene ya modelo
de esa familia: Groq lo decomisionó en 2025 y apaga `llama-3.3-70b-versatile` el
16-ago-2026, así que se usa el sucesor vigente que la nota del reto permite
(lo más reciente de **Llama** disponible en Groq):

| Modelo permitido | Estado |
|---|---|
| Google Gemini 1.5 Flash | Retirado; la familia 1.5 devuelve 404 |
| Llama 3.1 70B / 3.3 70B vía Groq | Decomisionado / apagado el 16-ago-2026 |
| **Llama vigente en Groq** (`llama-3.3-70b-versatile` hoy, Llama 4 tras el apagado) | ✅ **Modelo del agente** |
| Llama 3.2 (1B/3B) local | ✅ Vivo en Ollama (alternativa) |
| Phi-3.5 Mini (3.8B) local | ✅ Vivo en Ollama |

Se eligió el **sucesor de Llama en Groq** (nube, sin que la máquina de la demo
sostenga un modelo grande) frente a los dos locales. La alternativa local
`llama3.2:3b` está aún documentada y se puede volver a elegir cambiando
`LLM_PROVIDER=ollama` + `LLM_MODEL=llama3.2:3b`.

Whisper de Groq se usa para el reconocimiento de voz y comparte la misma clave.
No compromete G3, que restringe el modelo *que razona*, no el que transcribe.

Todas las mediciones que sostienen estas decisiones —y las que las cambiaron sobre
la marcha— están en [`docs/spikes-7-agosto.md`](docs/spikes-7-agosto.md).

---

## Pruebas

```bash
cd apps/backend && ../../.venv/bin/python -m pytest app/tests -q
```

802 tests. Los del motor de decisión, el léxico y el guion —484— corren sin modelo,
sin red y sin base de datos, en un cuarto de segundo:

```bash
cd apps/backend && ../../.venv/bin/python -m pytest app/tests/test_decision.py \
                     app/tests/test_nlu.py app/tests/test_script.py -q
```

Dos tests quedan fuera del camino por defecto: son sondas del *criterio* del modelo
—si acierta una paráfrasis, si descarta una pregunta ajena— y un 70B cambia de
opinión entre ejecuciones. Se activan con `TEST_JUICIO_MODELO=1`. Lo que esos
prompts garantizan de forma determinista sí está cubierto sin modelo.

### Evaluación sobre los 160 casos del dataset

```bash
.venv/bin/python scripts/run_dataset_eval.py --capa capa1 --out reports
```

Reproduce las conversaciones del dataset a través del agente completo y escribe
[`reports/dataset-eval.md`](reports/dataset-eval.md). Sale con **código 2 si hay un
falso negativo en rojo**, que es la falla que la rúbrica considera catastrófica.

Es distinto del test del motor de decisión, y la diferencia es el punto: aquel
alimenta las reglas con el cuadro clínico ya estructurado y mide solo la
calibración de los umbrales (95%); este mete la conversación cruda y mide la cadena
entera. **Cuando un caso falla, comparar los dos dice si la culpa fue del extractor
o de las reglas**, y por eso el informe incluye la exactitud por slot.

Resultado sobre la capa limpia, con los 160 casos:

```
  real \ predicho   verde  amarillo   rojo
  verde                79       34       10
  amarillo              1       13       11
  rojo                  0        1       11

  exactitud 103/160 (64.4%)   ·   1 falso negativo en rojo
```

**Ese 64% frente al 95% de las reglas es el dato honesto de este proyecto**, y la
distancia entre ambos se reparte en dos cosas.

La primera es la extracción: entender un cuadro clínico hablando con alguien que no
tiene vocabulario médico se acierta entre el 66% y el 75% según el slot. La segunda
—y es la mayoría de los errores— es **sobre-escalamiento deliberado**: 44 casos
verdes se clasificaron como amarillo o rojo, casi todos por la política de
incertidumbre. Cuando el paciente no suelta la información, el sistema escala en vez
de asumir que está bien. Se ve con claridad por estilo de paciente: con el
colaborativo acierta el 81% y con el minimizador el 92%, pero con el evasivo cae al
17%, y casi todo ese hundimiento son verdes escalados de más.

Es la asimetría que pide la rúbrica llevada a sus últimas consecuencias: 44
seguimientos innecesarios a cambio de un solo falso negativo. Si el criterio fuera
la exactitud, habría que aflojar la política; como el criterio es no dejar pasar una
emergencia, se queda.

El contraste entre las dos evaluaciones ya pagó tres correcciones que ningún test
escrito a mano habría sugerido: la cobertura del léxico en pretérito perfecto
(*"he comido bien"*, no solo *"como bien"*), la extracción de slots que el paciente
menciona sin que se los pregunten, y la regla de fiebre referida sin termómetro.
Entre las tres subieron la exactitud del 58% al 64% y bajaron los falsos negativos
en rojo de 2 a 1.

---

## Estructura

```
apps/backend/app/
  agent/        orquestador, guion de 6 slots, banco de frases deterministas
  nlu/          léxico colombiano, clasificación de intención, fusión por severidad
  decision/     reglas puras + umbrales calibrados contra el ground truth
  rag/          ChromaDB, embeddings, ingesta en caliente, recuperación con citas
  llm/          adaptador de Ollama y esquemas de extracción restringida
  voice/        pipeline Pipecat (Groq STT → orquestador → Piper TTS)
scripts/        construcción del índice, carga del dataset, métricas, spikes
docs/           calibración del triaje, mediciones, decisiones de arquitectura
```

Los diagramas de la arquitectura y del flujo de decisión —el entregable 02, con
cada caja anotada con el archivo que la implementa— están en
[`docs/arquitectura.md`](docs/arquitectura.md).

## Licencia

MIT (ver [`LICENSE`](LICENSE)). Los PDFs del corpus son obra de sus autores y no se
redistribuyen; el índice publicado contiene solo vectores y los fragmentos
necesarios para citar.

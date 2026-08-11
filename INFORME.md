# Informe final — Agente de voz para seguimiento postoperatorio

**Tech Sphere Challenge 2026** · Juan Pablo Moreno ·
[github.com/JuanmorenoPRO/clinical_assistant](https://github.com/JuanmorenoPRO/clinical_assistant)

Este informe documenta las decisiones, el proceso y los resultados medidos. Lo que
explica *cómo levantar y usar* la solución está en el [README](README.md); lo que
explica *dónde está cada pieza*, en [`docs/arquitectura.md`](docs/arquitectura.md).
Aquí no se repite ninguno de los dos: se enlazan.

---

## 1. Resumen ejecutivo

Un agente que llama al paciente después de una cirugía, conversa en español
colombiano, conduce un tamizaje de seis síntomas, entiende lo que le cuentan —que
rara vez viene en vocabulario médico— y decide si hay que alertar a personal humano.
Corre entero en una máquina: un proceso, un puerto, dos almacenes sin servidor.

**El principio que ordena todo el sistema: el modelo interpreta, el código decide.**
Ninguna decisión clínica pasa por el LLM. El modelo hace dos cosas: extraer lo que el
paciente quiso decir y redactar la frase.

| Qué | Cuánto | Dónde se verifica |
|---|---:|---|
| Exactitud del motor de decisión sobre las 160 trayectorias | **95,0 %** | `app/tests/test_triage_from_trajectories.py` |
| Falsos negativos en rojo (motor de decisión) | **0** | ídem |
| Exactitud de la cadena completa, capa ruidosa | 54,4 % | `reports/dataset-eval.md` |
| Turnos resueltos sin llamar al modelo | 57 % | `scripts/report_metrics.py` |
| Latencia de voz, de fin de habla a inicio de audio | ~1.400 ms | `scripts/spike_voice.py` |
| Costo real por llamada | USD 0,00026 | `docs/metricas.md` |
| Preguntas inventadas que el RAG rechaza | 9/10 | `scripts/calibrate_rag.py` |

---

## 2. Modelo usado, y por qué

**`llama-3.3-70b-versatile`, servido por Groq.** Es el sucesor vigente del puesto
«Llama 3.1 70B vía Groq» de la lista permitida (compuerta G3), y se eligió después de
verificar, el 7 de agosto de 2026, cuáles de los cuatro puestos se podían invocar de
verdad:

| Modelo permitido | Estado verificado |
|---|---|
| Google Gemini 1.5 Flash | Retirado; la familia 1.5 devuelve 404 |
| Llama 3.1 70B vía Groq | Decomisionado por Groq en 2025 |
| Llama 3.3 70B vía Groq | Vigente; Groq lo apaga el 16-ago-2026 |
| **Sucesor vigente de Llama en Groq** | ✅ **El del agente** |
| Llama 3.2 (1B/3B) local · Phi-3.5 Mini local | Vivos en Ollama; alternativa documentada |

Dos de los cuatro puestos ya no se pueden invocar, y el de Llama en Groq no tiene hoy
modelo de esa familia. La nota del reto permite el sucesor vigente del mismo
proveedor, y eso es lo que se usa. Cuando Groq apague el 3.3 el 16 de agosto, la
sustitución es una variable de entorno.

**Por qué la nube y no un modelo local.** La máquina de la demo tiene que sostener
además el reconocimiento de voz, los embeddings y la síntesis; un 70B local competiría
con todo eso. La alternativa local sigue documentada y operativa: `LLM_PROVIDER=ollama`
con `LLM_MODEL=llama3.2:3b`, sin tocar una línea de código.

**Whisper de Groq** (`whisper-large-v3-turbo`) hace el reconocimiento de voz y comparte
la misma clave. No compromete G3, que restringe el modelo *que razona*, no el que
transcribe. Los embeddings (`bge-m3`) corren en local con Ollama.

---

## 3. Arquitectura, en un vistazo

El recorrido de una respuesta, desde que el paciente deja de hablar:

```
voz → Whisper (Groq) → entender (léxico determinista + extracción LLM)
    → estado del paciente → MOTOR DE DECISIÓN → guion → ¿pregunta clínica?
    → RAG → LLM redacta → aduana determinista → Piper TTS → voz
```

Si el motor de decisión da rojo, la respuesta es un **guion de seguridad verbatim** y
la alerta ya está creada: esa rama no toca el LLM ni el RAG. La ruta crítica es la más
corta del sistema.

Los cinco diagramas —componentes, flujo de un turno, máquina de estados del guion,
árbol de decisión y cadena de abstención del RAG—, con cada caja anotada con el
archivo que la implementa, están en [`docs/arquitectura.md`](docs/arquitectura.md).

---

## 4. La decisión técnica que explica el resto

**Sacar el guion de la conversación y la decisión clínica fuera del modelo.** El guion
decide *qué* se pregunta; el modelo decide *cómo* se dice.

Las seis preguntas del tamizaje —dolor, fiebre, movilidad, herida, apetito, sueño— son
una máquina de estados en `app/agent/script.py`. Los umbrales de escalamiento son
funciones puras en `app/decision/rules.py` con sus valores en `thresholds.yaml`. El
estado de la llamada vive en la base de datos, no en el contexto del modelo.

**Alternativas evaluadas y descartadas.**

- *Un agente con herramientas*, donde el modelo conduce y llama a una función para
  escalar. Descartada: pone el escalamiento —lo único que no puede fallar— del lado no
  determinista, y un modelo pequeño se pierde en un guion de seis slots.
- *El guion dentro del prompt*, que es lo más rápido de construir. Descartada por dos
  razones: una inyección de prompt puede saltarse una pregunta o cambiar una decisión,
  y mantener el historial en el contexto cuesta unos 800 tokens por turno de deriva
  acumulada.

**Qué compra esta decisión.** Tres cosas comprobables: un modelo caído no puede
suprimir un escalamiento (`test_la_bandera_roja_sobrevive_a_un_ollama_caido`), una
inyección de prompt no se salta el léxico (`test_el_lexico_no_se_salta_por_una_inyeccion`),
y las seis preguntas canónicas se pueden pre-sintetizar en audio.

**Qué cuesta, y se asume.** Rigidez: un guion en código no improvisa, así que hubo que
dedicar trabajo explícito a que el paciente pueda salirse de él —preguntas clínicas en
cualquier fase, aclaraciones («¿qué es calentura?»), reclamos de «no me respondiste»,
muletillas de duda—. Esa mitad del orquestador existe por esta decisión.

---

## 5. Cómo se trabajó: medir, cambiar, volver a medir

El proyecto se construyó con asistencia de IA (Claude Code) sobre un ciclo fijo: **una
medición antes de cada decisión, y una medición después para comprobar si sirvió.** Las
mediciones están guardadas, incluidas las que salieron mal.

**Los spikes del 7 de agosto** ([`docs/spikes-7-agosto.md`](docs/spikes-7-agosto.md))
cambiaron el plan antes de escribir el agente. El más consecuente: al meter las
banderas de emergencia y la intención en el esquema de extracción del LLM,
`llama3.2:3b` marcó *«no puede respirar»* en la frase *«como un 7, la pastilla no me lo
quita»* —un falso positivo de emergencia— y de paso dejó de extraer el dolor. Ambas
salieron del modelo y pasaron a ser reglas en `app/nlu/`. Para la inyección de prompt
esto además es lo único defendible: **el detector no puede ser la misma pieza que el
atacante intenta manipular.**

**Cinco evaluaciones guardadas del mismo experimento**, en `reports/`, sobre la capa
ruidosa del dataset:

| Corrida | Exactitud | Qué cambió |
|---|---:|---|
| `baseline` | 42,5 % | Primera cadena completa |
| `despues` | 47,5 % | Léxico en pretérito perfecto («he comido bien», no solo «como bien») |
| `final` | 48,8 % | Slots mencionados sin que se pregunten; fiebre referida sin termómetro |
| `final2` · `final3` | 48,1 % | Endurecimiento de la política de incertidumbre |

La última fila es la interesante: **el cambio bajó la exactitud y se conservó igual**,
porque subió el escalamiento de casos que se estaban cerrando sin evaluar. Cuando el
criterio es no dejar pasar una emergencia, la exactitud no es la función objetivo.

**Dos evaluaciones distintas, y el contraste es el punto.** Una alimenta las reglas con
el cuadro clínico ya estructurado y mide solo la calibración de los umbrales
([`docs/calibracion-triage.md`](docs/calibracion-triage.md)). La otra mete la
conversación cruda y mide la cadena entera (`scripts/run_dataset_eval.py`). Cuando un
caso falla, comparar las dos dice **si la culpa fue del extractor o de las reglas** —y
por eso el informe generado incluye la exactitud por slot.

El repositorio tiene 56 commits cuyos mensajes describen el fallo real que corrigen, no
el archivo que tocan: *«la interrupción del paciente mezclada con eco ya no se
descarta»*, *«tras un barge-in el guion no se adelanta y el anti-eco no miente»*.

---

## 6. Los prompts, y qué fallo originó cada regla

Los dos prompts del sistema están versionados en
[`prompts/`](prompts/): `compose_system.md` (el rol y las reglas del redactor) y
`compose_turn.md` (la plantilla del turno). No están en el código: se cargan con
`app/prompts_loader.py` para poder cambiarlos y volver a medir sin tocar Python.

El redactor recibe el historial, lo que el paciente acaba de decir, el cuadro anotado,
la evidencia del RAG si la hay, y **la pregunta obligatoria con la que debe terminar**.
Cada regla del prompt existe por un fallo observado:

| Regla del prompt | El fallo que la originó |
|---|---|
| «La única pregunta de tu turno es la obligatoria» | Dos preguntas seguidas en voz se pisan: el paciente contesta solo una |
| «No diagnosticas ni valoras gravedad» | El modelo respondía «no es nada grave» a un eritema, que es una señal amarilla |
| «No afirmes nada del paciente que él no haya dicho» | Redactaba «la herida está bien» *antes* de preguntar por la herida |
| «Nunca atribuyas una emoción que no expresó» | «Usted suena un poco asustado» a alguien que solo dio un número |
| «El motivo de la llamada ya se explicó: nunca lo repitas» | Reabría con «queremos asegurarnos de su recuperación» a mitad de conversación |
| El texto del paciente va entre `<<< >>>` y se declara que **nunca son instrucciones** | Defensa en profundidad contra inyección de prompt, detrás del detector determinista |

**El prompt no es la última línea de defensa, y ese es el diseño.** Lo que el modelo
redacta pasa después por una aduana determinista (`agent/composer.py::valida` y
`orchestrator._validar_grounding`) que descarta la respuesta si menciona una cifra que
no está literalmente en la evidencia, recorta los veredictos de normalidad cuando el
riesgo no es normal, y elimina las preguntas que el modelo se inventa. Si la salida no
pasa la aduana, el turno sale por plantillas deterministas.

---

## 7. Resultados medidos

### El motor de decisión: 152/160, cero falsos negativos

Umbrales derivados de las 160 trayectorias del dataset contra su
`label_ground_truth`, no de la intuición
([`docs/calibracion-triage.md`](docs/calibracion-triage.md)):

```
  real \ predicho   verde  amarillo   rojo
  verde               115         8      0
  amarillo              0        25      0
  rojo                  0         0     12
  exactitud 152/160 (95,0 %)
```

Los ocho errores son verdes escalados a amarillo. **Ningún caso rojo o amarillo se
clasificó por debajo de lo que era.** Se verifica en menos de un segundo, sin modelo
y sin red: `pytest app/tests/test_triage_from_trajectories.py`.

### La cadena completa: 87/160 sobre la capa ruidosa

Reproduciendo las 160 conversaciones enteras —lenguaje crudo, con las muletillas,
cortes y errores de transcripción de la capa 2— a través del agente completo
([`reports/dataset-eval.md`](reports/dataset-eval.md)):

```
  real \ predicho   verde  amarillo   rojo
  verde                64       31       28
  amarillo              1       12       12
  rojo                  0        1       11

  exactitud 87/160 (54,4 %)   ·   2 falsos negativos, 1 de ellos en rojo
```

**Este es el dato honesto del proyecto, y la distancia con el 95 % anterior es
información, no ruido.** Se reparte en dos causas y las dos están medidas:

1. **La extracción.** Entender el cuadro clínico de alguien sin vocabulario médico,
   con el audio degradado, se acierta entre el **59 %** (dolor) y el **72 %**
   (movilidad) según el slot.
2. **Sobre-escalamiento deliberado**, que es la mayoría de los errores: **59 de los
   123 casos verdes** subieron a amarillo o rojo, casi todos por la política de
   incertidumbre. Cuando el paciente no suelta la información, el sistema escala en
   vez de asumir que está bien.

Por estilo de paciente se ve de dónde sale:

| Estilo | Casos | Exactitud | Falsos negativos |
|---|---:|---:|---:|
| minimizador | 37 | 81 % | 1 |
| colaborativo | 32 | 69 % | 0 |
| confundido | 35 | 51 % | 0 |
| ansioso | 27 | 44 % | 0 |
| **evasivo** | 29 | **17 %** | 1 |

El hundimiento con el paciente evasivo son casi todos verdes escalados de más: es
exactamente el comportamiento que se diseñó. **59 seguimientos innecesarios a cambio
de un solo falso negativo en rojo.** Si el criterio fuera la exactitud, habría que
aflojar la política de incertidumbre; como el criterio es no dejar pasar una
emergencia, se queda.

### El RAG: de rechazar 0 de 10 preguntas inventadas a rechazar 9 de 10

El hallazgo más contraintuitivo del proyecto: **la similitud vectorial no dice si la
evidencia responde**. Medido sobre 25 preguntas, *«¿cuál es el horario de visitas?»*
—que el corpus no responde— puntúa **0,868**, y *«¿cuándo me quitan los puntos?»*
—que sí— puntúa **0,795**. La pregunta ajena gana. No es un defecto del modelo: todo
el corpus es texto médico postoperatorio y el coseno mide cercanía temática, no
respuesta. **Ningún umbral separa esas dos**, así que subirlo no era la solución.

La cadena de cuatro filtros —umbral, nombres propios presentes, juicio de
pertinencia y validación de cifras— pasó de **0/10 a 9/10** preguntas inventadas
rechazadas, conservando 12 de 15 legítimas. Los tres fallos restantes son
abstenciones de más, que en clínica es el lado correcto donde equivocarse
([`docs/calibracion-rag.md`](docs/calibracion-rag.md)).

### Latencia, consumo y costo

| Métrica | Valor |
|---|---:|
| Latencia por turno, modo texto · P50 | 3 ms |
| Latencia por turno, modo texto · P95 | 3.400 ms |
| Presupuesto de voz, de fin de habla a inicio de audio | ~1.400 ms |
| Invocaciones al modelo por turno | 0,52 |
| Turnos resueltos **sin** modelo | 57 % |
| Tokens de entrada / salida por turno | 231 / 11 |
| Consultas al RAG por llamada | 0,50 |
| **Costo real por llamada** | **USD 0,00026** |

El P50 de 3 ms no es un truco: **el 57 % de los turnos no llaman al modelo**, porque
el léxico determinista resuelve lo formulaico —un dígito, «botando materia», «no
puedo respirar»— en cero milisegundos. De los ~1.400 ms del presupuesto de voz, 700
son el detector de actividad esperando a confirmar que el paciente terminó: la mitad
del presupuesto es una decisión de diseño, no una limitación.

El costo real es solo el reconocimiento de voz, porque el TTS corre en local y el LLM
va por la cuota gratuita de Groq. Extrapolado a precios de API de producción serían
USD 0,021 por llamada, con un reparto revelador: **USD 0,00008 el LLM y USD 0,0207 el
TTS**. Servir esto por API costaría 250 veces más en sintetizar la voz que en
razonar, y eso convierte correr el TTS en local en una decisión económica, no solo
técnica. El cálculo completo está en [`docs/metricas.md`](docs/metricas.md).

---

## 8. Configuración

La configuración completa está en `.env.example`, con valor por defecto para todo menos
la clave de Groq. Los parámetros que son una decisión, y no un valor arbitrario:

| Parámetro | Valor | Por qué |
|---|---|---|
| `LLM_MODEL` | `llama-3.3-70b-versatile` | Compuerta G3, § 2 |
| `LLM_TIMEOUT_S` | `2.5` | Un turno de voz que tarda más ya se siente roto; al expirar, caen las plantillas |
| `EMBEDDING_MODEL` | `bge-m3` | Multilingüe: el corpus mezcla español e inglés, y la pregunta llega en español |
| `RAG_TOP_K` / `RAG_FETCH_K` | `4` / `12` | Se sobre-recupera y se filtra en código, con tope por documento: si no, el top-k se llena de fragmentos casi idénticos del mismo PDF |
| `VAD_STOP_SECS` | `0.7` | La mitad del presupuesto de latencia, y es deliberado: a 0,4 s se corta a quien hace pausas, y son pacientes de hasta 82 años recién operados |
| `SILENCE_MAX_ATTEMPTS` | `3` | Sondear → avisar → colgar. Al segundo se cuelga a quien tarda en volver al teléfono |
| `BARGE_IN_VAD` | `false` | Sin cancelación de eco, el agente se interrumpiría con su propia voz. Se usa barge-in confirmado, tras el filtro anti-eco |
| `TTS_PROVIDER` | `piper` | ~5× más rápido en caliente que Kokoro, con voz nativa en español |

---

## 9. Capturas del demo

> **Nota:** las imágenes van en `docs/img/`. Cada pie describe qué debe verse.

**(a) Escalamiento y alerta.** La consola con la alerta creada, su regla disparada
(`herida_purulenta`) y el nivel de riesgo.

![Alerta en la consola](docs/img/01-alerta.png)

**(b) La traza de un turno.** `GET /console/conversations/{id}`: qué se entendió, qué
regla saltó, qué documentos se citaron, tokens y latencia. De estas mismas filas salen
las métricas del README.

![Traza del turno](docs/img/02-traza-turno.png)

**(c) El resumen de cierre.** Paciente, procedimiento, síntomas reportados, la decisión
turno a turno, las referencias usadas y los próximos pasos.

![Resumen de la llamada](docs/img/03-resumen.png)

**(d) Conocimiento vivo.** Documento subido desde la consola, respuesta del agente
citando documento y página, y la abstención tras borrarlo.

![Conocimiento vivo](docs/img/04-conocimiento-vivo.png)

---

## 10. Límites conocidos, y qué haría con dos semanas más

Lo que hoy no funciona bien, en orden de cuánto pesa:

1. **La extracción es el cuello de botella, no las reglas.** Entender un cuadro clínico
   hablando con alguien sin vocabulario médico se acierta entre el 59 % y el 72 % según
   el slot. Es la primera pieza que atacaría, y concretamente el reconocimiento de la
   evasión: con el paciente evasivo la exactitud cae al 17 %, y casi todo ese
   hundimiento son verdes escalados de más.
2. **La política de incertidumbre sobre-escala.** Es deliberado —comprar falsos
   positivos para cerrar una vía de falso negativo—, pero hoy usa un umbral global de
   completitud. Calibrarla **por slot** recuperaría exactitud sin tocar el recall de
   rojos.
3. **Los umbrales son 160 casos, no una población.** Están derivados del
   `label_ground_truth` del dataset del reto. Antes de un piloto real habría que
   revalidarlos con datos del hospital.
4. **Sin cancelación de eco.** Por eso el barge-in instantáneo por VAD está apagado y se
   usa el confirmado. Con AEC se podría cortar el TTS a ~0,2 s de detectar voz.
5. **19 documentos del corpus están fuera de alcance a propósito.** Los de
   `breast_cancer/` son de cáncer de cuello uterino, no de mama, mientras el
   procedimiento asociado es Mastectomía. Citarlos sería una afirmación falsa *con
   fuente*, que es peor que no responder, así que el agente declara el límite.

---

## Cómo reproducir todo lo que dice este informe

```bash
# Levantar (detalle completo en el README)
ollama serve & ; ollama pull bge-m3
.venv/bin/python scripts/fetch_index.py && .venv/bin/python scripts/init_db.py
.venv/bin/python -m uvicorn app.main:app --app-dir apps/backend --port 8000

# Calibración del motor de decisión (sin modelo, sin red, <1 s)
cd apps/backend && ../../.venv/bin/python -m pytest app/tests/test_triage_from_trajectories.py -q

# La cadena completa sobre los 160 casos del dataset
COMPOSE_PROVIDER=mock .venv/bin/python scripts/run_dataset_eval.py --capa capa1 --out reports

# Métricas de latencia, consumo y costo, leídas de la base
.venv/bin/python scripts/report_metrics.py

# Cuándo el agente dice "no sé"
.venv/bin/python scripts/calibrate_rag.py
```

**Documentos relacionados:** [README](README.md) ·
[arquitectura](docs/arquitectura.md) · [calibración del triaje](docs/calibracion-triage.md) ·
[calibración del RAG](docs/calibracion-rag.md) · [métricas](docs/metricas.md) ·
[spikes](docs/spikes-7-agosto.md)

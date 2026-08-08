# Cuándo el agente debe decir "no sé"

Reproducible con `.venv/bin/python scripts/calibrate_rag.py`, sobre 15 preguntas que
el corpus responde y 10 que no.

La rúbrica pregunta qué hace el agente ante algo que no está en su conocimiento: si
declara el límite o improvisa. Improvisar **citando un documento** es el peor caso
posible, porque la cita hace pasar por verificada una afirmación falsa. Este
documento explica por qué la solución obvia no funciona y qué hubo que hacer.

---

## Lo que no funciona: subir el umbral de similitud

El umbral heredado era 0.55, calibrado para `voyage-3`. Con `bge-m3` las
similitudes se apiñan entre 0.72 y 0.87, así que 0.55 deja pasar absolutamente
todo. La reacción natural es subirlo. **No sirve**, y no por poco:

| Pregunta | Similitud | ¿El corpus la responde? |
|---|---:|---|
| ¿Cuál es el horario de visitas del hospital? | 0.868 | **No** |
| ¿Cuánto cuesta la consulta de control? | 0.838 | **No** |
| ¿Cómo se hace una cesárea? | 0.827 | **No** |
| ¿Cuándo puedo volver a trabajar? | 0.822 | Sí |
| ¿Por qué me duele el hombro tras la cirugía de vesícula? | 0.815 | Sí |
| ¿Qué grado de la escala Zafiro tengo? | 0.796 | **No** |
| ¿Cuándo me quitan los puntos? | 0.795 | Sí |

Las preguntas ajenas puntúan **más alto** que las legítimas. No es un defecto del
modelo de embeddings: es lo que mide. Todo el corpus es texto médico
postoperatorio, y el coseno mide cercanía temática, no si un texto contiene la
respuesta. Cualquier umbral que rechace el horario de visitas (0.868) ya habrá
rechazado antes media docena de preguntas buenas.

Barriendo umbrales, el mejor daba 19/25 y aun así colaba 5 preguntas inventadas.

## Lo que tampoco funciona: solapamiento léxico

Segundo intento: exigir que alguna palabra de contenido de la pregunta aparezca en
la evidencia. Atrapa bien los protocolos inventados, pero se estrella contra la
morfología del español. "¿Me puedo **bañar**?" no casa con "**baño** diario";
"volver a **trabajar**" no casa con "reincorporación **laboral**". Rechazaba 4 de
15 preguntas legítimas.

## Lo que sí funciona: separar por tipo de error

Los dos fallos que hay que evitar son distintos y piden herramientas distintas.

**Nombre propio inexistente** — "el protocolo Esmeralda", "la escala Zafiro". Los
nombres propios no se declinan, así que la coincidencia exacta es fiable. Se
comprueba con código en `_nombres_propios_presentes`: si un nombre propio de la
pregunta no aparece literalmente en la evidencia, no se responde. Cuesta 0 ms.

Detalle que costó una iteración: el regex debe exigir que la mayúscula venga
**precedida de una palabra en minúscula**. Sin eso, "¿Cuándo me quitan los
puntos?" tomaba "Cuándo" por nombre propio y rechazaba la pregunta entera.

**Tema distinto** — "¿cuál es el horario de visitas?" frente a un texto sobre
cuidados de la herida. Aquí sí hace falta comprensión, y es justo lo que un 3B
puede hacer: no recordar, sino **reconocer** si un texto que tiene delante viene al
caso. Se le pregunta con salida restringida a un enum de dos valores, ~5 tokens,
~130 ms (`OllamaAdapter.evidencia_responde`).

Ese prompt necesitó dos calibraciones, y ambas enseñan lo mismo sobre los modelos
pequeños:

- La primera decía *"ante la duda responde no"*. El modelo tumbó **7 de 15**
  preguntas legítimas: una instrucción de prudencia se convierte en negarlo todo.
- La segunda añadía *"responde no si la pregunta menciona un nombre propio que no
  aparece en el texto"*. Entonces rechazaba también los nombres propios que **sí**
  estaban. Un 3B no sabe verificar la presencia de un término; por eso esa
  comprobación acabó en código.

Lo que funciona es pedirle solo lo que sabe hacer, con ejemplos de los dos casos.

---

## Resultado

| | Antes | Después |
|---|---:|---:|
| Responde preguntas del corpus | 15/15 | 12/15 |
| Rechaza preguntas ajenas | **0/10** | **9/10** |
| Aciertos | 15/25 | **21/25** |

Los tres fallos que quedan son abstenciones de más: el agente dice "no sé" cuando
podría haber respondido. En un agente clínico ese es el lado correcto donde
equivocarse, y es coherente con la asimetría que aplica el motor de decisión.

## Lo que esto cambió en la arquitectura

`rag/retrieve.py` ya no decide si hay evidencia suficiente; solo recupera y puntúa.
La decisión de responder o abstenerse vive en el orquestador, que encadena tres
filtros de coste creciente:

```
1. umbral de similitud          0 ms    descarta lo obviamente lejano
2. nombres propios presentes    0 ms    descarta protocolos y escalas inventados
3. juicio de pertinencia      ~130 ms   descarta lo que es de otro tema
```

Y por debajo de todo sigue la validación determinista de `_validar_grounding`: si
la respuesta menciona una cifra que no aparece en la evidencia, se sustituye por la
abstención. Cuatro capas, y solo una de ellas es un modelo.

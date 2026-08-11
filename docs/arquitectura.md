# Arquitectura y flujo de decisión

Entregable 02 del reto. Una vista de conjunto y cinco secciones de detalle: cómo
está montado el sistema, qué pasa en un turno, cómo conduce el agente la
conversación, cómo decide si alerta y cuándo se calla porque no sabe.

**Cada caja lleva el archivo y la función que la implementan.** El README explica
*por qué* está construido así; esto explica *dónde* está.

El principio que ordena todo lo que sigue: **el modelo interpreta, el código
decide.** Ninguna decisión clínica —el nivel de riesgo, la alerta, el guion de
seguridad, qué se pregunta ahora— pasa por el LLM. El modelo solo hace dos cosas:
extraer lo que el paciente quiso decir y redactar la frase.

---

## Vista de conjunto

El recorrido completo de una respuesta, desde que el paciente deja de hablar hasta
que vuelve a oír al agente. Unos **1.400 ms** típicos.

```mermaid
flowchart TB
    VOZ(["🎙️ El paciente habla"])
    STT["<b>Whisper large v3 turbo</b><br/>reconocimiento de voz · Groq"]
    ENT["<b>Entender</b><br/>léxico colombiano determinista<br/>+ extracción con el LLM si hace falta"]
    EST[("<b>Estado del paciente</b><br/>síntomas acumulados + guion en curso")]
    DEC["<b>Motor de decisión</b><br/>reglas puras · verde / amarillo / rojo"]
    SCR["<b>Guion</b><br/>qué se pregunta ahora"]
    PREG{"¿el paciente<br/>preguntó algo clínico?"}
    RAG["<b>RAG</b><br/>corpus clínico + filtros de abstención"]
    LLM["<b>Llama 3.3 70B</b> · Groq<br/>redacta la frase"]
    ADU["<b>Aduana determinista</b><br/>sin cifras inventadas,<br/>sin tranquilizar de más"]
    GS["<b>Guion de seguridad</b><br/>verbatim, sin modelo"]
    ALE[/"<b>Alerta a enfermería</b><br/>persistida y visible en la consola"/]
    RESP["<b>Respuesta</b>"]
    TTS["<b>Piper TTS</b><br/>voz en español, local"]
    OIR(["🔊 El paciente escucha"])

    VOZ --> STT --> ENT --> EST --> DEC
    DEC -->|"sigue la llamada"| SCR --> PREG
    PREG -->|"sí"| RAG --> LLM
    PREG -->|"no"| LLM
    LLM --> ADU --> RESP --> TTS --> OIR

    DEC -->|"rojo"| GS --> RESP
    DEC -->|"rojo o amarillo"| ALE

    classDef det fill:#dbeafe,stroke:#1e40af,color:#0b1220
    classDef mod fill:#fef3c7,stroke:#92400e,color:#0b1220
    classDef crit fill:#fee2e2,stroke:#991b1b,color:#0b1220
    class DEC,SCR,ADU,PREG det
    class STT,LLM,RAG mod
    class GS,ALE crit
```

<sub>Azul: determinista, el código decide. Ámbar: el modelo. Rojo: la ruta de escalamiento.</sub>

El **estado del paciente** es acumulativo y vive en la base de datos, no en el
contexto del modelo: cada turno añade lo que se entendió y nada lo baja de nivel.
Por eso el ciclo vuelve a empezar arriba con todo lo que ya se sabía.

Tres cosas que este dibujo ya dice, y que el resto del documento solo detalla:

- **El motor de decisión va antes que el guion y antes del modelo.** Si el cuadro
  es rojo, la respuesta es un guion verbatim y la alerta ya está creada: esa rama
  no toca el LLM ni el RAG, así que ni un modelo caído ni uno manipulado pueden
  suprimir un escalamiento.
- **El RAG solo se consulta si el paciente preguntó algo clínico** — de media,
  0,50 consultas por llamada al modelo. El resto del tiempo el agente está
  preguntando, no respondiendo.
- **Nada de lo que redacta el modelo sale sin pasar por la aduana**, que es código
  y no otro modelo.

Cada pieza, con su archivo y su función, en las cinco secciones que siguen.

---

## 1. Arquitectura de componentes

Un proceso, un puerto, dos almacenes sin servidor. No hay Docker, ni Node, ni
PostgreSQL: el porqué de cada ausencia está en el README.

```mermaid
flowchart TB
    UI["<b>Navegador</b><br/>consola de enfermería + interfaz de llamada<br/>app/static/index.html"]

    subgraph API["Proceso FastAPI · app/main.py · puerto 8000"]
        RC["routers/<br/>conversation.py"]
        RK["routers/<br/>knowledge.py"]
        RN["routers/<br/>console.py"]
        RV["routers/voice.py<br/>señalización WebRTC"]
        VOZ["voice/pipeline.py<br/>run_bot · ClinicalProcessor<br/><i>import perezoso</i>"]

        ORQ["<b>agent/orchestrator.py</b><br/>process_turn_async"]

        NLU["nlu/<br/>lexicon · intent<br/>polaridad · merge<br/>procedimiento"]
        GUION["agent/script.py<br/>máquina<br/>de estados"]
        DEC["decision/<br/>engine · rules<br/>thresholds.yaml<br/>safety_scripts"]
        RED["agent/composer.py<br/>+ phrasing.py<br/>redactor y frases"]
        RAG["rag/<br/>retrieve · store<br/>embeddings · ingest"]
        LLMA["llm/<br/>adapter · factory<br/>groq_adapter<br/>ollama_adapter"]
        RES["summary/service.py<br/>close_conversation<br/>build_summary"]
    end

    SQL[("SQLite · app/db.py<br/>patients · documents · conversations<br/>turns · alerts · summaries")]
    CHR[("ChromaDB<br/>rag/store.py")]
    GROQ{{"Groq API<br/>llama-3.3-70b-versatile<br/>whisper-large-v3-turbo"}}
    OLL{{"Ollama local<br/>bge-m3"}}

    UI -->|"HTTP"| RC
    UI -->|"HTTP"| RK
    UI -->|"HTTP"| RN
    UI -->|"SDP offer"| RV
    UI <-->|"audio WebRTC"| VOZ
    RV -.-> VOZ

    VOZ -->|"process_turn"| ORQ
    RC --> ORQ
    RK --> RAG
    RN --> SQL

    ORQ --> NLU
    ORQ --> GUION
    ORQ --> DEC
    ORQ --> RED
    ORQ --> RAG
    ORQ --> LLMA
    ORQ --> RES

    LLMA --> GROQ
    VOZ -->|"STT"| GROQ
    RAG --> OLL
    RAG --> CHR
    RAG --> SQL
    DEC --> SQL
    RES --> SQL

    classDef det fill:#dbeafe,stroke:#1e40af,color:#0b1220
    classDef mod fill:#fef3c7,stroke:#92400e,color:#0b1220
    classDef sto fill:#e5e7eb,stroke:#374151,color:#0b1220
    class NLU,GUION,DEC det
    class LLMA,GROQ,OLL mod
    class SQL,CHR sto
```

<sub>Azul: determinista. Ámbar: modelo. Gris: almacenamiento.</sub>

### El pipeline de voz

Siete etapas de Pipecat, sin ningún `STTService` ni `TTSService` a medida
(`app/voice/pipeline.py :: run_bot`). El `VADProcessor` va **explícito en la
cadena** porque `TransportParams` descarta `vad_analyzer` en silencio, y sin VAD
`GroqSTTService` no transcribe nada: la llamada entera se leería como silencio.

```mermaid
flowchart LR
    MIC(["micrófono"]) --> TIN["transport.input<br/>SmallWebRTCTransport"]
    TIN --> VAD["VADProcessor<br/>SileroVADAnalyzer"]
    VAD --> STT["GroqSTTService<br/>whisper-large-v3-turbo"]
    STT --> TUR["UserTurnProcessor<br/>fin de turno"]
    TUR --> CLI["ClinicalProcessor<br/>eco · barge-in · silencios<br/>llama a process_turn"]
    CLI --> TTS["TTS · _build_tts<br/>Piper es_MX-claude-high<br/>Kokoro opcional"]
    TTS --> TOUT["transport.output"]
    TOUT --> ALT(["altavoz"])

    classDef det fill:#dbeafe,stroke:#1e40af,color:#0b1220
    class VAD,TUR,CLI det
```

**El orden del borrado no es decorativo.** Tener dos almacenes hace frágil el
conocimiento vivo, así que `rag/ingest.py::delete_document` borra **primero en
ChromaDB y después en SQLite**, y toda consulta filtra por los documentos vivos
de SQLite
(`rag/retrieve.py::_allowed_document_ids`). Un vector huérfano no puede servirse
aunque el proceso muera a mitad de camino. Lo comprueba `test_conocimiento_vivo`.

### Superficie HTTP

| Método y ruta | Archivo |
|---|---|
| `POST /conversation/turn` | `routers/conversation.py` |
| `GET /conversation/apertura` | `routers/conversation.py` |
| `POST /conversation/{conversation_id}/close` | `routers/conversation.py` |
| `GET · POST /knowledge/documents` · `DELETE /knowledge/documents/{document_id}` | `routers/knowledge.py` |
| `GET /console/patients` · `/console/alerts` · `/console/conversations` · `/console/conversations/{id}` | `routers/console.py` |
| `POST /console/alerts/{alert_id}/attend` | `routers/console.py` |
| `GET /voice/status` · `POST · PATCH /voice/offer` | `routers/voice.py` |
| `GET /health` · `GET /` | `main.py` |

Pipecat se importa **perezosamente** desde `routers/voice.py`: quien nunca abre
una llamada no paga el arranque de la pila de voz.

---

## 2. Flujo de un turno

Lo que ocurre entre que el paciente termina de hablar y el agente contesta. Las
etiquetas A–E son las mismas que usa el README.

```mermaid
flowchart TB
    IN["Texto del paciente<br/>orchestrator.process_turn_async"]
    HIST["Historial y estado desde SQLite<br/>_prior_turns · _acumular"]

    subgraph A["A · determinista &lt;5 ms"]
        LEX["nlu/lexicon.py<br/>normalize · respuesta_polar<br/>léxico colombiano"]
        INT["nlu/intent.py :: classify<br/>is_injection · pide_aclaracion<br/>reclama_respuesta · menciona_automedicacion"]
    end

    SIL{"¿silencio?"}
    EMG{"¿bandera<br/>de emergencia?"}

    B["B · LLM ~325 ms<br/>llm.extract<br/>llm/groq_adapter.py<br/>extraction_schema.py"]

    subgraph C["C · determinista &lt;1 ms"]
        MER["nlu/merge.py :: merge_symptoms<br/>fusión por severidad, monótona"]
        ENG["decision/engine.py :: evaluate<br/>decision/rules.py · thresholds.yaml"]
    end

    D["D · determinista<br/>agent/script.py :: next_action<br/>qué se pregunta ahora"]

    RUTA{"¿qué toca decir?"}

    GS["Guion de seguridad<br/>decision/safety_scripts.py :: script_for<br/>sin modelo, sin RAG"]
    PLA["Frase de plantilla<br/>agent/phrasing.py<br/>cierre · sondeo · pregunta"]

    subgraph E["E · redactor anclado"]
        RET["rag/retrieve.py :: retrieve<br/>ChromaDB + bge-m3"]
        PER["llm.pregunta_es_del_dominio<br/>¿la evidencia responde?"]
        CTX["agent/composer.py :: build_context"]
        CMP["llm.compose_reply<br/>el guion decide QUÉ, el LLM el CÓMO"]
        VAL["composer.valida + _validar_grounding<br/>aduana determinista"]
        FB["_texto_fallback<br/>plantillas si el redactor falla"]
        RET --> PER --> CTX --> CMP --> VAL
        VAL -.->|"no pasa"| FB
    end

    APL["script.apply<br/>nuevo CallState"]
    ALE["_crear_alerta_si_procede<br/>_crear_alerta_procedimiento<br/>_crear_alerta_sin_respuesta"]
    PER2["Persistir Turn<br/>síntomas, reglas, citas, tokens, latencia"]
    CIE{"call_ended?"}
    RESU["summary/service.py :: close_conversation<br/>evaluate final=True + build_summary"]
    OUT["TurnResponse"]

    IN --> HIST --> A
    A --> SIL
    SIL -->|"sí · se salta el modelo"| C
    SIL -->|"no"| EMG
    EMG -->|"sí"| C
    EMG -->|"no"| B --> C
    C --> D --> RUTA
    RUTA -->|"crítico, primer turno"| GS
    RUTA -->|"silencio, cierre, rechazo"| PLA
    RUTA -->|"conversación y preguntas clínicas"| E
    GS --> APL
    PLA --> APL
    E --> APL
    APL --> ALE --> PER2 --> CIE
    CIE -->|"sí"| RESU --> OUT
    CIE -->|"no"| OUT

    classDef det fill:#dbeafe,stroke:#1e40af,color:#0b1220
    classDef llm fill:#fef3c7,stroke:#92400e,color:#0b1220
    classDef crit fill:#fee2e2,stroke:#991b1b,color:#0b1220
    class LEX,INT,MER,ENG,D,APL,PLA,VAL,FB det
    class B,CMP,PER llm
    class GS,EMG crit
```

**La ruta crítica es la más corta y no pasa por el modelo.** Si el léxico levanta
una bandera de emergencia, el turno salta B y E: se emite
`decision.safety_script` y se alerta. Ni un modelo caído ni un modelo manipulado
pueden suprimir un escalamiento — hay un test que lo comprueba apuntando el
adaptador a un puerto muerto.

El silencio corta por la misma razón de fondo: no hay nada que extraer de él, y
gastar segundos de modelo solo alarga la espera de alguien que ya no contesta
(`orchestrator.py`, `intent.classify(text) == "silencio"`).

Todo lo que decide el turno vive en `CallState` (`agent/script.py`), que se
persiste en la fila del `Turn` — **no en el contexto del modelo**. Eso recorta
unos 800 tokens por turno y elimina la deriva del modelo que se olvida de lo que
ya preguntó.

---

## 3. Máquina de estados del guion

Las seis preguntas del tamizaje son código, no una instrucción en el prompt. Una
inyección de prompt no puede saltarse una pregunta ni cambiar una decisión, y las
frases se pueden pre-sintetizar en audio.

```mermaid
stateDiagram-v2
    direction TB
    [*] --> TAMIZAJE : phrasing.APERTURA<br/>ya trae la pregunta de dolor

    state TAMIZAJE {
        direction LR
        dolor --> fiebre
        fiebre --> movilidad
        movilidad --> herida
        herida --> apetito
        apetito --> sueno
    }

    TAMIZAJE --> TAMIZAJE : sin respuesta<br/>MAX_REPREGUNTAS = 2<br/>luego el slot queda UNKNOWN
    TAMIZAJE --> ABIERTO : seis slots resueltos o perdidos
    ABIERTO --> CONFIRMACION : nada más que contar
    CONFIRMACION --> CIERRE : despedida, niega_mas_temas<br/>o MAX_TURNOS_CONFIRMACION = 2
    CIERRE --> TERMINADA

    TAMIZAJE --> ESCALAMIENTO : decision CRÍTICO
    ABIERTO --> ESCALAMIENTO : decision CRÍTICO
    CONFIRMACION --> ESCALAMIENTO : decision CRÍTICO
    ESCALAMIENTO --> CONFIRMACION : guion entregado<br/>el paciente sigue hablando
    ESCALAMIENTO --> TERMINADA : emergencia_123<br/>se cuelga rápido

    TAMIZAJE --> TERMINADA : intent rechazo
    CONFIRMACION --> TERMINADA : sin_progreso = 5<br/>o sin_respuesta = 3
```

Tres contadores gobiernan la salida, y son distintos a propósito
(`agent/script.py`):

| Contador | Tope | Qué pasa al llegar |
|---|---:|---|
| `repreguntas[slot]` | `MAX_REPREGUNTAS = 2` | El slot se marca UNKNOWN y el guion sigue. Nunca `False`: un dato que no se obtuvo no es un dato negativo. |
| `sin_progreso` | `SIN_PROGRESO_OFRECER_SALIDA = 3` | Se le ofrece al paciente que lo llame una enfermera. |
| `sin_progreso` | `SIN_PROGRESO_CERRAR = 5` | Se cierra la llamada — y `evaluate(final=True)` decide qué era. |
| `sin_respuesta` | `MAX_SILENCIOS = 3` | Escalera de presencia: **sondear → avisar → colgar**, con alerta de llamada sin respuesta. |
| `turnos` en confirmación | `MAX_TURNOS_CONFIRMACION = 2` | Se cierra. Una pregunta clínica contestada **no** gasta este presupuesto. |

Los silencios cuentan **seguidos**: cualquier cosa que diga el paciente devuelve
el contador a cero. Un silencio suelto en mitad de una llamada normal no puede
acercarla a colgarse. La parte pura de esa escalera —incluido el escalón GENTLE
(*"tómese su tiempo"*), que se emite localmente y ni siquiera crea un turno— vive
en `app/voice/silence.py`; el reloj está en `ClinicalProcessor`.

---

## 4. Flujo de decisión y escalamiento

Código puro, sin IA y sin motor de reglas genérico. Los umbrales salen de
`decision/thresholds.yaml`, calibrados contra las 160 trayectorias del dataset y
su `label_ground_truth`.

```mermaid
flowchart TB
    S["Symptoms acumulados<br/>decision/engine.py :: evaluate"]
    R1{"¿alguna de las seis<br/>banderas de emergencia?"}
    R2{"¿alguna bandera roja?"}
    R3{"¿dos o más señales amarillas?"}
    FIN{"¿es el cierre<br/>de la llamada?<br/>final=True"}
    COMP{"completeness"}

    E123["CRÍTICO · rojo<br/>escalation_action = emergencia_123<br/>sangrado_abundante · dificultad_respiratoria<br/>perdida_consciencia · dolor_toracico<br/>estado_mental_alterado · convulsion"]
    ROJO["CRÍTICO · rojo<br/>escalation_action = enfermeria_prioritaria<br/>fiebre_38 (≥ 38.0 °C) · dolor_severo (≥ 8)<br/>herida_purulenta · movilidad_incapacitante<br/>fiebre_referida_con_signos"]
    AMAR["ALTO · amarillo<br/>escalation_action = seguimiento<br/>vigilancia_multiples_signos (score ≥ 2)<br/>dolor_no_controlado (&gt; 7 y medicación inefectiva)"]
    VERDE["NORMAL · verde<br/>escalation_action = ninguna"]

    NOEV["CRÍTICO · no_se_pudo_evaluar<br/>enfermeria_prioritaria"]
    INSU["ALTO · informacion_insuficiente<br/>seguimiento"]

    GUION["safety_scripts.py :: script_for<br/>guion determinista, verbatim"]
    RESULT["DecisionResult<br/>risk_level · triage_color · escalation_action<br/>triggered_rules · safety_script · completeness"]
    ALERTA[("Alert en SQLite<br/>deduplicada por reglas nuevas")]

    S --> R1
    R1 -->|"sí"| E123
    R1 -->|"no"| R2
    R2 -->|"sí"| ROJO
    R2 -->|"no"| R3
    R3 -->|"sí"| AMAR
    R3 -->|"no"| FIN
    FIN -->|"no · turno intermedio"| VERDE
    FIN -->|"sí"| COMP
    COMP -->|"&lt; 0.34<br/>menos de 2 slots de 6"| NOEV
    COMP -->|"&lt; 0.5 · o un slot capaz de rojo<br/>sin responder con señal amarilla"| INSU
    COMP -->|"suficiente"| VERDE

    AMAR -.->|"al cerrar, con completeness &lt; 0.34"| NOEV

    E123 --> GUION
    ROJO --> GUION
    NOEV --> GUION
    GUION --> RESULT
    AMAR --> RESULT
    INSU --> RESULT
    VERDE --> RESULT
    RESULT --> ALERTA

    classDef rojo fill:#fee2e2,stroke:#991b1b,color:#0b1220
    classDef amar fill:#fef3c7,stroke:#92400e,color:#0b1220
    classDef verde fill:#dcfce7,stroke:#166534,color:#0b1220
    class E123,ROJO,NOEV,GUION rojo
    class AMAR,INSU amar
    class VERDE verde
```

**Señales amarillas** (`rules.yellow_signals`), suman 1 cada una y con dos basta:
dolor ≥ 5, temperatura ≥ 37.3 °C, eritema leve en la herida, apetito muy
disminuido, sueño muy alterado.

**La política de incertidumbre solo corre al cerrar y solo sube.** Un slot sin
responder nunca reduce el riesgo; una llamada que no se pudo evaluar no se
despide como si el paciente estuviera bien. Es un falso positivo comprado a
propósito para cerrar una vía de falso negativo — la asimetría que pide la
rúbrica.

Sobre los 160 casos del dataset, las reglas aciertan 152 (95,0 %) con **cero
falsos negativos**; los ocho errores son verdes escalados a amarillo. La
derivación completa está en [`calibracion-triage.md`](calibracion-triage.md) y se
verifica con `app/tests/test_triage_from_trajectories.py`.

Lo que queda registrado cuando se decide alertar —y el resumen de cierre con
paciente, procedimiento, síntomas, decisiones turno a turno, citas y próximos
pasos— lo construye `summary/service.py :: build_summary`.

---

## 5. Cuándo el agente dice "no sé"

Que la similitud vectorial sea alta **no significa** que la evidencia responda:
medido sobre 25 preguntas, *"¿cuál es el horario de visitas?"* puntúa 0.868 y
*"¿cuándo me quitan los puntos?"* puntúa 0.795. La ajena gana, porque todo el
corpus es texto médico postoperatorio y el coseno mide cercanía temática, no
respuesta. Ningún umbral separa esas dos, así que hay cuatro filtros encadenados
y **solo uno es un modelo**.

```mermaid
flowchart TB
    Q["Pregunta clínica del paciente<br/>build_query + procedimiento + día postop"]
    PROC{"¿el procedimiento que dice el paciente<br/>coincide con la ficha?"}
    ALL["rag/retrieve.py :: _allowed_document_ids<br/>solo documentos vivos en SQLite"]
    VEC["store.query · bge-m3<br/>rag_fetch_k, tope _MAX_POR_DOCUMENTO"]
    F1{"1 · similitud ≥ rag_min_confidence<br/>rag/retrieve.py"}
    F2{"2 · nombres propios presentes<br/>llm/ollama_adapter.py :: _nombres_propios_presentes"}
    F3{"3 · pertinencia<br/>llm.pregunta_es_del_dominio"}
    RED["compose_reply anclado a la evidencia"]
    F4{"4 · cifras dentro de la evidencia<br/>orchestrator._validar_grounding<br/>grounded_in_evidence"}
    OK["Respuesta + citas<br/>documento y página"]
    NO["ABSTENCION<br/>sin citas, y se lo pasa a enfermería"]

    Q --> PROC
    PROC -->|"no coincide"| NO
    PROC -->|"coincide"| ALL --> VEC --> F1
    F1 -->|"no"| NO
    F1 -->|"sí"| F2
    F2 -->|"falta un nombre propio"| NO
    F2 -->|"sí"| F3
    F3 -->|"no responde"| NO
    F3 -->|"sí"| RED --> F4
    F4 -->|"cifra inventada, o veredicto<br/>de normalidad con riesgo alto"| NO
    F4 -->|"sí"| OK

    classDef det fill:#dbeafe,stroke:#1e40af,color:#0b1220
    classDef llm fill:#fef3c7,stroke:#92400e,color:#0b1220
    classDef no fill:#fee2e2,stroke:#991b1b,color:#0b1220
    class ALL,VEC,F1,F2,F4,PROC det
    class F3,RED llm
    class NO no
```

El filtro del procedimiento existe porque el corpus está indexado **por
procedimiento**: si el paciente dice que lo operaron de otra cosa, recuperar con
la ficha discrepante solo puede citar documentos de la otra cirugía —una
afirmación falsa *con fuente*, que es peor que no responder—. Ni RAG ni citas: el
agente declara el límite y se crea una alerta para que alguien verifique el
registro (`_crear_alerta_procedimiento`).

La cadena pasó de **0/10 preguntas inventadas rechazadas a 9/10**, conservando
12/15 de las legítimas. El detalle está en
[`calibracion-rag.md`](calibracion-rag.md).

---

## Cómo comprobar que este diagrama es el código

Todo símbolo citado arriba existe en el repositorio. Para verificarlo de un
tirón, desde la raíz:

```bash
for s in process_turn_async "engine.py" evaluate next_action apply merge_symptoms \
         retrieve pregunta_es_del_dominio _nombres_propios_presentes \
         grounded_in_evidence _validar_grounding script_for close_conversation \
         build_summary MAX_SILENCIOS SIN_PROGRESO_CERRAR MAX_REPREGUNTAS \
         MAX_TURNOS_CONFIRMACION _allowed_document_ids ClinicalProcessor; do
  printf '%-34s %s\n' "$s" "$(grep -rl -- "$s" apps/backend/app --include='*.py' | head -1)"
done
```

Y la superficie HTTP de la tabla de §1, contra la app en marcha:

```bash
curl -s localhost:8000/openapi.json \
  | python -c "import json,sys; print(*sorted(json.load(sys.stdin)['paths']), sep='\n')"
```

---

**Documentos relacionados:** [`calibracion-triage.md`](calibracion-triage.md) ·
[`calibracion-rag.md`](calibracion-rag.md) · [`metricas.md`](metricas.md) ·
[`spikes-7-agosto.md`](spikes-7-agosto.md) · [`../README.md`](../README.md)

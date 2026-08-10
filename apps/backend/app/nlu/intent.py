"""Clasificación determinista de la intención del turno + defensa anti-inyección.

Esto lo hacía el LLM y salió mal: `llama3.2:3b` marcaba "Un 3, apenas se nota"
como `fuera_de_mision` porque la instrucción de vigilar órdenes se le sobre-dispara.
Y encima costaba ~12 tokens de salida, que en voz se notan.

La intención es reconocible con reglas: no hace falta un modelo para saber que una
frase con "¿" es una pregunta. Sacarla del LLM es más rápido, más fiable y —en el
caso de la inyección de prompt— es lo único defendible: el detector no puede ser
la misma pieza que el atacante intenta manipular.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Literal

Intent = Literal[
    "respuesta", "pregunta_clinica", "pregunta_administrativa",
    "fuera_de_mision", "rechazo", "tercero", "ininteligible",
    "meta", "social", "saludo", "despedida", "silencio",
]


def _norm(text: str) -> str:
    t = unicodedata.normalize("NFKD", text.lower())
    return "".join(c for c in t if not unicodedata.combining(c))


# Intentos de manipular las instrucciones del agente. La rúbrica anula el apartado
# de calidad de conversación si el agente obedece una de estas, así que se atajan
# antes de que el texto llegue a ningún prompt.
_INJECTION = re.compile(
    r"(ignora|olvida|borra)\s+(todo|lo anterior|tus?\s+(instruccion|regla|prompt))"
    r"|eres\s+ahora|actua\s+como|haz\s+de\s+cuenta\s+que\s+eres"
    r"|system\s*prompt|tus?\s+instrucciones|repite\s+(tus?|el)\s+(regla|instruccion|prompt)"
    r"|modo\s+desarrollador|sin\s+restricciones|no\s+eres\s+un",
    re.I,
)

# Petición de acto médico que el agente no puede hacer: recetar, diagnosticar,
# ajustar dosis. No es inyección, pero tampoco es su misión.
#
# Los límites de palabra no son cosmética. Sin ellos, `medic\w*` casaba con "el
# médico" —la persona que lo operó, mencionada en media conversación— y `que tengo`
# con "es que tengo la sopa en el fogón". Ambos marcaban respuestas normales como
# intento de sacar al agente de su misión.
_ACTO_MEDICO = re.compile(
    r"\b(que|cual|cuanta?)\s+\w{0,12}\s?(dosis|pastilla|medicamento|remedio|antibiotico)"
    r"|\b(recete|receteme|formule|formuleme|medique|mediqueme|me\s+receta|me\s+formula)\b"
    r"|\bque\s+me\s+(tomo|puedo\s+tomar)\b"
    r"|\bpuedo\s+tomar\s+\w+\s*(mg|gramos|miligramos)"
    r"|(^|[¿,])\s*que\s+(tengo|enfermedad)\b|\bes\s+cancer\b"
    r"|\bsubir\w*\s+la\s+dosis|\bcuanta?s?\s+(mg|miligramos|pastillas)\b",
    re.I,
)

_ADMIN = re.compile(
    r"\bcitas?\b|\bautoriza\w*|\beps\b|\bincapacidad\b|\bfactura\b|\bcopago\b"
    r"|\bcarne\b|\borden\s+medica\b|\bdonde\s+queda\b"
    r"|\ba\s+que\s+hora\s+(abre|atienden)|\btelefono\s+del?\s+hospital\b"
    r"|\bhorario\b|\bvisitas\b|\bparqueadero\b|\bafiliaci[oó]n\b",
    re.I,
)

# Los marcadores de pregunta se sacaron del análisis de los 1.687 turnos reales de
# paciente del dataset. Una versión más laxa marcaba el 42% de los turnos como
# pregunta clínica —"es normal después de la operación" y "cuando me muevo" son
# afirmaciones, no preguntas— y eso habría disparado el RAG en casi cada turno.

# Muletillas de confirmación. Comunísimas en el habla colombiana y no piden nada:
# "se ve rojita, ¿no?" es una respuesta, no una consulta.
_MULETILLA = re.compile(
    r"^\W*(no|ah|si|cierto|verdad|ya|ve|oyo|me entiende|sabe|cierto que si)\W*$", re.I)

# Meta-conversación sobre la llamada. Tiene respuesta determinista (repetir la
# pregunta, decir cuánto falta): no toca el RAG ni el LLM.
_META = re.compile(
    r"me\s+repite|no\s+(le\s+)?(entendi|escuche|oi)|(puede|podria)\s+repetir"
    r"|ya\s+(casi\s+)?terminamos|cuanto\s+(falta|mas)|cuantas\s+preguntas"
    r"|como\s+asi|que\s+dijo|mas\s+despacio|no\s+se\s+le\s+escucha",
    re.I,
)

# Cortesía y charla. Se despacha con una frase breve y se vuelve al guion.
# "¿usted qué cree?" a secas es charla; "¿usted cree que es normal sentirme así?"
# es una consulta clínica con forma de cortesía, y por eso la variante social
# exige que la pregunta termine ahí.
_SOCIAL = re.compile(
    r"\busted\s+como\s+(ha\s+)?(esta|estado)|\bcomo\s+esta\s+usted\b|y\s+usted\s+que\s+tal"
    r"|va\s+a\s+llover|\bel\s+clima\b|como\s+se\s+llama"
    r"|es\s+usted\s+(una\s+)?(robot|maquina|humano|persona)"
    r"|usted\s+que\s+(cree|piensa|opina)\s*\?*\s*$|de\s+donde\s+es\b",
    re.I,
)

# Pregunta clínica de verdad: pide una valoración o una instrucción de cuidado.
# `me preocup` lleva guarda de negación: "nada que me preocupe" es tranquilidad,
# no consulta, y sin la guarda marcaba como pregunta media capa limpia del dataset.
#
# Las dos alternativas de "cuándo puedo" existen porque Whisper transcribe sin
# signos de interrogación: "cuando podria volver a jugar futbol" llegaba sin `¿?`
# ni arranque de la lista cerrada, caía en `respuesta` y el paciente recibía la
# siguiente pregunta del guion en vez de una respuesta — medido en una llamada
# real donde lo preguntó tres veces seguidas. Exigen el verbo de posibilidad
# detrás ("cuando puedo/podré/podría...") a propósito: "cuando me muevo me duele"
# es una afirmación, y un "cuando" suelto marcaba media capa limpia del dataset.
# Verbos de actividad postoperatoria típicos de "¿puedo volver a...?": elegidos
# a propósito para NO solapar con el vocabulario de los seis slots (caminar,
# dormir, comer, moverse), que es justo lo que un "puedo" sin ancla podría
# confundir con la respuesta de movilidad/sueño/apetito en vez de una pregunta.
_ACTIVIDAD_POSTOP = (
    r"hacer\s+ejercicio|hacer\s+deporte|trotar|correr|nadar|manejar|conducir"
    r"|trabajar|viajar|volar|levantar\s+peso|cargar\s+peso"
    r"|tener\s+relaciones|hacer\s+fuerza|cocinar|bailar|jugar"
    r"|banarme|ducharme"
)

_PREGUNTA = re.compile(
    r"[¿?]"
    # "tengo que" lleva lookahead: "tengo que colgar/irme" es una despedida, no
    # una consulta, y desde que la pregunta tiene precedencia sobre la despedida
    # (ver `classify`) confundirlas mandaba la despedida al RAG.
    r"|^\s*(y\s+)?(puedo|debo"
    r"|tengo que (?!colgar|irme|salir|cortar|dejarl[oa]|terminar|contestar otra)"
    r"|sera|hasta cuando|cada cuanto|que hago"
    r"|que pasa si|en\s+cuantos?\s+(dias?|semanas?|meses?|tiempo)"
    r"|a\s+los\s+cuantos?\s+dias?|que\s+tan\s+pronto|como\s+hago)\b"
    r"|\b(cuando|hasta\s+cuando|en\s+cuantos?\s+(?:dias?|semanas?|meses?|tiempo)"
    r"|a\s+los\s+cuantos?\s+dias?|que\s+tan\s+pronto)"
    r"\s+(me\s+|se\s+)?(puedo|puede|podre|podria|voy\s+a\s+poder|vuelvo|volveria)\b"
    r"|\b(eso|esto|sera)\s+(es|esta)\s+(normal|bien|grave|peligroso)"
    r"|(?<!nada que )(?<!no me )\bme\s+(tengo\s+que\s+)?preocup"
    # Sin ancla al inicio del turno, a diferencia de la primera rama de arriba:
    # medido en una llamada real, "Un 4. Con ese dolor puedo volver a hacer
    # ejercicio." caía en `respuesta` porque el "puedo" no era la primera
    # palabra del turno completo (esa rama exige `^`) y no había "¿?" (Whisper
    # no los pone). El doble lookbehind de negación replica el de
    # `_AUTOMEDICACION`: "no puedo trabajar" es una respuesta, no una consulta.
    rf"|(?<!no\s)(?<!no\sme\s)\b(puedo|podre|podria|debo)\s+(volver\s+a\s+)?"
    rf"({_ACTIVIDAD_POSTOP})\b",
    re.I,
)

# Negarse a seguir. Antes incluía "adiós/chao/hasta luego", que no es lo mismo:
# despedirse al final de una llamada bien llevada es cooperar, no rechazar, y
# ahora es la señal de que se puede colgar (ver `despedida`).
_RECHAZO = re.compile(
    r"\bno\s+(quiero|voy a)\s+(hablar|contestar|seguir)"
    r"|dejeme\s+en\s+paz|no\s+me\s+moleste|cuelgue\s+ya|no\s+tengo\s+tiempo"
    r"|no\s+me\s+llame\s+mas",
    re.I,
)

# Negación simple de "¿algo más?": la respuesta natural a una pregunta de cierre.
# No es una despedida formal —"no, nada" no casa con `_DESPEDIDA` a menos que
# traiga "más"— pero en fase de CONFIRMACIÓN significa exactamente lo mismo:
# el paciente no tiene más temas y la llamada puede cerrar. Medido en una llamada
# real: "No, no, está muy bien. Muchas gracias." recibió otra pregunta de cierre,
# y la siguiente negación otra más, en bucle.
#
# Por tokens y no por regex: las negaciones se repiten y se combinan de formas
# que un patrón lineal no aguanta ("no no, así está bien, muchas gracias"). El
# turno cuenta si TODAS sus palabras son de negación/conformidad/cortesía y al
# menos una es de negación o conformidad — "muchas gracias" a secas no cierra
# nada, y cualquier palabra fuera de la lista ("no me duele") lo devuelve al
# flujo normal.
_TOKENS_CIERRE = frozenset({
    "no", "nada", "mas", "gracias", "muchas", "listo", "lista", "bueno", "buena",
    "bien", "esta", "estamos", "asi", "todo", "muy", "ok", "okey", "vale",
    "perfecto", "de", "acuerdo", "amable", "senor", "senora", "senorita",
    "doctor", "doctora", "eso", "es", "ya", "quedamos", "claro", "entendido",
})
_TOKENS_CONFORMIDAD = frozenset({
    "no", "nada", "listo", "lista", "bien", "perfecto", "vale", "ok", "okey",
    "claro", "entendido", "quedamos", "acuerdo",
})


def niega_mas_temas(text: str) -> bool:
    """¿El turno es una negación simple de "¿algo más?" ("no, nada, gracias")?

    Solo tiene sentido consultarlo en fase de CONFIRMACIÓN/ESCALAMIENTO, donde la
    última pregunta del agente fue de cierre: ahí un "no" pelado significa "no
    tengo más temas". En pleno tamizaje un "no" es la respuesta al síntoma y este
    detector no debe usarse.
    """
    palabras = re.findall(r"[a-zñ]+", _norm(text))
    if not palabras or len(palabras) > 10:
        return False
    if not all(p in _TOKENS_CIERRE for p in palabras):
        return False
    return bool(_TOKENS_CONFORMIDAD.intersection(palabras))


def contiene_despedida(text: str) -> bool:
    """¿El turno trae una fórmula de despedida, aunque traiga más cosas?

    A diferencia de `classify` —que ante despedida + pregunta clínica prioriza la
    pregunta—, esto reporta el hecho: el orquestador lo usa para responder la
    pregunta Y confirmar el cierre en el mismo turno, en vez de perder una de las
    dos cosas.
    """
    return bool(_DESPEDIDA.search(_norm(text)))


# Despedirse o dar por terminada la conversación. Es lo que el agente espera en
# fase de CONFIRMACIÓN para colgar: la llamada no acaba cuando el sistema termina
# su lista, acaba cuando el paciente entendió y se despide.
_DESPEDIDA = re.compile(
    r"\badios\b|\bchao\b|hasta\s+luego|nos\s+vemos|que\s+(este|le\s+vaya)\s+bien"
    r"|gracias,?\s+(hasta|adios|chao)|listo\s+entonces|ya\s+quedamos"
    r"|eso\s+es\s+todo|no\s+tengo\s+mas\s+(dudas|preguntas)"
    r"|(nada|ninguna)\s+mas\b|no,?\s+nada\s+mas|todo\s+claro|quedo\s+claro"
    r"|ya\s+entendi|entendido\s+gracias|puede\s+colgar",
    re.I,
)

# Saludo puro: el turno no dice nada más. "Buenas, el dolor va en un 3" NO entra
# aquí —el léxico tiene que sacar el 3—, por eso se ancla a toda la cadena.
# Medido: sin esto, "Hola, buenas." se clasificaba como `respuesta`, no aportaba
# dato y se comía uno de los dos reintentos del slot de dolor; el agente le
# contestaba a un saludo con la reformulación cerrada del dolor.
# Sin "si", "claro" ni "gracias" sueltos: un "sí" pelado es una RESPUESTA legítima
# a "¿ha tenido fiebre?", y clasificarlo como saludo perdería el dato. "Sí, dígame"
# sí entra, porque el "dígame" lo desambigua.
_SALUDO_PIEZA = (
    r"(?:hola+|buenas|buenos\s+dias|buenas\s+(?:tardes|noches)|alo+|si,?\s+digame"
    r"|a\s+la\s+orden|con\s+quien\s+hablo|quien\s+habla|digame|que\s+mas"
    r"|doctor\w*|doctora|senor\w*|senora|senorita|muy\s+amable)"
)
# Se encadenan: "Hola, buenas.", "Buenas, doctora", "Aló, sí dígame". Una sola
# pieza no basta.
_SALUDO = re.compile(
    rf"^\W*{_SALUDO_PIEZA}(?:[\s,]+{_SALUDO_PIEZA})*[\s.,!¡¿?]*$", re.I
)

_TERCERO = re.compile(
    r"\bsoy\s+(el|la)\s+(cuidador|hijo|hija|esposo|esposa|mama|papa|hermano|hermana|enfermer)"
    r"|le\s+paso\s+a|(el|ella)\s+no\s+puede\s+(hablar|contestar)"
    r"|(dice|cuenta)\s+que\s+(le|se)|lo\s+he\s+visto|la\s+he\s+visto",
    re.I,
)

# El paciente no dice NADA. Es distinto de `ininteligible` —donde algo llegó pero
# no se entendió— y por eso tiene intención propia: no se responde "¿me lo
# repite?" a quien no ha hablado, se comprueba si sigue en línea. El marcador
# `[silencio]` es el que ya usa `tests/dataset/dialogs.jsonl` (capa 2) y el que el
# pipeline de voz inyecta cuando vence el temporizador de inactividad.
_SILENCIO = re.compile(r"^\W*\[?\s*silencio\s*\]?\W*$", re.I)

# Silencios y audio degradado, tal como aparecen en la capa 2 del dataset.
_ININTELIGIBLE = re.compile(r"^\W*$|\[inaudible\]|^\.{2,}$|^(eh+|mm+|este\.{2,})\W*$", re.I)

# Vocales españolas (con tilde y diéresis). Ninguna palabra real del idioma
# carece de una en cada sílaba, así que su ausencia total en un token es la
# única señal fonotáctica sin falsos positivos conocidos.
_VOCALES = set("aeiouáéíóúü")
_CONSONANTES = set("bcdfghjklmnñpqrstvwxyz")


def _es_ruido_transcripcion(raw: str) -> bool:
    """Basura del STT en vez de silencio: "unufwef]", "asdkjhaskjdh", "xk29".

    `_ININTELIGIBLE` no los atrapaba porque no son vacíos ni el marcador
    `[inaudible]`: tienen letras, así que llegaban a `classify` como
    "respuesta" y de ahí al LLM, que —forzado por el esquema a elegir un
    valor del enum— alucinaba el más grave (así se coló
    `mobility=incapacitante_nueva` desde texto que nunca dijo eso).

    No hay una regla fonotáctica que separe TODO el ruido del español real:
    "ojoj" alterna vocal y consonante igual que "ojo", y "fewf"/"unufwef" no
    violan ningún límite razonable de racha de consonantes (comprobado a
    mano: "instrucciones", palabra real que ya vive en `_INJECTION`, tiene
    una racha de 4 con "nstr"; por eso el umbral de abajo es 5, no 4). Esos
    casos quedan para el LLM, con un ejemplo explícito en `_SYSTEM_EXTRACT`.

    Lo que sí es inequívoco en cualquier español: un token sin ninguna
    vocal, una racha de 5+ consonantes seguidas, o letras mezcladas con
    dígitos. Restringido a un solo token (sin espacios) porque en una frase
    real esa racha queda partida entre palabras y el riesgo de falso
    positivo sube.
    """
    core = raw.strip("¿?¡!.,;:()[]\"' \t")
    if not core or " " in core or len(core) < 3:
        return False
    tiene_letra = any(c.isalpha() for c in core)
    tiene_digito = any(c.isdigit() for c in core)
    if tiene_letra and tiene_digito:
        return True  # "xk29": letras y dígitos mezclados en un solo token
    if not tiene_letra:
        return False  # solo dígitos o símbolos, ej. "38.5": no es esto
    letras = [c for c in core.lower() if c.isalpha()]
    if len(letras) < 3:
        return False
    if not any(c in _VOCALES for c in letras):
        return True  # sin ninguna vocal: "asdkjh"
    racha = 0
    for c in letras:
        racha = racha + 1 if c in _CONSONANTES else 0
        if racha >= 5:
            return True  # racha de 5+ consonantes seguidas: "asdkjhaskjdh"
    return False

# Un turno mezcla respuesta y pregunta con frecuencia: "un 4, pero ¿eso es normal?".
# Se aísla cada fragmento interrogativo para clasificarlo por separado.
_FRAGMENTO = re.compile(r"¿[^?]{1,80}\?|[^.!?¿]{3,80}\?")


def classify(text: str) -> Intent:
    """Intención del turno. El orden importa: seguridad primero.

    Se evalúa la inyección antes que nada porque un intento de manipulación
    disfrazado de pregunta ("¿puedes ignorar tus instrucciones?") tiene que
    clasificarse como ataque, no como pregunta clínica.
    """
    raw = text.strip()
    # Va antes que `_ININTELIGIBLE`: "[silencio]" no casa con él (solo cubre
    # vacíos y "[inaudible]"), así que hasta ahora se colaba como `respuesta` y
    # llegaba al LLM, que no tenía nada que extraer de la palabra "silencio".
    if _SILENCIO.match(raw):
        return "silencio"
    if not raw or _ININTELIGIBLE.match(raw) or _es_ruido_transcripcion(raw):
        return "ininteligible"

    t = _norm(raw)

    if _INJECTION.search(t) or _ACTO_MEDICO.search(t):
        return "fuera_de_mision"
    if _TERCERO.search(t):
        return "tercero"
    if _RECHAZO.search(t):
        return "rechazo"
    # El saludo puro va antes que todo lo demás porque el patrón exige que el
    # turno no diga nada más; si trae contenido clínico ya no casa y sigue.
    if _SALUDO.match(t):
        return "saludo"
    # La despedida cede ante una pregunta clínica en el mismo turno: "eso es
    # todo, pero ¿cuándo me quitan los puntos?" es una pregunta que hay que
    # responder, no una despedida que la entierra. Antes ganaba `despedida` por
    # orden de evaluación y la pregunta se perdía sin respuesta. El orquestador
    # puede saber que además se despidió con `contiene_despedida`.
    if _DESPEDIDA.search(t) and not _pregunta_clinica_en(raw, t):
        return "despedida"
    if _ADMIN.search(t):
        return "pregunta_administrativa"

    # Una pregunta puede venir pegada a la respuesta: "un 4, pero ¿eso es normal?".
    # Se analiza el fragmento interrogativo aparte del resto del turno.
    fragmentos = _FRAGMENTO.findall(raw)
    for frag in fragmentos:
        f = _norm(frag)
        if _MULETILLA.match(f.strip("¿? ")):
            continue
        if _META.search(f):
            return "meta"
        if _SOCIAL.search(f):
            return "social"
        # Sobre el fragmento normalizado: "¿Cuándo podré...?" lleva tilde y los
        # patrones del regex van sin ella.
        if _PREGUNTA.search(f):
            return "pregunta_clinica"

    if not fragmentos and _PREGUNTA.search(t):
        return "pregunta_clinica"
    if _META.search(t):
        return "meta"
    return "respuesta"


def _pregunta_clinica_en(raw: str, t: str) -> bool:
    """¿El turno trae una pregunta clínica de verdad (no muletilla, no meta/social)?

    Réplica del análisis de fragmentos de `classify`, para decidir la precedencia
    despedida/pregunta sin duplicar la clasificación completa.
    """
    fragmentos = _FRAGMENTO.findall(raw)
    for frag in fragmentos:
        f = _norm(frag)
        if _MULETILLA.match(f.strip("¿? ")):
            continue
        if _META.search(f) or _SOCIAL.search(f):
            continue
        if _PREGUNTA.search(f):
            return True
    return bool(not fragmentos and _PREGUNTA.search(t))


# Declaración o pregunta indirecta de automedicación ("voy a tomar/me voy a
# tomar/pienso tomar[me]/me puedo tomar/puedo tomar X", "quisiera/quiero saber
# si..."), sin forma de pregunta directa: no matchea `_PREGUNTA` (sin "¿/?" ni
# verbo de posibilidad al inicio de la frase). Solo se consulta en la ventana
# posterior a un guion crítico (ver `orchestrator.py`), donde una mención así
# merece verificarse contra el RAG antes de despedirse. Medido en una llamada
# real: "me puedo tomar" y "si quisiera saber si..." son la forma MÁS común de
# preguntar algo así en voz, más común que "voy a tomar", y la primera versión
# de este detector no las cubría — el RAG nunca se consultaba.
#
# Doble lookbehind para la negación ("no puedo tomar nada" / "no me puedo
# tomar nada"): uno solo no bastaba, porque "me" se cuela entre "no" y
# "puedo" y el lookbehind de tres caracteres ya no ve el "no".
_AUTOMEDICACION = re.compile(
    r"(?<!no\s)(?<!no\sme\s)"
    r"\b(voy\s+a|me\s+voy\s+a|pienso|quiero|puedo|me\s+puedo)\s+tomar(me)?\b"
    r"|\bme\s+tom(o|ar[ée]|ar[íi]a)\b"
    r"|\bquisiera\s+saber\b|\bquiero\s+saber\b|\bpuedo\s+saber\b",
    re.I,
)


def menciona_automedicacion(text: str) -> bool:
    """¿El paciente anuncia que va a tomarse algo por su cuenta, o pregunta
    indirectamente si puede/debe hacerlo?

    No es parte de `classify()` a propósito: `_PREGUNTA` está calibrada contra
    los 1.687 turnos reales del dataset (ver comentario más arriba) y ampliarla
    arriesga esa calibración. Este detector vive aparte y solo lo consulta el
    orquestador en la ventana posterior a un guion crítico.
    """
    return bool(_AUTOMEDICACION.search(_norm(text)))


def is_injection(text: str) -> bool:
    """Solo el intento de manipular instrucciones, sin la petición de acto médico.

    Se separa porque se registran distinto: la inyección va a la traza del turno
    como `injection_attempt` y es una conducta que la rúbrica vigila; pedir una
    dosis es simplemente algo que el agente no hace.
    """
    return bool(_INJECTION.search(_norm(text)))

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
    "meta", "social",
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
_PREGUNTA = re.compile(
    r"[¿?]"
    r"|^\s*(puedo|debo|tengo que|sera|hasta cuando|cada cuanto|que hago|que pasa si)\b"
    r"|\b(eso|esto|sera)\s+(es|esta)\s+(normal|bien|grave|peligroso)"
    r"|(?<!nada que )(?<!no me )\bme\s+(tengo\s+que\s+)?preocup",
    re.I,
)

_RECHAZO = re.compile(
    r"\bno\s+(quiero|voy a)\s+(hablar|contestar|seguir)"
    r"|dejeme\s+en\s+paz|no\s+me\s+moleste|cuelgue|adios|chao|hasta\s+luego"
    r"|no\s+tengo\s+tiempo",
    re.I,
)

_TERCERO = re.compile(
    r"\bsoy\s+(el|la)\s+(cuidador|hijo|hija|esposo|esposa|mama|papa|hermano|hermana|enfermer)"
    r"|le\s+paso\s+a|(el|ella)\s+no\s+puede\s+(hablar|contestar)"
    r"|(dice|cuenta)\s+que\s+(le|se)|lo\s+he\s+visto|la\s+he\s+visto",
    re.I,
)

# Silencios y audio degradado, tal como aparecen en la capa 2 del dataset.
_ININTELIGIBLE = re.compile(r"^\W*$|\[inaudible\]|^\.{2,}$|^(eh+|mm+|este\.{2,})\W*$", re.I)

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
    if not raw or _ININTELIGIBLE.match(raw):
        return "ininteligible"

    t = _norm(raw)

    if _INJECTION.search(t) or _ACTO_MEDICO.search(t):
        return "fuera_de_mision"
    if _TERCERO.search(t):
        return "tercero"
    if _RECHAZO.search(t):
        return "rechazo"
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
        if _PREGUNTA.search(frag):
            return "pregunta_clinica"

    if not fragmentos and _PREGUNTA.search(raw):
        return "pregunta_clinica"
    if _META.search(t):
        return "meta"
    return "respuesta"


def is_injection(text: str) -> bool:
    """Solo el intento de manipular instrucciones, sin la petición de acto médico.

    Se separa porque se registran distinto: la inyección va a la traza del turno
    como `injection_attempt` y es una conducta que la rúbrica vigila; pedir una
    dosis es simplemente algo que el agente no hace.
    """
    return bool(_INJECTION.search(_norm(text)))

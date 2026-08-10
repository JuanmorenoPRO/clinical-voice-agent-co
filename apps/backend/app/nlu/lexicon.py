"""Léxico determinista de español colombiano: la ruta principal de extracción.

No es un respaldo del LLM, es lo que va primero. Tres razones, todas medidas:

  1. **Seguridad.** Una bandera de emergencia no puede depender de un modelo de 3B.
     Al meter la bandera en el esquema del LLM, éste marcó `no_puede_respirar` en
     "como un 7, la pastilla no me lo quita" — un falso positivo de emergencia — y
     además dejó de extraer el dolor. Aquí las banderas son regex sobre frases
     formulaicas, que es justo lo que la gente dice cuando algo va mal en serio.
  2. **Latencia.** Lo que resuelve el léxico cuesta 0 ms. El LLM cuesta ~325 ms.
  3. **Fiabilidad.** "Un 3" es un dígito: no hace falta una red neuronal.

El LLM se queda con lo único que solo él sabe hacer: mapear la paráfrasis rara que
no está en ninguna lista ("no me provoca nada", "ando maluco desde antier").
"""

from __future__ import annotations

import re
import unicodedata

from ..schemas import Symptoms
from . import otros_sintomas, polaridad


# Letras repetidas de más: "borrroso", "muuucho", "siiii". Salen del STT y de
# quien escribe alargando, y sin esto el patrón no casa y el síntoma se pierde
# entero (caso real: "estoy viendo borrroso").
#
# Se colapsan distinto según el tipo, y la asimetría importa:
#   - Vocales 3+ -> UNA. Ninguna palabra del español lleva tres vocales idénticas
#     seguidas, así que es seguro, y "muuucho" tiene que llegar a "mucho" para
#     que `_DOLOR_DESCRIPTOR` lo reconozca.
#   - Consonantes 3+ -> DOS, porque "rr", "ll", "cc" y "nn" sí son legítimas.
# Las dobles reales ("leer", "cooperar", "perro") no se tocan: hacen falta tres.
_VOCALES_REPETIDAS = re.compile(r"([aeiou])\1{2,}")
_CONSONANTES_REPETIDAS = re.compile(r"([^aeiou\W\d_])\1{2,}")


def normalize(text: str) -> str:
    t = unicodedata.normalize("NFKD", text.lower())
    t = "".join(c for c in t if not unicodedata.combining(c))
    t = _VOCALES_REPETIDAS.sub(r"\1", t)
    return _CONSONANTES_REPETIDAS.sub(r"\1\1", t)


# Negación que anula un hallazgo si va justo delante: "no está roja", "sin
# secreción", "nada de eso". Hace falta desde que los patrones categóricos
# reconocen el adjetivo pelado ("muy roja"): sin la guarda, "no está roja" se
# anotaría como eritema. Mismo mecanismo y misma redacción que
# `otros_sintomas._NEGACION`, duplicado a propósito porque este módulo ya importa
# a aquel y la vuelta cerraría el ciclo.
_NEGACION_PREVIA = re.compile(r"\b(no|sin|ni|nada|nunca|jamas|tampoco)\b[^.;,]{0,28}$")


def _afirmado(patron: re.Pattern[str], texto: str) -> bool:
    """¿`patron` aparece en `texto` sin una negación justo antes?

    Se recorren TODAS las apariciones: "no está roja, pero la otra sí está roja"
    afirma el hallazgo en la segunda. Ante varias, basta una sin negar — la
    asimetría de siempre, un falso negativo cuesta más que un falso positivo.
    """
    return any(not _NEGACION_PREVIA.search(texto[:m.start()])
               for m in patron.finditer(texto))


# --- banderas de emergencia ---------------------------------------------------
# Son formulaicas a propósito: se prefiere un falso positivo (una llamada de más)
# a un falso negativo, que es la falla catastrófica según la rúbrica.
_BANDERAS: list[tuple[str, re.Pattern[str]]] = [
    (
        "heavy_bleeding",
        re.compile(
            r"botando\s+(mucha\s+)?sangre"
            r"|sangr(ando|a)\s+(mucho|muchisimo|un\s+resto|bastante|abundante|harto)"
            # "no para" solo cuenta cerca de una mención de sangre, para no
            # capturar "no para de llorar"/"no para el dolor" fuera de contexto.
            r"|no\s+para\s+de\s+sangrar|sangr\w*.{0,25}no\s+para\b"
            r"|chorro\s+de\s+sangre|empapad\w+\s+de\s+sangre"
            r"|hemorragia"
        ),
    ),
    (
        "breathing_difficulty",
        re.compile(
            r"no\s+puedo\s+respirar|me\s+falta\s+(el\s+)?(aire|respiracion)"
            r"|me\s+ahogo|ahogad\w+|dificultad\s+para\s+respirar|no\s+me\s+entra\s+(el\s+)?aire"
            r"|siento\s+que\s+me\s+asfixio"
            # La forma MÁS común en el habla real, y la que faltaba: medido, "está
            # bien, pero me cuesta respirar" no encendía la bandera y el turno se
            # resolvía con `wound=normal` y la siguiente pregunta del guion. Un
            # falso negativo de emergencia es la falla catastrófica de este
            # sistema, así que aquí se cubre el verbo completo ("cuesta/dificulta/
            # ahoga") y no solo la negación absoluta ("no puedo").
            r"|me\s+(cuesta|dificulta|duele)\s+(mucho\s+)?respirar"
            r"|me\s+cuesta\s+(coger|tomar|agarrar)\s+(el\s+)?aire"
            r"|respiro\s+con\s+dificultad|me\s+quedo\s+sin\s+aire|sin\s+aliento"
            r"|me\s+agito\b|me\s+fatigo\b|me\s+sofoco\s+al\s+(caminar|moverme)"
        ),
    ),
    (
        "chest_pain",
        re.compile(
            r"dolor\s+(en\s+)?el\s+pecho|me\s+duele\s+el\s+pecho|opresion\s+en\s+el\s+pecho"
            r"|puntada\s+en\s+el\s+pecho|siento\s+un\s+peso\s+en\s+el\s+pecho"
        ),
    ),
    (
        "loss_of_consciousness",
        re.compile(
            r"me\s+desmay\w+|perdi\s+el\s+(conocimiento|sentido)|me\s+desvaneci"
            r"|me\s+cai\s+redond\w+|quede\s+inconsciente"
        ),
    ),
    (
        "seizure",
        re.compile(
            r"convulsion\w*|convulsion\w*|me\s+dio\s+un\s+ataque|se\s+puso\s+a\s+temblar\s+todo"
            r"|epilep\w+"
        ),
    ),
    (
        "altered_mental_status",
        re.compile(
            r"estoy\s+(muy\s+)?confundid\w+|no\s+se\s+(donde|quien)\s+"
            r"|desorientad\w+|no\s+reconoce|habla\s+incoherenc|delira\w*"
        ),
    ),
]

# --- dolor: dígito explícito --------------------------------------------------
# "como un 7", "un 3 de 10", "7/10". Se exige contexto para no capturar "37" de
# una temperatura ni "3 días".
_DOLOR_NUM = re.compile(
    r"(?:^|\b)(?:un|como\s+un|de|en|seria\s+un|diria\s+que\s+un)?\s*"
    r"(10|[0-9])\s*(?:/\s*10|\s+de\s+10|\b)(?!\s*(?:dias?|horas?|grados|semanas?|veces))",
)

# Números en <=  palabra: "nueve", "es un ocho", "diez de diez". El paciente mayor
# da la escala en letras, no en cifras ("Eh... nueve."), y sin esto el slot
# quedaba sin resolver y el guion quemaba repreguntas pidiendo lo ya contestado.
# Se excluyen "uno"/"una": en el habla colombiana "uno" es sobre todo el
# pronombre impersonal ("uno come bien"), y eso sería un falso positivo de dolor
# 1 en casi cada turno.
_NUMEROS: dict[str, int] = {
    "cero": 0,
    "dos": 2,
    "tres": 3,
    "cuatro": 4,
    "cinco": 5,
    "seis": 6,
    "siete": 7,
    "ocho": 8,
    "nueve": 9,
    "diez": 10,
}

_DOLOR_NUM_PALABRA = re.compile(
    r"(?:^|\b)(?:un|como\s+un|de|en|seria\s+un|diria\s+que\s+un)?\s*"
    r"(cero|dos|tres|cuatro|cinco|seis|siete|ocho|nueve|diez)"
    r"\s*(?:/\s*10|\s+de\s+10|\b)"
    r"(?!\s*(?:dias?|horas?|grados|semanas?|veces|de\s+la\s+(?:tarde|noche|manaana|madrugada)))",
)

_DOLOR_DESCRIPTOR: list[tuple[int, re.Pattern[str]]] = [
    # Redondeo conservador hacia arriba: "fuerte" cae en 8, que cruza el umbral
    # rojo. Es deliberado — ver docs/calibracion-triage.md.
    (
        9,
        re.compile(
            r"un\s+berraco|no\s+aguanto|insoportable|lo\s+peor|horrible"
            r"|me\s+estoy\s+muriendo|espantoso|una\s+chimba\s+de\s+dolor"
        ),
    ),
    (
        8,
        re.compile(
            r"me\s+duele\s+(mucho|muchisimo|bastante)|mucho\s+mucho|muy\s+fuerte"
            r"|fuertisimo|durisimo|tenaz|maluquisimo"
            r"|no\s+me\s+deja\s+(dormir|moverme)"
        ),
    ),
    (5, re.compile(r"\bmoderad\w+|mas\s+o\s+menos|ahi\s+va|soportable|regular")),
    (
        2,
        re.compile(
            r"un\s+dolorcito|leve|poquit\w+|casi\s+no\s+(se\s+nota|duele)"
            r"|molestia\s+no\s+mas|apenas\s+se\s+nota"
        ),
    ),
    (
        0,
        re.compile(
            r"no\s+me\s+duele\s+nada|ningun\s+dolor|nada\s+de\s+dolor"
            r"|no\s+tengo\s+dolor"
        ),
    ),
]

# --- temperatura --------------------------------------------------------------
_TEMP = re.compile(r"\b(3[5-9]|4[0-3])\s*(?:[.,]\s*(\d))?\s*(?:grados|°|c\b)?")
# "37 y algo", "38 y medio"
_TEMP_FRAC = re.compile(r"\b(3[5-9]|4[0-2])\s+y\s+(medio|algo|pico|cachito)")

# Este es el slot con más respuestas evasivas del dataset, y tiene explicación: el
# propio README del reto avisa de que el paciente "a veces ni un termómetro" tiene.
# "No me he tomado la temperatura" es una respuesta legítima, no un fallo, y por eso
# el guion la reformula en cerrada ("¿lo ha sentido como fiebre, sí o no?").
# La sensación térmica sin medir SÍ se recoge: es una señal, aunque sea blanda, y
# el criterio de todo el sistema es que la ausencia de dato nunca baje el riesgo.
_FIEBRE_SI = re.compile(
    r"\bfiebre|calentura|destemplad\w+|escalofrio|con\s+frio\s+y\s+calor"
    r"|ardiendo|hirviendo|temperatura\s+alta"
    # "me siento caliente" no casaba: `siento.*calor` no cubre "caliente" y la
    # alternativa de abajo exige "ando/estoy". Se unifican en un solo patrón de
    # verbo de sensación + raíz térmica.
    r"|(siento|sentido|senti|ando|estoy|estaba|estado|amaneci)\s+.{0,14}(calor|calient|tibi|arde)"
    # "acalorada" es la forma colombiana corriente de referir febrícula, y no
    # llevaba verbo reconocido delante ("he estado como acalorada un poco"), así
    # que solo la salvaba el LLM — y solo si le tocaba el slot de fiebre. Costaba
    # un rojo entero: en caso_tray_pac_42_00030_7 la fiebre de 38.9 nunca se mide,
    # y sin `fever=True` la regla de fiebre referida no puede dispararse.
    r"|acalorad\w+|sofocad\w+|con\s+bochorno"
)
# Va ANTES que `_FIEBRE_SI` en `extract`, así que todo lo que se cubra aquí queda
# a salvo del `\bfiebre` sin anclar de arriba. Esa es la razón de ampliarlo por
# aquí y no de estrechar el patrón afirmativo: la asimetría del módulo dice que un
# falso negativo de fiebre es la falla catastrófica y un falso positivo solo es
# ruido, así que se toca el lado seguro.
# Medido: "No fiebre, pero sí estoy temblando" daba `fever=True` —el paciente la
# NIEGA— y el agente le contestaba "Con calentura, entonces." (`phrasing._REFLEJO`).
# Ninguna de las formas escuetas de negar estaba cubierta: todas exigían un verbo.
_FIEBRE_NO = re.compile(
    r"no\s+(he\s+tenido|tengo|ha\s+tenido)\s+(fiebre|calentura|temperatura)"
    r"|sin\s+fiebre|nada\s+de\s+fiebre|de\s+eso\s+(nada|nada\s+que\s+ver)"
    r"|fiebre\s+no\s+(he\s+tenido|ha\s+habido)|ninguna\s+fiebre"
    # Negación escueta, que es como de verdad contesta el paciente a una pregunta
    # cerrada: "no fiebre", "fiebre no", "no, calentura no", "que va, fiebre no".
    r"|\bno,?\s+(fiebre|calentura)\b|\b(fiebre|calentura),?\s+no\b"
    r"|\bnegativo\s+(a\s+)?(fiebre|calentura)\b"
)

# "Sí me la tomé" contesta al ACTO de medir, no al HALLAZGO. Medido con
# llama3.2:3b: "si la he tomado" y "si" a secas devolvían `fever=True`, el slot
# quedaba resuelto y la cifra —la que dispara `fiebre_38`— no se pedía nunca.
# Cuando esto casa y el turno no trae número, no se toca `fever`: se marca que
# falta la cifra y el guion la persigue con un SEGUIMIENTO.
_TOMO_TEMPERATURA = re.compile(
    r"\b(me\s+la\s+|la\s+he\s+|me\s+la\s+he\s+|si\s+la\s+|ya\s+me\s+la\s+)?"
    r"(tome|tomado|tomo|medi|medido|mido|puse\s+el\s+termometro)\b"
    r"|con\s+el\s+termometro|si\s+tengo\s+termometro"
)
# Y su contrario: no hay con qué medir. Es respuesta legítima y cierra el
# seguimiento sin insistir.
_SIN_TERMOMETRO = re.compile(
    r"no\s+(tengo|tenemos|hay|consegui|tuve)\s+.{0,12}termometro"
    r"|sin\s+termometro|no\s+me\s+la\s+(he\s+)?(tomado|medido|tome)"
    r"|no\s+(he\s+podido|puedo)\s+.{0,14}(tomar|medir)"
    r"|termometro\s+no\s+(tengo|hay)"
)

# Sí/no pelados, que es como de verdad se contesta una pregunta cerrada. Medido
# en una llamada real, tres veces la misma pregunta a quien ya había contestado:
#
#   Agente:   ¿Ha tenido fiebre o calentura estos días?
#   Paciente: No, yo me he sentido bien.        -> el slot no se resolvía
#   Agente:   ¿Se ha sentido caliente o con escalofríos...?     (repregunta 1)
#   Paciente: No, yo me he sentido muy bien.    -> seguía sin resolverse
#   Agente:   Sin termómetro: ¿lo ha sentido como fiebre, sí o no?  (repregunta 2)
#
# Estos patrones solo detectan la POLARIDAD de la respuesta; qué significa la
# decide `nlu/polaridad.py` a partir de la pregunta que se hizo, porque el mismo
# "no" vale cosas opuestas según se preguntara "¿ha tenido dificultad?" o "¿ha
# podido caminar sin problema?".
#
# Anclados al principio del turno: es donde va la respuesta a una pregunta
# cerrada.
_POLAR_NO = re.compile(
    r"^\W*(no|nada|ning[uo]n\w*|nunca|jamas|negativo|para\s+nada"
    r"|que\s+va|que\s+esperanzas|en\s+absoluto|nombre)\b"
)
_POLAR_SI = re.compile(
    r"^\W*(si+|sip|claro|obvio|correcto|exacto|efectivamente|asi\s+es"
    r"|eso\s+si|pues\s+si|un\s+poc\w+|un\s+poquit\w+|algo|a\s+veces"
    r"|de\s+vez\s+en\s+cuando)\b"
)

# El paciente afirma, en general, que está bien. No es el valor de ningún slot
# —para eso está `_NORMAL_GENERICO`— sino el árbitro de un turno contradictorio.
# Medido en la llamada reportada:
#
#   Agente:   ¿Ha podido levantarse y caminar sin problema estos días?
#   Paciente: No, yo estoy bien.
#
# El "no" apunta a limitación y el "estoy bien" a normalidad, porque ese "no" no
# niega la pregunta: es el "no pasa nada" del habla corriente. Cuando los dos
# chocan no se resuelve el slot y se repregunta, que sale más barato que anotar
# el dato al revés.
#
# El lookbehind distingue el caso que sí es una negación de verdad: en "no estoy
# bien" el "no" va pegado al verbo; en "no, yo estoy bien" no.
_BIEN_GENERICO = re.compile(
    r"(?<!no\s)\b(estoy|ando|sigo|voy|me\s+siento|me\s+he\s+sentido|he\s+estado)"
    r"\s+(muy\s+|super\s+|bastante\s+)?(bien|normal|estable|tranquil\w+)"
)

# Valores que significan "aquí no pasa nada". Un turno contradictorio puede
# resolverse hacia uno de estos —no añade riesgo— pero nunca hacia un hallazgo.
_BENIGNO: frozenset = frozenset({"normal", False})

# --- herida -------------------------------------------------------------------
_HERIDA: list[tuple[str, re.Pattern[str]]] = [
    (
        "secrecion_purulenta",
        re.compile(
            r"\bpus\b|materia|supura\w*"
            r"|bota\w*\s+.{0,18}(liquido|liquidito|amarill|verdos|secrecion)"
            # El cuantificador intermedio es lo que fallaba: "le sale un poquito
            # de líquido, como amarillito" es la forma real en que el paciente
            # minimizador cuenta una secreción purulenta, y era un rojo perdido
            # (caso_tray_pac_42_00028_7 del dataset).
            r"|sale\s+.{0,18}(liquido|liquidito|amarill|verdos|secrecion|pus)"
            r"|sale\s+(como\s+)?(un\s+)?algo|huele\s+(feo|mal|maluco)"
            r"|secrecion\s+(amarilla|verde|purulenta)"
        ),
    ),
    (
        "eritema_leve",
        re.compile(
            r"rojit\w+|roja?\s+(en\s+)?(el\s+)?(borde|los\s+bordes)|enrojecid\w+|colorad\w+"
            r"|un\s+poco\s+(roja|inflamada|hinchada)|eritema"
            # El adjetivo pelado, que es como el paciente lo dijo dos veces
            # seguidas en la llamada reportada ("la área está muy roja") y no
            # casaba con nada: hasta ahora hacía falta el diminutivo o el borde.
            # Una herida roja perdida entera es el falso negativo que este módulo
            # existe para evitar. La guarda de `_afirmado` es la que impide que
            # "no está roja" entre por aquí.
            r"|(esta|estaba|se\s+ve|se\s+ha\s+visto|la\s+veo|la\s+tengo|tengo)"
            r"\s+\w{0,10}\s?roj\w+"
            r"|\b(muy|mas|bien|algo|bastante|toda|media)\s+roj\w+"
            # "rosadita" es como la paciente colombiana describe un eritema leve, y
            # faltaba. Costaba un rojo entero: en caso_tray_pac_42_00030_7 la fiebre
            # de 38.9 nunca se mide ("he estado como acalorada"), así que el rojo
            # depende de `_fever_reported_unmeasured`, que exige DOS señales
            # amarillas. Sin el eritema solo había una y el caso salía verde.
            r"|rosad\w+|rosita|irritad\w+|un\s+poco\s+(rosada|morada)|morad\w+"
        ),
    ),
    (
        "normal",
        re.compile(
            r"(la\s+herida\s+)?(esta|se\s+ve|(yo\s+)?la\s+veo)\s+(muy\s+|super\s+|bastante\s+)?"
            r"(bien|normal|sana|cerrada|seca)"
            r"|no\s+tengo\s+nada\s+en\s+la\s+herida|cicatrizando\s+bien"
        ),
    ),
]

# --- movilidad ----------------------------------------------------------------
_MOVILIDAD: list[tuple[str, re.Pattern[str]]] = [
    (
        "incapacitante_nueva",
        re.compile(
            r"no\s+me\s+puedo\s+(ni\s+)?(parar|levantar|mover)|no\s+puedo\s+caminar"
            r"|no\s+me\s+levanto|no\s+aguanto\s+(parad\w+|de\s+pie)|postrad\w+"
            r"|no\s+me\s+responde\s+la\s+pierna"
        ),
    ),
    (
        "limitada_esperada",
        re.compile(
            r"(me\s+)?cuesta\s+(un\s+poco\s+)?(caminar|moverme|levantarme)|despacito"
            r"|con\s+ayuda|me\s+demoro|poco\s+a\s+poco|con\s+dificultad|lentito"
        ),
    ),
    (
        "normal",
        re.compile(
            r"camino\s+(bien|normal)|me\s+muevo\s+(bien|normal)|sin\s+problema\s+para"
            r"|hago\s+(todo|mis\s+cosas)|normal\s+para\s+(caminar|moverme)"
        ),
    ),
]

# --- apetito ------------------------------------------------------------------
# Los patrones se ampliaron midiendo contra los 160 turnos reales de cada slot:
# la primera versión solo cubría presente ("como bien") y se perdía la mitad,
# porque la gente responde en pretérito perfecto ("he comido bien") o en gerundio
# ("comiendo bien"). Ese detalle gramatical valía 30 puntos de cobertura.
_APETITO: list[tuple[str, re.Pattern[str]]] = [
    (
        "muy_disminuido",
        re.compile(
            r"no\s+me\s+provoca\s+(nada|casi\s+nada)"
            r"|no\s+(he\s+)?com(o|ido|iendo)\s+(casi\s+)?nada"
            r"|casi\s+no\s+(he\s+)?com(o|ido)"
            # "no he podido comer" (con "podido") no matcheaba: el patrón exigía
            # "comido/como" pegado a "no he", sin espacio para el verbo auxiliar.
            r"|casi\s+no\s+he\s+podido\s+comer|no\s+(he\s+)?podido\s+comer\s+casi\s+nada"
            r"|se\s+me\s+quitaron\s+las\s+ganas(\s+de\s+comer)?"
            r"|no\s+me\s+pasa\s+(la\s+)?comida|sin\s+ganas\s+de\s+comer"
            r"|se\s+me\s+cerro\s+el\s+estomago|no\s+tengo\s+(nada\s+de\s+)?(hambre|apetito)"
            r"|todo\s+me\s+sabe\s+mal|perdi\s+(el\s+)?apetito"
        ),
    ),
    (
        "levemente_disminuido",
        re.compile(
            r"com(o|ido|iendo)\s+(menos|poquito|poco)|(un\s+)?poco\s+desganad\w+"
            r"|no\s+tanto\s+como\s+antes|se\s+me\s+bajo\s+(un\s+poco\s+)?el\s+(hambre|apetito)"
            r"|a\s+ratos\s+me\s+provoca|apetito\s+.{0,14}(bajo|bajito|flojo|regular)"
            r"|no\s+me\s+provoca\s+much|como\s+por\s+obligacion|con\s+desgano"
        ),
    ),
    (
        "normal",
        re.compile(
            r"(he\s+)?com(o|ido|iendo)\s+(bien|normal|de\s+todo|como\s+siempre)"
            r"|buen\s+apetito|con\s+(mucha\s+)?hambre"
            r"|(el\s+)?apetito\s+.{0,18}(bien|normal|igual|de\s+siempre)"
            r"|sin\s+problema\s+para\s+comer|no\s+me\s+ha\s+faltado\s+el\s+apetito"
            r"|normal\s+para\s+comer"
        ),
    ),
]

# --- sueño --------------------------------------------------------------------
_SUENO: list[tuple[str, re.Pattern[str]]] = [
    (
        "muy_alterado",
        re.compile(
            r"no\s+(he\s+)?p(u|o)(dido|de)\s+(pegar\s+el\s+ojo|dormir)"
            # "no pegué el ojo" es la forma más común del modismo, sin "pude/pudo"
            # delante — mismo bug que el tiempo verbal de apetito/sueño de
            # docs/spikes-7-agosto.md ("como bien" vs. "he comido bien").
            r"|(casi\s+)?no\s+pegu[eé]\s+el\s+ojo"
            r"|no\s+(he\s+)?d(uermo|ormido)\s+(casi\s+)?nada|casi\s+no\s+(he\s+)?d(uermo|ormido)"
            r"|paso\s+la\s+noche\s+en\s+vela|toda\s+la\s+noche\s+despiert\w+"
            r"|dando\s+vueltas\s+toda\s+la\s+noche|no\s+concilio\s+el\s+sueno"
            r"|(he\s+)?dormido\s+(muy\s+)?mal|pesimo\s+.{0,10}dorm"
            # "no he dormido bien" se anotaba como `normal`: el patrón de abajo
            # casa "dormido bien" y nadie miraba el "no" de delante. Ahora la
            # guarda de `_afirmado` lo suprime allí, y aquí se recoge por lo que
            # es. El lookahead cede "...bien que digamos" al nivel leve, que es
            # más específico y se evalúa después.
            r"|no\s+(he\s+)?(dormido|duermo)\s+(muy\s+)?bien(?!\s+que\s+digamos)"
        ),
    ),
    (
        "levemente_alterado",
        re.compile(
            r"me\s+despierto\s+(varias\s+veces|a\s+ratos|por)|d(uermo|ormido)\s+a\s+ratos"
            r"|a\s+pedazos|me\s+cuesta\s+(un\s+poco\s+)?(coger|conciliar|agarrar)"
            r"|interrumpid\w+|no\s+(he\s+)?dormido\s+(muy\s+)?bien\s+que\s+digamos"
            r"|me\s+desvelo|con\s+(algo\s+de\s+)?incomodidad\s+al\s+acostar"
        ),
    ),
    (
        "normal",
        re.compile(
            r"(he\s+)?d(uermo|ormido)\s+(bien|normal|de\s+corrido|tranquil\w+)"
            r"|descanso\s+bien|sin\s+problema\s+para\s+dormir"
            r"|(el\s+)?sueno\s+.{0,18}(bien|normal|igual)|he\s+descansado"
        ),
    ),
]

_CATEGORICOS: dict[str, tuple[str, list[tuple[str, re.Pattern[str]]]]] = {
    "herida": ("wound", _HERIDA),
    "movilidad": ("mobility", _MOVILIDAD),
    "apetito": ("appetite", _APETITO),
    "sueno": ("sleep", _SUENO),
}

# --- respuestas elípticas: solo significan algo tras SU pregunta --------------
# El agente hace una pregunta cerrada y el paciente contesta con las palabras que
# la propia pregunta le ofrece. Barrido de las seis repreguntas del guion contra
# las respuestas que invitan: 24 de 30 no se entendían. Es el "no me está
# escuchando" en su forma más literal — el léxico no sabe leer las respuestas a
# sus propias preguntas:
#
#   Agente:   ¿Camina como antes de la cirugía, o le cuesta más?
#   Paciente: Me cuesta un poco más.        -> nada, y el slot se dio por perdido
#   Agente:   ¿La herida está seca, o ha manchado el apósito con algo?
#   Paciente: Seca.                         -> nada
#   Agente:   ¿Durmió bien anoche o mal?
#   Paciente: Mal.                          -> nada
#
# Van APARTE de `_CATEGORICOS` y solo se aplican al slot que se preguntó, igual
# que `_NORMAL_GENERICO`. La regla para decidir dónde va cada patrón: *¿dicha
# mientras se pregunta otra cosa, la frase sigue significando lo mismo?* "La
# herida está roja" sí, y por eso vive arriba; "me cuesta más" no —puede ser
# comer— y "hinchada" tampoco —puede ser la pierna—, así que viven aquí.
_CORTAS: dict[str, list[tuple[str, re.Pattern[str]]]] = {
    "movilidad": [
        (
            "incapacitante_nueva",
            re.compile(r"^\W*(no\s+puedo|nada|imposible)\b|no\s+me\s+aguanto"),
        ),
        (
            "limitada_esperada",
            re.compile(
                # "me cuesta" sin verbo detrás: el patrón de `_MOVILIDAD` exige
                # "caminar|moverme|levantarme" pegado, y la repregunta del guion
                # invita justo a la forma corta.
                r"(me\s+)?cuesta(\s+(un\s+poco|mas|bastante|harto|mucho))*\s*$"
                r"|(me\s+)?cuesta\s+(un\s+poco\s+)?mas"
                r"|necesito\s+(que\s+me\s+)?ayud|me\s+(ayudan|toca\s+con\s+ayuda)"
                r"|mas\s+(despacio|lento|lenta)|no\s+(puedo|como)\s+antes"
                r"|con\s+dificultad|apenas"
            ),
        ),
        (
            "normal",
            re.compile(
                r"^\W*(sol[oa]|bien|normal|igual|perfecto)\b"
                r"|puedo\s+sol[oa]|sin\s+ayuda|por\s+mi\s+cuenta"
                r"|(como|igual\s+que)\s+antes|el\s+mismo\s+de\s+antes"
            ),
        ),
    ],
    "herida": [
        (
            # Solo el ensuciamiento, nunca el apósito a secas: "la gasa está
            # limpia" es la respuesta CONTRARIA y con `\bgasa\b` suelto salía
            # `secrecion_purulenta`, que dispara CRÍTICO ella sola.
            "secrecion_purulenta",
            re.compile(r"manch\w+|humed\w+|moj\w+|suci\w+"
                       r"|(gasa|aposito|vendaje)\s+\w{0,8}\s?(amarill|verdos|feo)"),
        ),
        (
            "eritema_leve",
            re.compile(r"^\W*(hinchad|inflamad|irritad|morad|rosad|roj)\w*"
                       r"|\b(hinchad|inflamad)\w+"),
        ),
        (
            "normal",
            re.compile(r"^\W*(sec[ao]|limpi[ao]|bien|normal|igual|perfecta)\b"
                       r"|\b(limpi[ao]|sec[ao])\b"
                       r"|como\s+siempre|del\s+color\s+normal|ningun\s+cambio"
                       # Negar el hallazgo que la repregunta ofrece ES contestar
                       # que está normal: "¿ha manchado el apósito?" -> "no ha
                       # manchado nada".
                       r"|\bno\s+\w{0,6}\s?(ha\s+|he\s+)?(manch|bota|sale|sali)\w*"),
        ),
    ],
    "apetito": [
        (
            "muy_disminuido",
            re.compile(r"^\W*(nada|ningun\w*)\b|ni\s+ganas"),
        ),
        (
            "levemente_disminuido",
            re.compile(r"picote\w+|^\W*(poquit|poc)\w*\b|a\s+ratos|regular"),
        ),
        (
            "normal",
            re.compile(r"^\W*(bien|normal|igual|completo|perfecto)\b"
                       r"|algo\s+completo|de\s+todo|como\s+siempre"),
        ),
    ],
    "sueno": [
        (
            "muy_alterado",
            re.compile(r"^\W*(mal|pesimo|horrible|fatal|terrible)\b|nada\s+bien"
                       r"|\b(cuatro|cinco|seis|siete|ocho|nueve|diez|[4-9])\s+veces"),
        ),
        (
            # "¿Cuántas veces se despierta?" — la respuesta es un número, y a
            # partir de cuatro despertares la noche ya no es un sueño interrumpido.
            "levemente_alterado",
            re.compile(r"^\W*(dos|tres|2|3)\b|\b(dos|tres|2|3)\s+veces"
                       r"|a\s+ratos|por\s+ratos|regular|mas\s+o\s+menos"),
        ),
        (
            "normal",
            re.compile(r"^\W*(bien|normal|igual|perfecto|ninguna)\b"
                       r"|de\s+corrido|toda\s+la\s+noche|como\s+siempre"),
        ),
    ],
}

# El dolor no es categórico: sus respuestas cortas devuelven un nivel. "Mucho" y
# "poquito" son literalmente las dos opciones que ofrece su segunda repregunta
# ("Digámoslo fácil: ¿le duele mucho o poquito?") y ninguna de las dos casaba —
# `_DOLOR_DESCRIPTOR` exige "me duele mucho", con el verbo.
_CORTAS_DOLOR: list[tuple[int, re.Pattern[str]]] = [
    (9, re.compile(r"^\W*(muchisimo|terrible|insoportable|horrible)\b")),
    (8, re.compile(r"^\W*(mucho|bastante|harto|fuerte|duro)\b")),
    (5, re.compile(r"^\W*(regular|normal|ahi|medio|mitad)\b|mas\s+o\s+menos")),
    (2, re.compile(r"^\W*(poquit|poc|lev|casi\s+nada)\w*\b")),
    (0, re.compile(r"^\W*(nada|ninguno|ninguna|nulo)\b")),
]

# Afirmación genérica de normalidad: "todo bien", "sin novedad", "nada raro".
# Vale para cualquier slot categórico, y es de lo más común en la capa limpia del
# dataset. Se evalúa la última, para que un "todo bien, aunque se ve rojita" se
# quede con el eritema.
_NORMAL_GENERICO = re.compile(
    r"\btodo\s+(bien|normal|en\s+orden)|sin\s+novedad|nada\s+raro|nada\s+anormal"
    r"|no\s+he\s+tenido\s+(ningun\s+)?problema|(esta|va)\s+todo\s+bien|normal\s*$"
)

_MEDICACION_NO = re.compile(
    r"(la\s+)?(pastilla|droga|medicamento|analgesic\w+|calmante)\s+no\s+(me\s+)?"
    r"(lo\s+)?(quita|sirve|hace\s+(nada|efecto)|calma)"
    r"|no\s+me\s+(hace|ha\s+hecho)\s+(nada|efecto)|ni\s+con\s+las\s+pastillas"
)
_MEDICACION_SI = re.compile(
    r"(la\s+)?(pastilla|droga|medicamento|calmante)\s+(si\s+)?(me\s+)?"
    r"(lo\s+)?(quita|sirve|calma|ayuda|funciona)|con\s+la\s+pastilla\s+se\s+(me\s+)?pasa"
)


def habla_de_medir(t: str) -> bool:
    """¿La frase (ya normalizada) se refiere al acto de tomarse la temperatura?"""
    return bool(_TOMO_TEMPERATURA.search(t) or _SIN_TERMOMETRO.search(t))


# A propósito MÁS ancho que `_FIEBRE_SI`: aquel afirma `fever=True` y tiene que
# ser estricto; este solo decide si el turno menciona algo térmico, y se usa para
# NO suprimir un hallazgo. La asimetría es deliberada — un falso negativo de
# fiebre es la falla catastrófica. Caso real del dataset: "me sen- un poco
# calientica, la tomé y marcaba como [inaudible]" habla de fiebre y de medir a la
# vez; con el patrón estricto (que exige verbo + raíz térmica, y aquí el STT
# cortó "sen-") la guarda se comía un caso rojo.
# No lleva "temperatura", "termómetro" ni "grados": esas son las palabras del
# ACTO de medir, justo lo que hay que poder distinguir del hallazgo.
_MENCION_TERMICA = re.compile(
    r"fiebre|calentura|calient|calor|tibi|destempl|escalofrio|ardiendo|hirviendo"
)


def habla_de_fiebre(t: str) -> bool:
    """¿La frase (ya normalizada) menciona algo térmico, no solo el termómetro?"""
    return bool(_MENCION_TERMICA.search(t))


def respuesta_polar(text: str) -> bool | None:
    """¿El turno es un sí o un no pelados? True, False o None si no lo es.

    Solo la POLARIDAD; qué significa depende de la pregunta. Se expone para el
    orquestador, que la necesita fuera del tamizaje: la oferta de "¿prefiere que
    lo llame una enfermera?" también se contesta con un monosílabo, y hasta ahora
    nadie la leía.
    """
    d = parte_declarativa(text)
    if _POLAR_NO.search(d):
        return False
    if _POLAR_SI.search(d):
        return True
    return None


# Todo lo que un turno puede dejar además de la fiebre. Si algo de esto salió, el
# turno ya se sabe de qué hablaba y su "no" inicial no es una respuesta polar:
# "No he dormido nada" contesta por el sueño, no niega la calentura.
_OTROS_HALLAZGOS: tuple[str, ...] = (
    "pain_level", "mobility", "wound", "appetite", "sleep", "temperature_c",
    *(campo for campo, _ in _BANDERAS),
)


def _resolvio_algo(sym: Symptoms) -> bool:
    return any(getattr(sym, campo) is not None for campo in _OTROS_HALLAZGOS)


# Fragmento interrogativo, mismo criterio que `intent._FRAGMENTO` (se duplica en
# vez de importarse para que este módulo —el núcleo determinista— no dependa del
# clasificador de intención, que sí depende conceptualmente de él).
_FRAGMENTO_PREGUNTA = re.compile(r"¿[^?]{1,80}\?|[^.!?¿]{3,80}\?")

# Marca de que el paciente habla de SÍ MISMO. Es lo único que separa una pregunta
# que reporta un síntoma de una que solo pide información, y la distinción no es
# teórica: sin ella, recortar los fragmentos interrogativos se comía
# "¿es normal que me esté saliendo pus de la herida?" —un `secrecion_purulenta`
# que dispara CRÍTICO por sí solo— y eso es la falla catastrófica de este sistema.
_PRIMERA_PERSONA = re.compile(
    r"\b(me|mi|mis|yo|tengo|siento|estoy|ando|he|mio|mia|conmigo|noto|veo|llevo)\b"
)

# Pregunta definicional SIN signos, que es como la entrega Whisper: "que es
# calentura", "no se que es eso". `_FRAGMENTO_PREGUNTA` exige el `?` literal y
# no la ve, así que "calentura" seguía en la parte declarativa y `_FIEBRE_SI`
# anotaba fiebre de una pregunta. Opera sobre texto YA normalizado (sin
# tildes). Mismo criterio de duplicación que `_FRAGMENTO_PREGUNTA`: no se
# importa de `intent` para que el núcleo determinista no dependa del
# clasificador. Guardas del primer alternante: "lo que es la fiebre, si la he
# tenido" (topicalizador) y "que es lo que le digo" (eco) no son preguntas.
_FRAGMENTO_DEFINICIONAL = re.compile(
    r"(?<!lo\s)\b(que\s+es\s+(?!lo\b)|que\s+significa|que\s+quiere\s+decir"
    r"|a\s+que\s+se\s+refiere|como\s+es\s+eso\s+de|no\s+se\s+que\s+es)"
    r"[^.!?,;]{0,60}"
)

# Palabras del propio patrón definicional y deícticos, que no cuentan como el
# TÉRMINO preguntado a la hora de decidir si la aclaración es sobre la pregunta.
_VACIAS_ACLARACION = frozenset({
    "que", "significa", "quiere", "decir", "refiere", "como", "eso", "esa",
    "ese", "esto", "esta", "este", "cuando", "para", "pero", "entonces",
})


def _aclara_el_termino(t: str, question: str | None) -> bool:
    """¿El turno pregunta por un término de la PROPIA pregunta del guion?

    Es la guarda del caso mixto ("sí, pero qué es calentura"): recortado el
    fragmento definicional queda "sí, pero", y el bloque polar anotaría el
    hallazgo desde ese "sí" — incoherente, porque no se afirma lo que no se
    entiende. Solo aplica si alguna palabra de contenido del fragmento aparece
    en la pregunta; "qué es eso que suena" en mitad de otra cosa no bloquea.
    """
    if not question:
        return False
    m = _FRAGMENTO_DEFINICIONAL.search(t)
    if not m:
        return False
    q = set(normalize(question).split())
    contenido = [w for w in m.group(0).split()
                 if len(w) >= 4 and w not in _VACIAS_ACLARACION]
    return any(w in q for w in contenido)


def parte_declarativa(text: str) -> str:
    """El turno sin sus preguntas IMPERSONALES, ya normalizado.

    Preguntar por un síntoma no es tenerlo. Medido: "¿Qué temperatura se considera
    fiebre?" daba `fever=True` porque `_FIEBRE_SI` abre con `\\bfiebre` sin anclar,
    y de ahí pasaba al motor de decisión como si el paciente hubiera referido
    calentura. Lo mismo con "¿eso es normal?", que afirmaba el slot en curso como
    `normal` sin que el paciente hubiera dicho nada de él.

    Solo se recortan los fragmentos SIN marca de primera persona, y la asimetría
    es deliberada, del mismo signo que el resto del módulo: una pregunta que habla
    del propio cuerpo está reportando ("¿es normal que me esté saliendo pus?"),
    y perder eso sería un falso negativo, que es lo que aquí no se puede permitir.
    Una pregunta impersonal no reporta nada, y quedarse con ella solo produce
    ruido en el triaje.

    La guarda equivalente para la ruta del LLM vive en
    `ollama_adapter._to_symptoms` (`_ES_PREGUNTA`), y no cubría este caso: allí se
    descarta el valor si el turno es una pregunta Y no menciona el slot, y una
    pregunta SOBRE la fiebre menciona la fiebre.

    No se aplica a `_BANDERAS` ni a `temperature_c`: ahí manda la lectura
    conservadora, y "¿es grave no poder respirar?" tiene que escalar igual.
    """
    def _quitar(m: re.Match[str]) -> str:
        frag = normalize(m.group(0))
        return m.group(0) if _PRIMERA_PERSONA.search(frag) else " "

    d = normalize(_FRAGMENTO_PREGUNTA.sub(_quitar, text))

    # Segunda pasada, sobre el texto ya normalizado: la pregunta definicional
    # llega de Whisper SIN signos ("que es calentura") y la primera pasada no
    # la ve. Misma guarda de primera persona y por el mismo motivo: "no se que
    # es eso que me esta saliendo de la herida" reporta una secreción real.
    def _quitar_norm(m: re.Match[str]) -> str:
        return m.group(0) if _PRIMERA_PERSONA.search(m.group(0)) else " "

    return _FRAGMENTO_DEFINICIONAL.sub(_quitar_norm, d)


def extract(text: str, slot: str | None = None, *,
            question: str | None = None) -> Symptoms:
    """Lo que se puede leer del texto sin modelo.

    `slot` indica qué se preguntó, para desambiguar: "normal" significa cosas
    distintas según se esté hablando de la herida o del sueño. Las banderas de
    emergencia y la temperatura se buscan SIEMPRE, se preguntara lo que se
    preguntara, porque el paciente las suelta cuando quiere.

    `question` es la pregunta canónica del guion que el paciente está
    contestando, y hace falta para el sí/no pelado: el slot dice DE QUÉ se habla,
    pero solo la pregunta dice qué significa un "no" (ver `nlu/polaridad.py`).
    Sin ella el resto de la extracción funciona igual.
    """
    sym = Symptoms()
    t = normalize(text)
    # Los hallazgos que el paciente AFIRMA se leen solo de la parte declarativa;
    # las banderas y la temperatura siguen leyéndose del turno entero.
    d = parte_declarativa(text)

    # Fuera de catálogo: se busca SIEMPRE, como las banderas, porque justamente
    # es lo que el guion no pregunta y el paciente sí cuenta.
    sym.other = otros_sintomas.detectar(t)

    for campo, patron in _BANDERAS:
        if patron.search(t):
            setattr(sym, campo, True)
            sym.sources[campo] = "lexicon"

    if m := _TEMP_FRAC.search(t):
        sym.temperature_c = float(m.group(1)) + 0.5
        sym.sources["temperature_c"] = "lexicon"
    elif m := _TEMP.search(t):
        entero, dec = m.group(1), m.group(2)
        sym.temperature_c = float(f"{entero}.{dec}" if dec else entero)
        sym.sources["temperature_c"] = "lexicon"

    if _FIEBRE_NO.search(d):
        sym.fever = False
        sym.sources["fever"] = "lexicon"
    elif _FIEBRE_SI.search(d):
        sym.fever = True
        sym.sources["fever"] = "lexicon"

    # Si mide, se anota aparte de `fever`: son dos hechos distintos y confundirlos
    # es lo que hacía perder la cifra. `_SIN_TERMOMETRO` gana porque contiene
    # formas que también casan con `_TOMO_TEMPERATURA` ("no me la he tomado").
    if _SIN_TERMOMETRO.search(t):
        sym.temperature_measured = False
        sym.sources["temperature_measured"] = "lexicon"
    elif sym.temperature_c is not None or _TOMO_TEMPERATURA.search(t):
        sym.temperature_measured = True
        sym.sources["temperature_measured"] = "lexicon"

    if _MEDICACION_NO.search(t):
        sym.medication_effective = False
        sym.sources["medication_effective"] = "lexicon"
    elif _MEDICACION_SI.search(t):
        sym.medication_effective = True
        sym.sources["medication_effective"] = "lexicon"

    # El dolor se busca siempre, se esté preguntando o no: mismo motivo que el
    # bucle de los categóricos más abajo (el paciente suelta el número del dolor
    # mientras el guion ya pregunta por fiebre o movilidad, y perderlo es perder
    # una escalada real — ver "Recuerda ubicación del dolor" y "Dolor creciente
    # con medicación inútil" en tests/reports/).
    if (nivel := _dolor(t)) is not None:
        sym.pain_level = nivel
        sym.sources["pain_level"] = "lexicon"

    # El slot que se preguntó se resuelve primero, y en tres pasadas de menos a
    # más elíptico: el patrón autosuficiente, la respuesta corta a ESTA pregunta,
    # y la afirmación genérica de normalidad ("todo bien" significa lo que se
    # acaba de preguntar).
    # Aclaración sobre un término de la propia pregunta ("sí, pero qué es
    # calentura"): las respuestas cortas y el sí/no pelado de ESTE slot no se
    # leen — no se afirma lo que no se entiende. Las banderas, la temperatura y
    # los patrones autosuficientes siguen leyéndose del turno entero.
    aclara = _aclara_el_termino(t, question)

    if slot in _CATEGORICOS:
        campo, patrones = _CATEGORICOS[slot]
        if not _primera_coincidencia(sym, campo, patrones, d):
            cortas = [] if aclara else _CORTAS.get(slot, [])
            if not _primera_coincidencia(sym, campo, cortas, d):
                if _afirmado(_NORMAL_GENERICO, d):
                    setattr(sym, campo, "normal")
                    sym.sources[campo] = "lexicon"

    # Y el dolor, que no es categórico pero también se contesta con una palabra.
    if slot == "dolor" and sym.pain_level is None:
        for nivel, patron in _CORTAS_DOLOR:
            if _afirmado(patron, d):
                sym.pain_level = nivel
                sym.sources["pain_level"] = "lexicon"
                break

    # Y después se buscan los DEMÁS slots con sus patrones específicos. El paciente
    # no contesta por turnos: suelta "no he pegado el ojo" mientras le preguntan por
    # la comida, y perder eso es perder una señal de alarma. Medido: sin esto, el
    # sueño se extraía en el 36% de las conversaciones aunque el léxico acierta el
    # 84% cuando se le da el turno correcto — el agente perdía el paso al
    # reformular una pregunta y ya no lo recuperaba.
    #
    # Aquí NO entra `_CORTAS`: "mal" o "me cuesta más" solo significan algo como
    # respuesta a su pregunta, y aplicarlas a los demás slots sería inventarse
    # hallazgos con las palabras de otro tema.
    for otro, (campo, patrones) in _CATEGORICOS.items():
        if otro == slot or getattr(sym, campo) is not None:
            continue
        _primera_coincidencia(sym, campo, patrones, d)

    # El sí/no pelado. Va el ÚLTIMO a propósito: la única forma de saber que ese
    # "no" contesta lo que se preguntó es comprobar que el turno no dejó ningún
    # otro hallazgo, y eso solo se sabe aquí abajo.
    #
    # Qué significa lo dice la PREGUNTA, no el slot (`nlu/polaridad.py`): el mismo
    # "no" vale `normal` tras "¿ha tenido alguna dificultad?" y `limitada_esperada`
    # tras "¿ha podido caminar sin problema?".
    #
    # `habla_de_medir` mantiene la distinción que costó un rojo: "no me la he
    # tomado" y "sí la he tomado" empiezan por no/sí pero contestan al TERMÓMETRO,
    # no al hallazgo. Ahí el slot sigue abierto para que el guion reformule en
    # cerrada (ver `script.seguimiento_pendiente`).
    polar = polaridad.de(question)
    if polar is not None and not aclara and not _resolvio_algo(sym):
        campo, si, no = polar
        estorba = habla_de_medir(t) if campo == "fever" else False
        if (getattr(sym, campo) is None and not estorba
                and not (campo == "fever" and habla_de_fiebre(d))):
            valor = (no if _POLAR_NO.search(d)
                     else si if _POLAR_SI.search(d) else None)
            # Turno contradictorio ("No, yo estoy bien" a "¿ha podido caminar sin
            # problema?"): el monosílabo no puede anotar un hallazgo contra lo que
            # el paciente acaba de afirmar de sí mismo. Se deja sin resolver.
            if valor not in _BENIGNO and _BIEN_GENERICO.search(d):
                valor = None
            if valor is not None:
                setattr(sym, campo, valor)
                sym.sources[campo] = "lexicon"

    return sym


def _primera_coincidencia(sym: Symptoms, campo: str,
                          patrones: list[tuple[str, re.Pattern[str]]],
                          texto: str) -> bool:
    """Anota el primer valor cuyo patrón se afirme en `texto`. ¿Anotó algo?

    Los patrones vienen ordenados de más a menos severo: la primera coincidencia
    gana, que es la lectura conservadora de siempre.
    """
    for valor, patron in patrones:
        if _afirmado(patron, texto):
            setattr(sym, campo, valor)
            sym.sources[campo] = "lexicon"
            return True
    return False


def _dolor(t: str) -> int | None:
    # El dígito explícito manda sobre el descriptor: si el paciente dice "un 3",
    # es un 3, aunque además diga "me duele mucho".
    if m := _DOLOR_NUM.search(t):
        # Se descarta si el número venía de una temperatura ya capturada.
        if not re.search(
            rf"\b3[5-9][.,]?\d*\s*(grados|°)?\s*{re.escape(m.group(1))}\b", t
        ):
            return int(m.group(1))
    # La palabra numérica en la escala ("nueve", "un ocho") pesa igual que el
    # dígito. "Dije nueve." tras una repregunta ES el dato, no una evasiva.
    if m := _DOLOR_NUM_PALABRA.search(t):
        return _NUMEROS[m.group(1)]
    for nivel, patron in _DOLOR_DESCRIPTOR:
        if patron.search(t):
            return nivel
    return None

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
from . import otros_sintomas


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
    r"|(siento|sentido|senti|ando|estoy|estaba|amaneci)\s+.{0,14}(calor|calient|tibi|arde)"
)
_FIEBRE_NO = re.compile(
    r"no\s+(he\s+tenido|tengo|ha\s+tenido)\s+(fiebre|calentura|temperatura)"
    r"|sin\s+fiebre|nada\s+de\s+fiebre|de\s+eso\s+(nada|nada\s+que\s+ver)"
    r"|fiebre\s+no\s+(he\s+tenido|ha\s+habido)|ninguna\s+fiebre"
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


def extract(text: str, slot: str | None = None) -> Symptoms:
    """Lo que se puede leer del texto sin modelo.

    `slot` indica qué se preguntó, para desambiguar: "normal" significa cosas
    distintas según se esté hablando de la herida o del sueño. Las banderas de
    emergencia y la temperatura se buscan SIEMPRE, se preguntara lo que se
    preguntara, porque el paciente las suelta cuando quiere.
    """
    sym = Symptoms()
    t = normalize(text)

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

    if _FIEBRE_NO.search(t):
        sym.fever = False
        sym.sources["fever"] = "lexicon"
    elif _FIEBRE_SI.search(t):
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

    # El slot que se preguntó se resuelve primero, y solo ahí vale la afirmación
    # genérica de normalidad: "todo bien" significa lo que se acaba de preguntar.
    if slot in _CATEGORICOS:
        campo, patrones = _CATEGORICOS[slot]
        for valor, patron in patrones:
            if patron.search(t):
                setattr(sym, campo, valor)
                sym.sources[campo] = "lexicon"
                break
        else:
            if _NORMAL_GENERICO.search(t):
                setattr(sym, campo, "normal")
                sym.sources[campo] = "lexicon"

    # Y después se buscan los DEMÁS slots con sus patrones específicos. El paciente
    # no contesta por turnos: suelta "no he pegado el ojo" mientras le preguntan por
    # la comida, y perder eso es perder una señal de alarma. Medido: sin esto, el
    # sueño se extraía en el 36% de las conversaciones aunque el léxico acierta el
    # 84% cuando se le da el turno correcto — el agente perdía el paso al
    # reformular una pregunta y ya no lo recuperaba.
    for otro, (campo, patrones) in _CATEGORICOS.items():
        if otro == slot or getattr(sym, campo) is not None:
            continue
        for valor, patron in patrones:
            if patron.search(t):
                setattr(sym, campo, valor)
                sym.sources[campo] = "lexicon"
                break

    return sym


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

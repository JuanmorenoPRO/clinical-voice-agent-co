"""Qué significa un "sí" o un "no" pelados, según qué se preguntó.

El paciente contesta una pregunta cerrada con una palabra, y esa palabra solo
significa algo en el contexto de la pregunta. El léxico no puede resolverlo con
el slot abierto porque **la polaridad se invierte dentro del mismo slot**:

    "¿Ha tenido alguna dificultad?"                  -> "no" es `normal`
    "¿Ha podido levantarse y caminar sin problema?"  -> "no" es `limitada_esperada`
    "¿Tiene termómetro a mano para tomársela?"       -> "no" es no haber medido

Medido en una llamada real (conversación 0cf3f8d3, turno 2): el agente preguntó
por el TERMÓMETRO, el paciente dijo "No." y una regla anclada al slot lo leyó
como negación de fiebre. Sobrevivió por la monotonía de `merge_symptoms`, pero
era la interpretación equivocada.

La tabla se indexa por el texto de la pregunta, no por el slot. Y por la pregunta
que el GUION pidió hacer, no por lo que se dijo: el redactor puede reformularla
(`composer.valida` solo exige que termine en interrogación), así que el
orquestador guarda la canónica en `CallState.ultima_pregunta`.

Las cadenas están duplicadas de `agent/phrasing.py` —`nlu` no puede importar
`agent` sin ciclo, igual que `_CAMPO_DE_SLOT` está duplicado en `phrasing.py` y
en `merge.py`—, y la duplicación se sostiene con `tests/test_polaridad.py`, que
falla si alguien reescribe una pregunta y deja su entrada huérfana.
"""
from __future__ import annotations

from typing import Any

# pregunta -> (campo de Symptoms, valor si "sí", valor si "no")
POLARIDAD: dict[str, tuple[str, Any, Any]] = {
    # --- fiebre: las cinco formulaciones preguntan por el mismo hallazgo -------
    "¿Ha tenido fiebre o calentura estos días?": ("fever", True, False),
    "¿Ha sentido calentura o escalofríos desde la cirugía?": ("fever", True, False),
    "¿Ha notado fiebre, o esa sensación de destemple?": ("fever", True, False),
    "¿Se ha sentido caliente o con escalofríos, aunque no se haya tomado la temperatura?":
        ("fever", True, False),
    "Sin termómetro: ¿lo ha sentido como fiebre, sí o no?": ("fever", True, False),

    # --- fiebre, seguimiento: el ACTO de medir, que no es el hallazgo ----------
    # Esta es la entrada que arregla el turno 2 de la llamada reportada.
    "¿Tiene termómetro a mano para tomársela?": ("temperature_measured", True, False),
    "¿Ha podido medírsela con termómetro?": ("temperature_measured", True, False),

    # --- movilidad: aquí la polaridad se invierte entre dos preguntas ---------
    "¿Cómo se siente al moverse o caminar? ¿Ha tenido alguna dificultad?":
        ("mobility", "limitada_esperada", "normal"),
    "¿Ha podido levantarse y caminar sin problema estos días?":
        ("mobility", "normal", "limitada_esperada"),

    # --- herida ---------------------------------------------------------------
    # Un "sí" a "¿tiene enrojecimiento, secreción o hinchazón?" no dice CUÁL de
    # las tres. Se anota `eritema_leve`: registra que algo no está bien sin
    # inventarse una secreción purulenta —que dispara CRÍTICO ella sola— a partir
    # de un monosílabo. Si el paciente lo concreta después, `merge_symptoms` sube.
    "¿Cómo está la herida quirúrgica? ¿Tiene enrojecimiento, secreción o hinchazón?":
        ("wound", "eritema_leve", "normal"),
    "Hábleme de la herida: ¿la ha visto roja, hinchada, o le sale algo?":
        ("wound", "eritema_leve", "normal"),

    # --- apetito --------------------------------------------------------------
    "¿Ha podido comer con normalidad estos días?":
        ("appetite", "normal", "levemente_disminuido"),
    "¿Le provoca comer, sí o no?": ("appetite", "normal", "muy_disminuido"),
}

# Preguntas del guion que a propósito NO tienen polaridad, con el motivo. Existe
# para que el test de sincronía distinga "falta una entrada" de "aquí no la hay",
# y para que añadir una pregunta obligue a decidir cuál de las dos cosas es.
#
# Casi todas son disyuntivas ("¿A o B?"), donde un sí/no pelado no dice a cuál de
# las dos ramas contesta. Ahí no resolver el slot cuesta una repregunta y
# resolverlo al revés cuesta el dato: se omite. Esas respuestas las recoge
# `lexicon._CORTAS`, que sí entiende "solo", "me cuesta más", "bien" o "mal".
SIN_POLARIDAD: frozenset[str] = frozenset({
    # El dolor es una cifra: ningún sí/no es un nivel.
    "¿Cómo ha estado el dolor desde la cirugía, en una escala del cero al diez?",
    "En una escala del cero al diez, ¿cómo calificaría el dolor que siente ahora?",
    "Cuénteme del dolor: si cero es nada y diez es lo peor, ¿en cuánto está?",
    "Para hacerme una idea: ¿el dolor está más cerca de tres o más cerca de ocho?",
    "Digámoslo fácil: ¿le duele mucho o poquito?",
    # Disyuntivas y abiertas.
    "Cuénteme cómo va con el movimiento: ¿camina bien, le cuesta, necesita ayuda?",
    "¿Puede levantarse de la cama solo, o necesita que alguien lo ayude?",
    "¿Camina como antes de la cirugía, o le cuesta más?",
    "¿Cómo ve la herida? ¿Está limpia y seca, o ha notado algún cambio?",
    "¿La herida está seca, o ha manchado el apósito con algo?",
    "¿La ve del color normal de la piel, o más roja de lo que estaba?",
    "¿Cómo ha estado su apetito desde la cirugía?",
    "¿Qué tal el apetito? ¿Le provoca la comida o le ha costado?",
    "¿Ha comido hoy algo completo, o solo picoteado?",
    "¿Cómo ha estado durmiendo desde la cirugía?",
    "¿Ha logrado descansar en las noches, o algo se lo ha impedido?",
    "Cuénteme cómo va el sueño: ¿duerme de corrido o se despierta?",
    "¿Cuántas veces se despierta en la noche, más o menos?",
    "¿Durmió bien anoche o mal?",
    # Piden un número o un recuerdo, no un sí/no.
    "¿Y cuánto le marcó?",
    "¿Se acuerda de cuánto marcaba el termómetro?",
})

# De más larga a más corta: la pregunta llega dentro de una frase mayor (acuse +
# pregunta) y hay formulaciones que son subcadena de otras. La más larga es la
# que de verdad se hizo.
_POR_LONGITUD: list[str] = sorted(POLARIDAD, key=len, reverse=True)


def de(pregunta: str | None) -> tuple[str, Any, Any] | None:
    """Qué significan "sí" y "no" contestados a `pregunta`.

    Acepta tanto la pregunta canónica como la respuesta completa que la contiene
    (acuse + reflejo + pregunta), y devuelve None cuando lo último que dijo el
    agente no fue una pregunta cerrada del guion — el caso de la oferta de
    enfermera y de las preguntas de cierre, donde un "no" no habla de ningún slot.
    """
    if not pregunta:
        return None
    for texto in _POR_LONGITUD:
        if texto in pregunta:
            return POLARIDAD[texto]
    return None

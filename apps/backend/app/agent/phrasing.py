"""Banco de frases deterministas: lo que el agente dice de verdad.

Esto lo generaba el LLM y salió mal. Ante "me duele un berraco" `llama3.2:3b`
devolvió como acuse empático la palabra `"agudo"`, y ante "no me provoca nada",
`"se siente mal"`. No son acuses, son etiquetas. Además costaban ~11 tokens de
salida, que a 40 tok/s son ~275 ms en cada turno.

Con plantillas se gana en las tres dimensiones a la vez: suenan mejor, cuestan
0 ms y se pueden pre-sintetizar en audio durante el arranque.

La rotación anti-repetición no es cosmética. `tests/evaluators/style.py` penaliza
reutilizar la misma apertura o la misma fórmula de empatía dentro de una ventana
de cinco turnos, y con razón: repetir "Entiendo su preocupación" cuatro veces es
lo que hace que un agente suene a máquina.
"""
from __future__ import annotations

import hashlib

# Las seis preguntas canónicas. Varias formulaciones por slot, tomadas del
# fraseo real del dataset, para que dos llamadas seguidas no suenen idénticas.
PREGUNTAS: dict[str, tuple[str, ...]] = {
    "dolor": (
        "¿Cómo ha estado el dolor desde la cirugía, en una escala del cero al diez?",
        "En una escala del cero al diez, ¿cómo calificaría el dolor que siente ahora?",
        "Cuénteme del dolor: si cero es nada y diez es lo peor, ¿en cuánto está?",
    ),
    "fiebre": (
        "¿Ha tenido fiebre o ha notado aumento de temperatura estos días?",
        "¿Se ha tomado la temperatura? ¿Ha sentido escalofríos o calentura?",
        "¿Ha notado fiebre, o esa sensación de destemple, desde la cirugía?",
    ),
    "movilidad": (
        "¿Cómo se siente al moverse o caminar? ¿Ha tenido alguna dificultad?",
        "¿Ha podido levantarse y caminar sin problema estos días?",
        "Cuénteme cómo va con el movimiento: ¿camina bien, le cuesta, necesita ayuda?",
    ),
    "herida": (
        "¿Cómo está la herida quirúrgica? ¿Tiene enrojecimiento, secreción o hinchazón?",
        "Hábleme de la herida: ¿la ha visto roja, hinchada, o le sale algo?",
        "¿Cómo ve la herida? ¿Está limpia y seca, o ha notado algún cambio?",
    ),
    "apetito": (
        "¿Cómo ha estado su apetito desde la cirugía?",
        "¿Ha podido comer con normalidad estos días?",
        "¿Qué tal el apetito? ¿Le provoca la comida o le ha costado?",
    ),
    "sueno": (
        "¿Cómo ha estado durmiendo desde la cirugía?",
        "¿Ha logrado descansar en las noches, o algo se lo ha impedido?",
        "Cuénteme cómo va el sueño: ¿duerme de corrido o se despierta?",
    ),
}

# Reformulaciones progresivamente más cerradas. La primera abre, la segunda
# ancla entre dos valores. Es la técnica que funciona con el paciente evasivo:
# preguntar lo mismo más fuerte no sirve, preguntar más fácil sí.
REPREGUNTAS: dict[str, tuple[str, ...]] = {
    "dolor": (
        "Para hacerme una idea: ¿el dolor está más cerca de tres o más cerca de ocho?",
        "Digámoslo fácil: ¿le duele mucho o poquito?",
    ),
    "fiebre": (
        "¿Se ha sentido caliente o con escalofríos, aunque no se haya tomado la temperatura?",
        "Sin termómetro: ¿lo ha sentido como fiebre, sí o no?",
    ),
    "movilidad": (
        "¿Puede levantarse de la cama solo, o necesita que alguien lo ayude?",
        "¿Camina como antes de la cirugía, o le cuesta más?",
    ),
    "herida": (
        "¿La herida está seca, o ha manchado el apósito con algo?",
        "¿La ve del color normal de la piel, o más roja de lo que estaba?",
    ),
    "apetito": (
        "¿Ha comido hoy algo completo, o solo picoteado?",
        "¿Le provoca comer, sí o no?",
    ),
    "sueno": (
        "¿Cuántas veces se despierta en la noche, más o menos?",
        "¿Durmió bien anoche o mal?",
    ),
}

# Acuses de recibo. Cortos y sin contenido clínico a propósito: reconocen que se
# escuchó, no valoran. Valorar es tarea del motor de decisión.
ACUSES: tuple[str, ...] = (
    "Listo, gracias.", "Bien, tomo nota.", "Entendido.", "Perfecto.",
    "Vale, anotado.", "De acuerdo.", "Le entiendo.", "Ya veo.",
)

# Cuando el paciente reporta algo que no está bien. Reconoce sin dramatizar ni
# tranquilizar: tranquilizar ante un síntoma de alarma es una de las conductas
# que la rúbrica penaliza explícitamente.
ACUSES_PREOCUPANTE: tuple[str, ...] = (
    "Gracias por contármelo.", "Hizo bien en decírmelo.",
    "Qué bueno que me lo cuenta.", "Me alegra que me lo mencione.",
)

# El saludo lleva ya la primera pregunta. No es cosmética: si abre pidiendo permiso
# ("¿le parece bien?"), el paciente contesta "sí" y se gasta un turno entero en
# nada. En el dataset oficial el agente pregunta por el dolor desde el turno cero,
# y esa es también la razón por la que CallState arranca en TAMIZAJE.
APERTURA = (
    "Buenos días, le llamo del hospital para el seguimiento de su cirugía. "
    "Le voy a hacer unas preguntas rápidas. Para empezar: ¿cómo ha estado el dolor "
    "desde la cirugía, en una escala del cero al diez?"
)

PREGUNTA_ABIERTA = "¿Hay algo más que quiera contarme o preguntarme?"

# Meta-conversación: tiene respuesta fija y no toca el RAG ni el modelo.
META_REPETIR = "Claro, se la repito."
META_PROGRESO = "Ya casi, nos faltan un par de preguntas no más."
SOCIAL = "Muy amable, pero mejor sigamos con usted, que es a quien llamo."
FUERA_DE_MISION = (
    "Eso no se lo puedo decir yo: no receto ni cambio tratamientos. "
    "Se lo paso a enfermería y ellos lo orientan."
)

# Puente tras una abstención del RAG ("no tengo información..."). Sin esto, la
# siguiente pregunta clínica se pegaba justo después del "se lo paso a
# enfermería" y sonaba a que el agente ignoró lo que acababa de decir en vez de
# retomar el hilo del seguimiento.
TRANSICION_ABSTENCION: tuple[str, ...] = (
    "Voy a continuar entonces con las preguntas de su seguimiento.",
    "Sigamos entonces con las preguntas correspondientes a su cirugía.",
    "Continuemos con el resto de las preguntas.",
)
RECHAZO = (
    "Sin problema, no le quito más tiempo. Si algo cambia o se siente mal, "
    "comuníquese con el hospital. Que siga bien."
)
NO_ENTENDI = "Perdone, no le escuché bien. ¿Me lo repite?"
TERCERO = "Gracias por acompañarlo, lo que me cuente también me sirve."


def _rotar(opciones: tuple[str, ...], semilla: str, usadas: list[str]) -> str:
    """Elige de forma estable pero variada, evitando lo dicho hace poco.

    Determinista a partir de `semilla` (el id del turno) para que una llamada sea
    reproducible en los tests, y no `random`, que haría irrepetibles los fallos.
    """
    libres = [o for o in opciones if o not in usadas] or list(opciones)
    idx = int(hashlib.sha256(semilla.encode()).hexdigest(), 16) % len(libres)
    return libres[idx]


def pregunta(slot: str, semilla: str, usadas: list[str] | None = None) -> str:
    return _rotar(PREGUNTAS[slot], semilla, usadas or [])


def repregunta(slot: str, intento: int) -> str:
    opciones = REPREGUNTAS[slot]
    return opciones[min(intento - 1, len(opciones) - 1)]


def acuse(semilla: str, usadas: list[str] | None = None, *, preocupante: bool = False) -> str:
    banco = ACUSES_PREOCUPANTE if preocupante else ACUSES
    return _rotar(banco, semilla, usadas or [])


def transicion_abstencion(semilla: str, usadas: list[str] | None = None) -> str:
    return _rotar(TRANSICION_ABSTENCION, semilla, usadas or [])


def cierre_tras_escalamiento(nombre: str | None) -> str:
    """El segundo turno después de un guion de seguridad, y el último.

    El guion completo (con las instrucciones de primeros auxilios) ya se dio en
    el turno anterior; repetirlo aquí sonaría a robot atascado. Esto solo
    confirma que quedó registrado y cierra, sin reabrir la conversación clínica.
    """
    saludo = f"{nombre}, " if nombre else ""
    return (f"Ya quedó registrado lo que me contó, {saludo}y enfermería está al "
            "tanto. Vamos a colgar aquí; si empeora, no espere, llame al 123.")


def cierre(nombre: str | None, escalado: bool) -> str:
    saludo = f"{nombre}, " if nombre else ""
    if escalado:
        return (f"Listo {saludo}con eso ya tengo lo que necesitaba. "
                "Enfermería lo va a contactar por lo que hablamos. "
                "Si algo empeora antes, no espere: llame al 123.")
    return (f"Listo {saludo}muchas gracias por su tiempo. Todo lo que me contó "
            "queda registrado. Si aparece fiebre, la herida cambia o el dolor "
            "aumenta, comuníquese con el hospital. Que siga bien.")


# Todo el texto fijo que conviene pre-sintetizar al arrancar: con esto, el TTS de
# un turno normal cuesta 0 ms.
def textos_cacheables() -> list[str]:
    fijos = [APERTURA, PREGUNTA_ABIERTA, META_REPETIR, META_PROGRESO, SOCIAL,
             FUERA_DE_MISION, RECHAZO, NO_ENTENDI, TERCERO]
    for banco in (PREGUNTAS, REPREGUNTAS):
        for opciones in banco.values():
            fijos.extend(opciones)
    fijos.extend(ACUSES)
    fijos.extend(ACUSES_PREOCUPANTE)
    fijos.extend(TRANSICION_ABSTENCION)
    return fijos

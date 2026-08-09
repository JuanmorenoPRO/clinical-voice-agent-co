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

from ..nlu import otros_sintomas, procedimiento

# Las seis preguntas canónicas. Varias formulaciones por slot, tomadas del
# fraseo real del dataset, para que dos llamadas seguidas no suenen idénticas.
PREGUNTAS: dict[str, tuple[str, ...]] = {
    "dolor": (
        "¿Cómo ha estado el dolor desde la cirugía, en una escala del cero al diez?",
        "En una escala del cero al diez, ¿cómo calificaría el dolor que siente ahora?",
        "Cuénteme del dolor: si cero es nada y diez es lo peor, ¿en cuánto está?",
    ),
    # Una sola pregunta, y sobre la SENSACIÓN, no sobre el termómetro. La variante
    # antigua encadenaba dos ("¿se la tomó? ¿sintió calentura?") y el enum de
    # extracción solo puede contestar una: un "sí me la tomé" se guardaba como
    # `fever=True` y el slot quedaba resuelto sin la cifra, que es la que dispara
    # `fiebre_38`. El termómetro se pregunta después, en SEGUIMIENTO.
    "fiebre": (
        "¿Ha tenido fiebre o calentura estos días?",
        "¿Ha sentido calentura o escalofríos desde la cirugía?",
        "¿Ha notado fiebre, o esa sensación de destemple?",
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
    # Ver SEGUIMIENTOS más abajo: la fiebre tiene además una repregunta por la
    # cifra, que no consume estos dos intentos porque el paciente sí contestó.
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

# Se abandona un slot: se dice, no se calla. Hasta ahora el guion pasaba al
# siguiente en silencio, y desde fuera eso se lee como que no escuchó.
SLOT_PERDIDO: tuple[str, ...] = (
    "Lo dejo anotado como que no me lo pudo precisar, y sigo.",
    "No importa, lo anoto como que no me supo decir.",
    "Sin problema, lo dejo así anotado.",
)

# Varios turnos sin sacar un dato. Insistir con otra variante de la misma
# pregunta no funciona; ofrecerle una salida sí, y además es lo correcto: si la
# llamada no puede evaluarlo, quien tiene que hablarle es una persona.
OFRECER_SALIDA: tuple[str, ...] = (
    "Le noto que le cuesta contestarme por teléfono y no quiero incomodarlo. "
    "¿Prefiere que le pida a una enfermera que lo llame?",
    "Veo que por aquí no nos estamos entendiendo bien, y no es culpa suya. "
    "¿Le parece si mejor le pido a una enfermera que lo contacte?",
)


def slot_perdido(semilla: str, usadas: list[str] | None = None) -> str:
    return _rotar(SLOT_PERDIDO, semilla, usadas or [])


def ofrecer_salida(semilla: str, usadas: list[str] | None = None) -> str:
    return _rotar(OFRECER_SALIDA, semilla, usadas or [])


# El paciente no dice nada. No es lo mismo que no entenderle —ahí vale "¿me lo
# repite?"—: aquí lo primero que hay que averiguar es si sigue al teléfono, y por
# eso se pregunta por él antes de volver al guion.
SONDEO_PRESENCIA: tuple[str, ...] = (
    "¿Sigue ahí? No le escuché nada.",
    "¿Aló? No le oigo, no sé si se cortó la llamada.",
    "¿Me escucha bien? No me llegó su respuesta.",
)

# Segundo silencio seguido. Se avisa de lo que va a pasar ANTES de que pase: una
# llamada de seguimiento que se corta sin previo aviso se vive como un cuelgue.
AVISO_ULTIMO_INTENTO: tuple[str, ...] = (
    "Sigo sin escucharlo. Si no me contesta, voy a tener que terminar la llamada.",
    "Todavía no le oigo nada. Si no me responde ahora, doy la llamada por terminada.",
)

# Tercer silencio: se cumple el aviso. No se despide con un "que siga bien" —eso
# afirmaría que el paciente está bien, y precisamente no se sabe—, y se dice en
# voz alta que alguien va a volver a intentarlo, que es lo que de verdad ocurre:
# el cierre levanta una alerta ALTA (ver agent/orchestrator.py).
CIERRE_SIN_RESPUESTA = (
    "No logro escucharlo, así que voy a terminar la llamada por ahora. "
    "Aviso a enfermería para que lo contacten. Si me está oyendo y se siente mal, "
    "llame al 123."
)


def sondeo(intento: int, semilla: str, usadas: list[str] | None = None) -> str:
    """Qué se dice en el silencio número `intento` (1 = sondeo, 2 = aviso)."""
    banco = SONDEO_PRESENCIA if intento <= 1 else AVISO_ULTIMO_INTENTO
    return _rotar(banco, semilla, usadas or [])


# Lo que el paciente cuenta fuera del guion (nlu/otros_sintomas.py). Sin esto, en
# una llamada real dijo ocho veces "veo borroso" y el agente no lo mencionó nunca:
# nombrarlo es la diferencia entre escuchar y rellenar un formulario. No valora
# gravedad —eso es del motor de decisión— y no tranquiliza; solo confirma que se
# anotó y a dónde va.
ACUSE_OTRO: tuple[str, ...] = (
    "Lo de {} se lo anoto y se lo paso a enfermería.",
    "Me apunto lo de {} para que enfermería lo revise.",
    "Anoto lo de {} y queda en su reporte.",
)

# Puente para retomar el guion después de atender lo que el paciente trajo. Sin
# él, la siguiente pregunta se pega al acuse y suena a que se ignoró lo dicho.
VOLVIENDO: tuple[str, ...] = (
    "Volviendo a lo que le preguntaba:",
    "Sigamos con el seguimiento:",
    "Retomo entonces:",
)


# El paciente cuenta un síntoma Y pregunta si es normal. Esa pregunta no la
# responde el corpus: pide un juicio clínico sobre su propio caso, que es
# justamente lo que el agente no hace. Medido en una llamada real — "estoy viendo
# borroso, ¿es normal?" mandó al RAG a buscar en PDFs de radiología de
# apendicitis y el 3B contestó "No, no es normal. Debe ser compresible".
SINTOMA_CONSULTADO: tuple[str, ...] = (
    "Eso no se lo puedo valorar yo por teléfono, pero lo de {} hay que revisarlo. "
    "Lo dejo anotado para que enfermería lo contacte.",
    "No soy quien para decirle si lo de {} es esperable o no; eso lo valora el "
    "equipo. Se lo paso a enfermería con lo que me ha contado.",
)


def sintoma_consultado(etiquetas: list[str], semilla: str,
                       usadas: list[str] | None = None) -> str:
    cuales = etiquetas[0] if len(etiquetas) == 1 else (
        ", ".join(etiquetas[:-1]) + f" y {etiquetas[-1]}")
    return _rotar(SINTOMA_CONSULTADO, semilla, usadas or []).format(cuales)


def volviendo(semilla: str, usadas: list[str] | None = None) -> str:
    return _rotar(VOLVIENDO, semilla, usadas or [])


# --- lo que el paciente cuenta sobre su propia cirugía -------------------------
# El procedimiento entraba solo por la ficha y nunca por la conversación. Medido
# en una llamada real: a la pregunta de fiebre el paciente contestó "me operaron
# de cesárea" y recibió "Sin termómetro: ¿lo ha sentido como fiebre, sí o no?".
# Nombrarlo es la diferencia entre escuchar y rellenar un formulario, igual que
# con ACUSE_OTRO.

PROCEDIMIENTO_CONFIRMADO: tuple[str, ...] = (
    "De su {}, así lo tengo aquí.",
    "Sí, de su {}, eso es lo que tengo registrado.",
    "Correcto, su {}.",
)

# Discrepancia con la ficha. Se dice lo que el paciente dijo Y lo que hay
# registrado —callar cualquiera de los dos sería raro— y se le explica qué se
# hace con eso. NO se le corrige ni se le lleva la contraria: quién tiene razón
# no se resuelve por teléfono, lo verifica enfermería (ver nlu/procedimiento.py).
PROCEDIMIENTO_DISCREPA: tuple[str, ...] = (
    "De {dicho}, entonces — yo tengo registrada una {ficha}. "
    "Lo dejo anotado para que enfermería lo verifique.",
    "Anoto que me dice {dicho}; en mi registro aparece una {ficha}. "
    "Se lo paso a enfermería para que lo confirmen.",
)


# Sin ficha contra la que contrastar (llamada sin paciente asociado). No se puede
# decir "así lo tengo aquí" de algo que no se tiene: se anota y ya.
PROCEDIMIENTO_ANOTADO: tuple[str, ...] = (
    "De {}, lo dejo anotado.",
    "Anoto que fue de {}.",
)


def procedimiento_confirmado(nombre: str, semilla: str,
                             usadas: list[str] | None = None) -> str:
    return _rotar(PROCEDIMIENTO_CONFIRMADO, semilla, usadas or []).format(nombre.lower())


def procedimiento_anotado(nombre: str, semilla: str,
                          usadas: list[str] | None = None) -> str:
    return _rotar(PROCEDIMIENTO_ANOTADO, semilla, usadas or []).format(nombre.lower())


def procedimiento_discrepa(dicho: str, ficha: str, semilla: str,
                           usadas: list[str] | None = None) -> str:
    return _rotar(PROCEDIMIENTO_DISCREPA, semilla, usadas or []).format(
        dicho=dicho.lower(), ficha=ficha.lower())


def abstencion_procedimiento(dicho: str, ficha: str | None) -> str:
    """El paciente pregunta por una cirugía que no es la suya según la ficha.

    Es una abstención, no una respuesta: el corpus está indexado por
    procedimiento (`rag/retrieve.py::_allowed_document_ids`) y contestar con los
    documentos de otra cirugía es una afirmación falsa CON fuente, que es lo peor
    que puede producir este sistema. Ocurrió: una pregunta sobre la cesárea se
    respondió citando cuatro PDFs de apendicectomía.
    """
    tengo = f"lo que tengo aquí es su {ficha.lower()}" if ficha else "no tengo su cirugía registrada"
    return (f"Sobre {dicho.lower()} no le puedo responder desde aquí, porque {tengo}. "
            "Prefiero que se lo confirme enfermería antes que decirle algo que no me consta.")


def acuse_otro(etiquetas: list[str], semilla: str, usadas: list[str] | None = None) -> str:
    """Reconoce por su nombre lo que el paciente contó fuera del guion."""
    if not etiquetas:
        return ""
    if len(etiquetas) == 1:
        cuales = etiquetas[0]
    else:
        cuales = ", ".join(etiquetas[:-1]) + f" y {etiquetas[-1]}"
    return _rotar(ACUSE_OTRO, semilla, usadas or []).format(cuales)


# Seguimientos: el paciente SÍ contestó, pero la respuesta confirma sin
# cuantificar y falta el dato que decide. No son repreguntas —no gastan
# MAX_REPREGUNTAS— porque insistir aquí no es insistir sobre alguien que evade,
# es terminar de anotar lo que ya empezó a contar.
#
# `cifra`: dijo que se la tomó pero no dio el número. Sin él, un 38.5 real queda
# fuera de `fiebre_38` y el caso rojo se vuelve verde — el falso negativo que la
# rúbrica considera catastrófico.
# `termometro`: refiere calentura sin haber medido. Si dice que no tiene, se
# acepta y se avanza: es una respuesta legítima (muchos pacientes no tienen), y
# `_fever_reported_unmeasured` ya cubre el riesgo sin la cifra.
SEGUIMIENTOS: dict[str, tuple[str, ...]] = {
    "cifra": (
        "¿Y cuánto le marcó?",
        "¿Se acuerda de cuánto marcaba el termómetro?",
    ),
    "termometro": (
        "¿Tiene termómetro a mano para tomársela?",
        "¿Ha podido medírsela con termómetro?",
    ),
}


def seguimiento(clase: str, semilla: str, usadas: list[str] | None = None) -> str:
    return _rotar(SEGUIMIENTOS[clase], semilla, usadas or [])


# Acuses de recibo. Cortos y sin contenido clínico a propósito: reconocen que se
# escuchó, no valoran. Valorar es tarea del motor de decisión.
ACUSES: tuple[str, ...] = (
    "Listo, gracias.", "Bien, tomo nota.", "Entendido.", "Perfecto.",
    "Vale, anotado.", "De acuerdo.", "Le entiendo.", "Ya veo.",
)

# El paciente habló y no se le pudo sacar el dato. Es DISTINTO de ACUSES y la
# diferencia es de honestidad, no de estilo: "vale, anotado" afirma que se anotó
# algo, y aquí no se anotó nada (esa es la doctrina del docstring de
# `orchestrator._texto_de`). Estas reconocen haber OÍDO, que es lo único cierto.
#
# Sin ellas, un turno que no resuelve el slot sale como la repregunta pelada —
# "Sin termómetro: ¿lo ha sentido como fiebre, sí o no?"— pegada a lo que sea que
# haya dicho el paciente, y desde fuera eso se lee como no escuchar. Es el caso
# que motivó todo este cambio.
# Sin vocativos con género ("sí señor"): la mitad del padrón son mujeres y el
# sistema no guarda el género. Todas valen para cualquier paciente.
# Disjunto de ACUSES a propósito: significan cosas distintas y hay que poder
# distinguirlos en la traza y en los tests. "Ya veo." vive en ACUSES.
ACUSE_ESCUCHADO: tuple[str, ...] = (
    "Le escucho.", "Ajá.", "Claro.", "Le sigo.", "Ya.", "Sí, dígame.",
)


def acuse_escuchado(semilla: str, usadas: list[str] | None = None) -> str:
    return _rotar(ACUSE_ESCUCHADO, semilla, usadas or [])


# Cuando el paciente reporta algo que no está bien. Reconoce sin dramatizar ni
# tranquilizar: tranquilizar ante un síntoma de alarma es una de las conductas
# que la rúbrica penaliza explícitamente.
# Ocho, no cuatro: la ventana anti-repetición es de cinco turnos, así que con
# cuatro opciones se agotaban y `_rotar` caía al respaldo repitiendo.
ACUSES_PREOCUPANTE: tuple[str, ...] = (
    "Gracias por contármelo.", "Hizo bien en decírmelo.",
    "Qué bueno que me lo cuenta.", "Me alegra que me lo mencione.",
    "Menos mal me lo dice.", "Vale, eso me sirve saberlo.",
    "Le agradezco que me lo comparta.", "Tomo nota de eso.",
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

# Trámites: horarios, costos, EPS, transporte, citas. No los responde el corpus
# clínico y no son de esta llamada, pero callarlos tampoco vale: `intent ==
# pregunta_administrativa` no tenía rama propia y caía en el `else` genérico, así
# que el paciente preguntaba por el horario de visitas y el agente seguía con la
# siguiente pregunta del guion como si no hubiera oído nada.
ADMINISTRATIVA = (
    "Eso es de administración y desde aquí no se lo puedo consultar. "
    "Llamando al conmutador del hospital se lo resuelven."
)

# El paciente saluda. Se le devuelve el saludo y se retoma la pregunta pendiente
# en su forma abierta. Antes esto caía en `respuesta`, no aportaba dato y gastaba
# un reintento: el agente contestaba a un "Hola, buenas." con la reformulación
# cerrada del dolor ("¿más cerca de tres o de ocho?"), que es exactamente lo que
# hace que suene a máquina.
SALUDO_DE_VUELTA: tuple[str, ...] = (
    "Buenos días, gracias por contestar.",
    "Hola, muy buenas, gracias por atenderme.",
    "Buenas, qué bueno que lo encuentro.",
)


def saludo_de_vuelta(semilla: str, usadas: list[str] | None = None) -> str:
    return _rotar(SALUDO_DE_VUELTA, semilla, usadas or [])


# Fase de CONFIRMACIÓN: el agente ya dijo lo suyo y le devuelve el turno al
# paciente en vez de colgar. NO repite el guion de seguridad —ese bug ya está
# cubierto por test_orchestrator_close.py—, solo comprueba que quedó claro.
CONFIRMAR_CIERRE: tuple[str, ...] = (
    "¿Le quedó claro lo que va a pasar? ¿Tiene alguna otra duda antes de colgar?",
    "¿Hay algo más en lo que le pueda ayudar antes de despedirnos?",
    "¿Quedamos así, o quiere preguntarme algo más?",
)


def confirmar_cierre(semilla: str, usadas: list[str] | None = None) -> str:
    return _rotar(CONFIRMAR_CIERRE, semilla, usadas or [])
TERCERO = "Gracias por acompañarlo, lo que me cuente también me sirve."


def _rotar(opciones: tuple[str, ...], semilla: str, usadas: list[str]) -> str:
    """Elige de forma estable pero variada, evitando lo dicho hace poco.

    Determinista a partir de `semilla` (el id del turno) para que una llamada sea
    reproducible en los tests, y no `random`, que haría irrepetibles los fallos.
    """
    # `usadas` son las RESPUESTAS completas de los últimos turnos, no las opciones
    # sueltas, así que comparar por igualdad nunca casaba y la rotación no evitaba
    # nada: el agente decía "Gracias por contármelo." dos turnos seguidos. Se
    # comprueba por contención.
    libres = [o for o in opciones
              if not any(o in u for u in usadas)] or list(opciones)
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


# Reflejo de lo entendido: repetirle al paciente el dato que acaba de dar. Es lo
# que separa "escuchar" de "rellenar un formulario", y a diferencia de ACUSES
# —que son ocho frases intercambiables sin contenido— demuestra QUÉ se entendió,
# que es además la forma de que el paciente corrija si el STT se equivocó.
#
# Sigue sin valorar gravedad: eso es del motor de decisión, y decirle a alguien
# que su dolor "no es para tanto" es justo la conducta que la rúbrica penaliza.
# Cuando el riesgo no es NORMAL manda ACUSES_PREOCUPANTE y esto no se usa.
_REFLEJO_DOLOR: tuple[tuple[int, str], ...] = (
    (0, "Cero, entonces: sin dolor."),
    (2, "Un {n}, bastante llevadero."),
    (5, "Un {n}, ahí en la mitad."),
    (7, "Un {n}, eso ya molesta bastante."),
    (10, "Un {n}, es un dolor fuerte."),
)

_REFLEJO: dict[str, dict[str, str]] = {
    "fever": {"True": "Con calentura, entonces.", "False": "Sin fiebre, bien."},
    "mobility": {
        "normal": "Se mueve sin problema, bien.",
        "limitada_esperada": "Se mueve pero con cuidado, entendido.",
        "incapacitante_nueva": "Que no pueda moverse como antes, lo anoto.",
    },
    "wound": {
        "normal": "La herida limpia y seca, bien.",
        "eritema_leve": "Rojita en el borde, tomo nota.",
        "secrecion_purulenta": "Que le esté saliendo secreción, lo anoto.",
    },
    "appetite": {
        "normal": "Comiendo bien, me alegra.",
        "levemente_disminuido": "Comiendo un poco menos, anotado.",
        "muy_disminuido": "Casi sin comer, lo anoto.",
    },
    "sleep": {
        "normal": "Durmiendo bien, qué bueno.",
        "levemente_alterado": "Durmiendo a ratos, anotado.",
        "muy_alterado": "Durmiendo mal, lo anoto.",
    },
}

# slot del guion -> campo de Symptoms. Duplicado deliberado de script.SLOT_FIELD:
# este módulo no importa `script` (sería un ciclo) y son tres líneas.
_CAMPO_DE_SLOT = {"dolor": "pain_level", "fiebre": "fever", "movilidad": "mobility",
                  "herida": "wound", "apetito": "appetite", "sueno": "sleep"}


def reflejo(slot: str | None, symptoms, del_turno=None) -> str | None:
    """Devuelve el dato entendido dicho en voz alta, o None si no hay qué reflejar.

    `del_turno` son los síntomas que aportó ESTE turno. Si se pasa, solo se
    refleja lo que el paciente acaba de decir: `symptoms` es el acumulado de la
    llamada, así que sin este filtro el agente repetía "Con calentura, entonces."
    en cada turno posterior a la fiebre, que suena a bucle.
    """
    campo = _CAMPO_DE_SLOT.get(slot or "")
    if campo is None:
        return None
    if del_turno is not None and getattr(del_turno, campo, None) is None:
        return None
    valor = getattr(symptoms, campo, None)
    if valor is None:
        return None
    if campo == "pain_level":
        for tope, plantilla in _REFLEJO_DOLOR:
            if valor <= tope:
                return plantilla.format(n=valor)
        return None
    return _REFLEJO.get(campo, {}).get(str(valor))


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
             FUERA_DE_MISION, RECHAZO, NO_ENTENDI, TERCERO, ADMINISTRATIVA,
             CIERRE_SIN_RESPUESTA]
    for banco in (PREGUNTAS, REPREGUNTAS):
        for opciones in banco.values():
            fijos.extend(opciones)
    fijos.extend(ACUSES)
    fijos.extend(ACUSES_PREOCUPANTE)
    fijos.extend(TRANSICION_ABSTENCION)
    fijos.extend(SLOT_PERDIDO)
    fijos.extend(OFRECER_SALIDA)
    fijos.extend(SONDEO_PRESENCIA)
    fijos.extend(AVISO_ULTIMO_INTENTO)
    fijos.extend(VOLVIENDO)
    fijos.extend(SALUDO_DE_VUELTA)
    fijos.extend(CONFIRMAR_CIERRE)
    fijos.extend(ACUSE_ESCUCHADO)
    # Las de procedimiento llevan el nombre de la cirugía dentro, así que se
    # expanden por las etiquetas conocidas — son nueve y cerradas
    # (`nlu/procedimiento`), igual que se hace abajo con ACUSE_OTRO.
    for plantilla in PROCEDIMIENTO_CONFIRMADO:
        fijos.extend(plantilla.format(p.lower()) for p in procedimiento.etiquetas())
    for opciones in SEGUIMIENTOS.values():
        fijos.extend(opciones)
    # Los reflejos del dolor llevan {n}: se enumeran los once valores para que
    # sigan siendo texto fijo y el TTS los pueda pre-sintetizar.
    for n in range(11):
        for tope, plantilla in _REFLEJO_DOLOR:
            if n <= tope:
                fijos.append(plantilla.format(n=n))
                break
    for valores in _REFLEJO.values():
        fijos.extend(valores.values())
    # ACUSE_OTRO lleva una etiqueta variable: se pre-sintetizan las combinaciones
    # de una sola etiqueta, que son la inmensa mayoría de los turnos.
    # Las que llevan una etiqueta variable se expanden por etiqueta, que es el
    # caso de la inmensa mayoría de los turnos (una sola señal).
    for plantilla in (*ACUSE_OTRO, *SINTOMA_CONSULTADO):
        fijos.extend(plantilla.format(e) for e in otros_sintomas.etiquetas())
    return fijos

"""Guiones deterministas de seguridad para nivel CRÍTICO (ADR-006).

En CRÍTICO no se confía ni la redacción al LLM: se descarta su respuesta y se
emite este texto fijo, revisado y testeable, junto con la alerta y el cierre
seguro de la llamada. Tono empático; validar escuchándolo en la voz TTS elegida.

⏳ 7 de agosto: guiones por tipo de cirugía con el dataset.
"""
from __future__ import annotations

_DEFAULT = (
    "Lo que me cuenta puede ser una señal de alarma y su seguridad es lo primero. "
    "Voy a avisar de inmediato al personal de enfermería para que lo contacten ya. "
    "Por favor no se mueva de donde está; si empeora o pierde el conocimiento, "
    "llame a emergencias al 123. Quédese en la línea un momento."
)

_BY_RULE: dict[str, str] = {
    "sangrado_abundante": (
        "El sangrado que describe requiere atención inmediata. Voy a alertar ahora "
        "mismo al personal de enfermería. Mientras tanto, aplique presión firme sobre "
        "la zona con un paño limpio y no la retire. Si el sangrado no cede, llame a "
        "emergencias al 123."
    ),
    "dificultad_respiratoria": (
        "La dificultad para respirar es urgente. Estoy avisando al personal de "
        "enfermería en este momento. Intente sentarse derecho y respirar despacio. "
        "Si le cuesta más respirar, llame de inmediato a emergencias al 123."
    ),
    "perdida_consciencia": (
        "Un desmayo tras la cirugía es una señal seria. Voy a alertar al personal de "
        "enfermería ahora mismo. Si hay alguien con usted, pídale que se quede a su "
        "lado; si vuelve a desvanecerse, deben llamar a emergencias al 123."
    ),
    "dolor_toracico": (
        "Un dolor en el pecho después de la cirugía es una señal de alarma que hay que "
        "atender de inmediato. Estoy alertando ahora mismo al personal de enfermería. "
        "Por favor quédese quieto y sentado; si el dolor aumenta, se irradia al brazo o "
        "le falta el aire, llame de una vez a emergencias al 123."
    ),
    "estado_mental_alterado": (
        "La confusión o desorientación tras la cirugía es una señal seria y su seguridad "
        "es lo primero. Voy a avisar de inmediato al personal de enfermería. Si hay "
        "alguien con usted, pídale que se quede a su lado y, si empeora, que llame a "
        "emergencias al 123."
    ),
    "convulsion": (
        "Una convulsión es una emergencia. Estoy alertando al personal de enfermería en "
        "este momento. Si alguien está con usted, que lo recueste de lado en un lugar "
        "seguro, que no le sujete la lengua ni la boca, y que llame ya a emergencias al 123."
    ),
}


# Apertura empática común: ADR-006 advierte que el guion crítico "puede sonar
# robótico" y prescribe "redactarlo con tono empático". Validamos la emoción ANTES
# de la instrucción clínica (mismo principio que la categoría emocional de las
# pruebas: primero validar, luego lo clínico).
_EMPATHIC_OPENER = (
    "Entiendo que esto lo puede asustar y estoy aquí con usted; vamos a actuar de "
    "inmediato. "
)


# Los rojos derivados del dataset (fiebre ≥38, herida purulenta, movilidad
# incapacitante, dolor ≥8) son complicaciones postoperatorias reales, no emergencias
# vitales. Escalan igual de rápido, pero el guion no puede sonar a infarto: alarmar
# de más a un paciente que lo que necesita es una consulta hoy también es un daño.
_PRIORITARIA: dict[str, str] = {
    "fiebre_38": (
        # Sin género: el resto de guiones dice "lo contacte" y este decía "la
        # contacte"/"desorientada". El sistema no sabe el género del paciente, así
        # que aquí se evita la concordancia en vez de acertarla por defecto.
        "Esa fiebre después de la cirugía sí hay que revisarla hoy mismo, no es algo "
        "para esperar. Voy a pedir que enfermería se comunique con usted en las "
        "próximas horas. Mientras tanto tome abundante líquido y vuelva a medirse la "
        "temperatura en un par de horas. Si pasa de 39, le cuesta respirar o nota "
        "confusión, llame al 123 sin esperar la llamada."
    ),
    "herida_purulenta": (
        "Esa secreción en la herida hay que valorarla hoy, porque puede ser una "
        "infección que necesita tratamiento. Voy a pedir que enfermería lo contacte en "
        "las próximas horas. No se aplique cremas ni destape la herida; cúbrala con una "
        "gasa limpia y seca. Si aparece fiebre alta o el enrojecimiento se extiende, "
        "llame al 123."
    ),
    "movilidad_incapacitante": (
        "Que de un momento a otro no pueda moverse como antes es un cambio que hay que "
        "revisar hoy. Voy a pedir que enfermería lo contacte en las próximas horas. "
        "Evite forzarse y pida ayuda para levantarse. Si siente la pierna hinchada, "
        "caliente o le falta el aire, llame al 123 de inmediato."
    ),
    "dolor_severo": (
        "Un dolor de esa intensidad a estas alturas no es lo esperado y hay que "
        "revisarlo hoy. Voy a pedir que enfermería lo contacte en las próximas horas. "
        "Tome la analgesia como se la formularon y quédese en reposo. Si el dolor sigue "
        "subiendo, aparece fiebre o vomita, llame al 123."
    ),
    "fiebre_referida_con_signos": (
        "Con la fiebre que me contó, aunque no la haya podido medir, y las otras "
        "molestias que me ha ido contando, sí hay que revisarlo hoy. Voy a pedir que "
        "enfermería lo contacte en las próximas horas. Si consigue un termómetro, "
        "tómese la temperatura y anótela. Si pasa de 39 o le cuesta respirar, llame "
        "al 123 sin esperar."
    ),
    "no_se_pudo_evaluar": (
        "No alcanzamos a repasar bien cómo ha seguido, y prefiero no quedarme con la "
        "duda. Voy a pedir que enfermería lo llame hoy para revisarlo con calma. Si "
        "mientras tanto le da fiebre, la herida cambia o el dolor aumenta, comuníquese "
        "al 123."
    ),
}

_PRIORITARIA_DEFAULT = (
    "Lo que me cuenta hay que revisarlo hoy mismo. Voy a pedir que el personal de "
    "enfermería lo contacte en las próximas horas para valorarlo. Si mientras tanto "
    "empeora de forma marcada, no espere la llamada y comuníquese al 123."
)

# Apertura más contenida para la vía de enfermería: reconoce sin dramatizar.
# Con punto y espacio: todos los guiones de abajo empiezan en mayúscula, y sin el
# separador salía pegado ("Gracias por contármelo,Esa fiebre...").
_CALM_OPENER = "Gracias por contármelo. "


def script_for(triggered_rules: list[str], action: str = "emergencia_123") -> str:
    """Selecciona el guion determinista según la vía de escalamiento y la regla.

    Antepone una validación emocional breve (ADR-006) al guion clínico. El tono lo
    fija `action`: el 123 abre reconociendo el susto; enfermería prioritaria abre
    agradeciendo el reporte, porque no es una emergencia vital.
    """
    if action == "enfermeria_prioritaria":
        for rule in triggered_rules:
            if rule in _PRIORITARIA:
                return _CALM_OPENER + _PRIORITARIA[rule]
        return _CALM_OPENER + _PRIORITARIA_DEFAULT

    for rule in triggered_rules:
        if rule in _BY_RULE:
            return _EMPATHIC_OPENER + _BY_RULE[rule]
    return _EMPATHIC_OPENER + _DEFAULT

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


def script_for(triggered_rules: list[str]) -> str:
    """Selecciona el guion según la primera regla CRÍTICA disparada.

    Antepone una validación emocional breve (ADR-006) al guion clínico determinista.
    """
    for rule in triggered_rules:
        if rule in _BY_RULE:
            return _EMPATHIC_OPENER + _BY_RULE[rule]
    return _EMPATHIC_OPENER + _DEFAULT

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
}


def script_for(triggered_rules: list[str]) -> str:
    """Selecciona el guion según la primera regla CRÍTICA disparada."""
    for rule in triggered_rules:
        if rule in _BY_RULE:
            return _BY_RULE[rule]
    return _DEFAULT

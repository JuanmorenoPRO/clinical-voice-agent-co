"""De qué dice el paciente que lo operaron.

El procedimiento entraba al sistema por un solo sitio —`Patient.surgery`, la
ficha— y nunca por la conversación. Eso dejaba un agujero que se vio en una
llamada real: a la pregunta de fiebre el paciente contestó "me operaron de
cesárea" y el agente respondió "Sin termómetro: ¿lo ha sentido como fiebre, sí o
no?", sin una palabra sobre lo que acababa de oír. No es que el LLM no analizara
la respuesta: es que su esquema de salida para ese slot es
`{"fiebre": "si"|"no"|"no_dice"}` y no existía ningún canal por el que pudiera
decir "el paciente me habló de su cirugía". El dato se perdía por construcción.

Este módulo es ese canal, con la misma forma que `otros_sintomas.py`: una tabla
de patrones y un `detectar()` que recibe el texto ya normalizado. Determinista a
propósito — de qué operaron a alguien decide qué documentos puede citar el RAG,
y eso no puede depender de un 3B.

**La ficha manda.** Lo que el paciente dice no reescribe el registro clínico: se
reconoce en voz alta, se anota, y si no coincide se levanta una alerta para que
enfermería lo verifique. Ver `agent/orchestrator.py`.

Dos niveles por procedimiento, y la distinción es lo que evita el falso positivo
que arruinaría esto:

- `nombre`: la palabra clínica, que solo significa la cirugía ("cesárea",
  "apendicectomía"). Cuenta por sí sola.
- `coloquial`: cómo lo dice la gente ("me sacaron la vesícula", "la cadera").
  Solo cuenta si el turno además tiene marco quirúrgico (`_MARCO`), porque un
  paciente de reemplazo de cadera que contesta "me duele la cadera" está
  respondiendo a la pregunta del dolor, no nombrando su procedimiento.
"""
from __future__ import annotations

import re
import unicodedata

# Los cinco del corpus del reto, con la etiqueta EXACTA de `Patient.surgery` —
# es la que se compara contra la ficha y la que se le dice al paciente en voz
# alta, así que se escribe como se pronuncia.
APENDICECTOMIA = "Apendicectomía"
COLECISTECTOMIA = "Colecistectomía"
COLECTOMIA = "Colectomía"
REEMPLAZO = "Reemplazo de cadera/rodilla"
MASTECTOMIA = "Mastectomía"

# Y los frecuentes que NO están en el corpus. Se detectan precisamente porque no
# están: es el caso en que hay que abstenerse en vez de citar los documentos de
# otra cirugía, que es lo que pasó en la llamada que motivó este módulo.
CESAREA = "Cesárea"
HISTERECTOMIA = "Histerectomía"
HERNIA = "Hernia"
PROSTATA = "Próstata"

# Marco quirúrgico: el turno está hablando de la operación, no de un síntoma.
# Incluye las formas de CORRECCIÓN ("en realidad fue", "perdón, fue"), que son
# justo las que importan aquí — el paciente enmendando lo que el sistema creía.
_MARCO = re.compile(
    r"me\s+(operaron|opere|oper[oa]ron|hicieron|practicaron|sacaron|quitaron"
    r"|extirparon|pusieron|cambiaron|removieron|intervinieron)"
    r"|\b(cirugia|operacion|intervencion)\b"
    r"|lo\s+mio\s+fue|a\s+mi\s+me\s+(operaron|sacaron|quitaron)"
    r"|(en\s+realidad|perdon|disculpe|no),?\s+(fue|era|me\s+operaron)"
    r"|fue\s+(una?|de)\b|era\s+(una?|de)\b"
)

# (etiqueta canónica, nombre clínico, forma coloquial). `None` en coloquial =
# no hay forma coloquial que no sea ya ambigua con un síntoma.
_TABLA: list[tuple[str, re.Pattern[str], re.Pattern[str] | None]] = [
    (
        CESAREA,
        re.compile(r"cesarea|cesaria"),
        # "me operaron para tener al bebé" / "del parto": el marco lo aporta la
        # cirugía y el complemento lo desambigua.
        re.compile(r"\bparto\b|nacio\s+(el|la)\s+(bebe|nina|nino)|tener\s+(al|el)\s+bebe"),
    ),
    (
        APENDICECTOMIA,
        re.compile(r"apendicectomia|apendicitis|apendice"),
        None,
    ),
    (
        COLECISTECTOMIA,
        re.compile(r"colecistectomia|colecistitis"),
        re.compile(r"vesicula|calculos?\s+(en\s+)?(la\s+)?(vesicula|hiel)|piedras?\s+en\s+la\s+hiel"),
    ),
    (
        COLECTOMIA,
        re.compile(r"colectomia|colostomia"),
        re.compile(r"\bcolon\b|\bintestino\b|bolsa\s+de\s+colostomia"),
    ),
    (
        REEMPLAZO,
        re.compile(r"artroplastia|reemplazo\s+(total\s+)?de\s+(cadera|rodilla)"
                   r"|protesis\s+de\s+(cadera|rodilla)"),
        re.compile(r"\bcadera\b|\brodilla\b"),
    ),
    (
        MASTECTOMIA,
        re.compile(r"mastectomia"),
        re.compile(r"\bseno\b|\bsenos\b|\bmama\b|\bpecho\s+izquierdo|\bpecho\s+derecho"),
    ),
    (
        HISTERECTOMIA,
        re.compile(r"histerectomia"),
        re.compile(r"\bmatriz\b|\butero\b|me\s+sacaron\s+todo\s+por\s+dentro"),
    ),
    (
        HERNIA,
        re.compile(r"herniorrafia|hernioplastia"),
        re.compile(r"\bhernia\b"),
    ),
    (
        PROSTATA,
        re.compile(r"prostatectomia"),
        re.compile(r"\bprostata\b"),
    ),
]


def _norm(s: str) -> str:
    """Minúsculas sin tildes. Mismo criterio que `rag/retrieve.py::_norm`."""
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    cleaned = "".join(c.lower() if c.isalnum() else " " for c in s)
    return " ".join(cleaned.split())


def detectar(texto_normalizado: str) -> str | None:
    """Qué procedimiento nombra el paciente, o None.

    Recibe el texto YA normalizado por `lexicon.normalize` (minúsculas, sin
    tildes), igual que `otros_sintomas.detectar`.

    Devuelve UNO solo: el primero de la tabla que case. En un turno que nombra
    dos ("no fue apendicitis, fue cesárea") la corrección se resuelve arriba, en
    el orquestador, comparando contra la ficha; aquí se prefiere devolver algo
    reconocible a intentar adivinar cuál de los dos era el bueno.
    """
    hay_marco = bool(_MARCO.search(texto_normalizado))
    for etiqueta, nombre, coloquial in _TABLA:
        if nombre.search(texto_normalizado):
            return etiqueta
        if hay_marco and coloquial is not None and coloquial.search(texto_normalizado):
            return etiqueta
    return None


def canonico(procedimiento: str | None) -> str | None:
    """La etiqueta canónica de un procedimiento de ficha, o None si es desconocido.

    Pasa el valor de `Patient.surgery` por el mismo vocabulario que la voz del
    paciente. Sin esto, comparar "Reemplazo de rodilla" (ficha) con
    "Reemplazo de cadera/rodilla" (detectado) daría discrepancia por una
    diferencia de redacción, y el agente acusaría de un error que no existe.
    """
    if not procedimiento:
        return None
    t = _norm(procedimiento)
    for etiqueta, nombre, coloquial in _TABLA:
        if nombre.search(t) or (coloquial is not None and coloquial.search(t)):
            return etiqueta
    return None


def coincide(dicho: str, registrado: str | None) -> bool:
    """¿Lo que dice el paciente concuerda con la ficha?

    Ante la duda, **sí**: si la ficha trae un procedimiento que este módulo no
    sabe canonizar, no se puede afirmar que haya discrepancia, y acusar de una
    inexistente sería peor que callar. La alerta que sale de aquí le cuesta a
    alguien revisar un registro clínico.
    """
    canon = canonico(registrado)
    return canon is None or canon == dicho


def etiquetas() -> list[str]:
    """Todas las etiquetas posibles. La usa el TTS para pre-sintetizar."""
    return [e for e, _, _ in _TABLA]


def en_corpus(etiqueta: str) -> bool:
    """¿Hay documentos indexados para este procedimiento?

    Los cinco del reto tienen corpus; el resto se detecta justamente para poder
    declarar el límite en vez de responder con los PDFs de otra cirugía.
    """
    return etiqueta in (APENDICECTOMIA, COLECISTECTOMIA, COLECTOMIA,
                        REEMPLAZO, MASTECTOMIA)

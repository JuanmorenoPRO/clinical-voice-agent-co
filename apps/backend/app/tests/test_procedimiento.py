"""De qué dice el paciente que lo operaron — detección determinista.

El agujero que cierra este módulo: el procedimiento entraba solo por
`Patient.surgery` y nunca por la conversación, así que "me operaron de cesárea"
se caía al vacío y el agente contestaba con la repregunta de la fiebre.

El riesgo de un detector así es el falso positivo, y por eso la mitad de este
archivo son negativos: un paciente de reemplazo de cadera que contesta "me duele
la cadera" está respondiendo a la pregunta del dolor, no corrigiendo su ficha, y
acusarle una discrepancia le cuesta a alguien revisar un registro clínico.

Corre sin Ollama y sin red.
"""

from __future__ import annotations

import pytest

from app.nlu import lexicon, procedimiento


def _detectar(texto: str) -> str | None:
    return procedimiento.detectar(lexicon.normalize(texto))


@pytest.mark.parametrize(
    "texto,esperado",
    [
        # El caso que motivó todo esto.
        ("me operaron de cesárea", procedimiento.CESAREA),
        # Formas de CORRECCIÓN: el paciente enmendando lo que el sistema creía.
        ("Perdón, fue una cesárea", procedimiento.CESAREA),
        ("En realidad fue una cesárea, no una apendicectomía", procedimiento.CESAREA),
        # Coloquiales: necesitan marco quirúrgico, y aquí lo tienen.
        ("A mí me sacaron la vesícula", procedimiento.COLECISTECTOMIA),
        ("Me operaron de la rodilla", procedimiento.REEMPLAZO),
        ("Me pusieron una prótesis de cadera", procedimiento.REEMPLAZO),
        ("No, a mí me operaron del colon", procedimiento.COLECTOMIA),
        # Nombres clínicos: cuentan solos, sin marco.
        ("Lo mío fue de apendicitis", procedimiento.APENDICECTOMIA),
        ("Me hicieron una mastectomía", procedimiento.MASTECTOMIA),
        ("Me operaron de una hernia", procedimiento.HERNIA),
        ("Fue una histerectomía", procedimiento.HISTERECTOMIA),
    ],
)
def test_detecta_el_procedimiento_que_nombra_el_paciente(texto, esperado):
    assert _detectar(texto) == esperado


@pytest.mark.parametrize(
    "texto",
    [
        # Sin marco quirúrgico, la parte del cuerpo es un SÍNTOMA, no la cirugía.
        # Este es el falso positivo que arruinaría el detector: un paciente de
        # reemplazo de cadera contesta esto a la pregunta del dolor.
        "Me duele la cadera",
        "Me duele el seno izquierdo",
        "Tengo un dolor en el intestino",
        # Hablar de la operación sin nombrarla no es nombrarla.
        "La operación salió bien",
        "El médico me dijo que estaba bien",
        "Me operaron hace ocho días",
        # Y las respuestas normales del guion no pueden disparar nada.
        "Me duele un 8",
        "La herida se ve rojita",
        "No he tenido fiebre",
        "Me cuesta respirar",
    ],
)
def test_no_confunde_un_sintoma_con_el_procedimiento(texto):
    assert _detectar(texto) is None


def test_coincide_compara_por_vocabulario_no_por_texto():
    """La ficha y la voz del paciente pasan por el mismo vocabulario.

    Sin `canonico()`, comparar "Reemplazo de rodilla" (ficha) con
    "Reemplazo de cadera/rodilla" (detectado) daría discrepancia por una
    diferencia de redacción, y el agente acusaría de un error inexistente.
    """
    assert procedimiento.coincide(procedimiento.REEMPLAZO, "Reemplazo de rodilla")
    assert procedimiento.coincide(procedimiento.APENDICECTOMIA, "Apendicectomía")
    assert not procedimiento.coincide(procedimiento.CESAREA, "Apendicectomía")


def test_ante_la_duda_no_se_acusa_discrepancia():
    """Si la ficha trae algo que este módulo no sabe canonizar, no hay discrepancia.

    La alerta que sale de aquí le cuesta a alguien revisar un registro clínico:
    afirmar un error que no se puede sostener es peor que callar.
    """
    assert procedimiento.coincide(procedimiento.CESAREA, "Cirugía rarísima")
    assert procedimiento.coincide(procedimiento.CESAREA, None)


def test_solo_cinco_procedimientos_tienen_corpus():
    """Cesárea y compañía se detectan JUSTAMENTE porque no están indexadas.

    Es lo que permite declarar el límite en vez de responder con los PDFs de
    otra cirugía, que fue el fallo original.
    """
    assert procedimiento.en_corpus(procedimiento.APENDICECTOMIA)
    assert not procedimiento.en_corpus(procedimiento.CESAREA)
    assert len([e for e in procedimiento.etiquetas()
                if procedimiento.en_corpus(e)]) == 5

"""Un sí/no pelado significa lo que diga la PREGUNTA, no el slot.

`nlu/polaridad.py` duplica las cadenas de `agent/phrasing.py` porque `nlu` no
puede importar `agent` sin cerrar un ciclo. Este archivo es lo que hace segura esa
duplicación: si alguien reescribe una pregunta del guion y no toca la tabla, el
test falla en vez de que la polaridad deje de aplicarse en silencio.
"""
from __future__ import annotations

import pytest

from app.agent import phrasing
from app.nlu import lexicon, polaridad


def _todas_las_preguntas() -> list[str]:
    preguntas: list[str] = []
    for banco in (phrasing.PREGUNTAS, phrasing.REPREGUNTAS, phrasing.SEGUIMIENTOS):
        for opciones in banco.values():
            preguntas.extend(opciones)
    return preguntas


@pytest.mark.parametrize("pregunta", _todas_las_preguntas())
def test_toda_pregunta_del_guion_esta_clasificada(pregunta):
    """O tiene polaridad, o está declarada como que no la tiene. Nunca olvidada.

    La lista `SIN_POLARIDAD` no es un vertedero: obliga a que añadir una pregunta
    sea una decisión consciente sobre qué significa contestarle "no".
    """
    assert (pregunta in polaridad.POLARIDAD) != (pregunta in polaridad.SIN_POLARIDAD), (
        f"{pregunta!r} no está ni en POLARIDAD ni en SIN_POLARIDAD (o está en las dos)"
    )


def test_la_tabla_no_tiene_entradas_huerfanas():
    """Al revés: ninguna entrada apunta a una pregunta que ya no se hace."""
    vivas = set(_todas_las_preguntas())
    huerfanas = (set(polaridad.POLARIDAD) | set(polaridad.SIN_POLARIDAD)) - vivas
    assert not huerfanas, f"preguntas que ya nadie hace: {sorted(huerfanas)}"


# --- la polaridad se invierte dentro del mismo slot ---------------------------
# Es la razón de ser del módulo: con la tabla indexada por slot, una de las dos
# preguntas de abajo tendría que estar mal por fuerza.

def test_el_mismo_no_significa_lo_contrario_segun_la_pregunta():
    dificultad = lexicon.extract(
        "No", slot="movilidad",
        question="¿Cómo se siente al moverse o caminar? ¿Ha tenido alguna dificultad?")
    sin_problema = lexicon.extract(
        "No", slot="movilidad",
        question="¿Ha podido levantarse y caminar sin problema estos días?")

    assert dificultad.mobility == "normal"
    assert sin_problema.mobility == "limitada_esperada"


def test_el_no_del_termometro_no_es_una_negacion_de_fiebre():
    """El fallo del turno 2 de la llamada reportada (conversación 0cf3f8d3).

    El agente preguntó por el TERMÓMETRO y el paciente dijo "No.". Una regla
    anclada al slot —que sigue siendo `fiebre` durante el seguimiento— lo leyó
    como que negaba la fiebre.
    """
    s = lexicon.extract("No", slot="fiebre",
                        question="¿Ha podido medírsela con termómetro?")

    assert s.temperature_measured is False
    assert s.fever is None, "el paciente habló del termómetro, no del síntoma"


def test_la_oferta_de_enfermera_no_contesta_por_ningun_slot():
    """Durante la oferta, `slot_actual` sigue siendo el del tamizaje.

    Es el turno 8 de la llamada reportada: "No." a "¿prefiere que le pida a una
    enfermera que lo llame?" no dice nada de la herida. `pregunta_emitida`
    devuelve None para esa acción y por eso aquí no hay polaridad que aplicar.
    """
    oferta = phrasing.OFRECER_SALIDA[0]
    assert polaridad.de(oferta) is None
    assert lexicon.extract("No", slot="herida", question=oferta).wound is None
    assert phrasing.pregunta_emitida("ofrecer_salida", "herida", 0, None,
                                     semilla="x") is None


def test_un_turno_contradictorio_no_se_resuelve_hacia_el_hallazgo():
    """"No, yo estoy bien." a "¿ha podido caminar sin problema?" (turno 3 real).

    El "no" apunta a limitación y el "estoy bien" a normalidad. Anotar el
    hallazgo contra lo que el paciente acaba de afirmar de sí mismo sería peor
    que repreguntar.
    """
    s = lexicon.extract("No, yo estoy bien.", slot="movilidad",
                        question="¿Ha podido levantarse y caminar sin problema estos días?")
    assert s.mobility is None

    # Pero una negación de verdad —el "no" pegado al verbo— sí resuelve.
    s = lexicon.extract("No, no estoy bien", slot="movilidad",
                        question="¿Ha podido levantarse y caminar sin problema estos días?")
    assert s.mobility == "limitada_esperada"


def test_la_pregunta_llega_dentro_de_la_respuesta_entera():
    """`de()` acepta el turno completo: el acuse y el reflejo van delante."""
    entero = "Un 5, ahí en la mitad. ¿Ha tenido fiebre o calentura estos días?"
    assert polaridad.de(entero) == ("fever", True, False)
    assert polaridad.de(None) is None
    assert polaridad.de("una frase cualquiera") is None

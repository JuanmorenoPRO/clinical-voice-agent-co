"""La calibración de `nlu/intent.py`, medida sobre el corpus en vez de recordada.

El módulo lleva desde el principio afirmando estar "calibrado contra los turnos
reales del dataset" y citando un 42% de falsos positivos de una versión más laxa,
pero eso vivía sólo en comentarios: ningún test ni script pasaba `classify` sobre
`tests/dataset/dialogs.jsonl`. Lo único que protegía la calibración eran cuatro
cadenas copiadas a mano en `test_nlu.py`.

El precio de no medir fue este agujero: como Whisper no devuelve `¿?` y la primera
alternativa de `_PREGUNTA` es literalmente `[¿?]`, DOS DE CADA TRES preguntas del
paciente caían al default `respuesta` en cuanto la transcripción llegaba pelada —
y ahí el RAG no se consulta nunca. Nadie podía verlo porque nadie lo contaba.

Los dos tests de aquí son trinquetes con margen, no igualdades: el corpus se
regenera desde los `.xlsx` del kit oficial (`scripts/load_dataset.py`), así que
clavar el número exacto haría fallar la suite por un cambio de dataset. Lo que
tienen que atrapar es la tendencia — que la clasificación no se afloje hasta
disparar el RAG en cada turno, y que las preguntas sin signos no vuelvan a
perderse.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.nlu.intent import classify

DIALOGOS = Path(__file__).parents[4] / "tests" / "dataset" / "dialogs.jsonl"


@pytest.fixture(scope="module")
def turnos() -> list[str]:
    """Todo lo que dice quien NO es el agente, en las dos capas del dataset.

    Se incluye al cuidador (`tercero`) además del paciente: habla por el mismo
    canal, lo clasifica el mismo código y es el corte de 2.071 turnos que citan
    `intent.py` y `docs/spikes-7-agosto.md`.
    """
    if not DIALOGOS.exists():  # pragma: no cover — sin dataset no hay nada que medir
        pytest.skip(f"falta el corpus: {DIALOGOS} (regenerar con scripts/load_dataset.py)")
    textos: list[str] = []
    for linea in DIALOGOS.read_text(encoding="utf-8").splitlines():
        if not linea.strip():
            continue
        for t in json.loads(linea)["turnos"]:
            if t["hablante"] != "agente" and t["texto"].strip():
                textos.append(t["texto"])
    assert len(textos) > 1500, f"el corpus encogió a {len(textos)} turnos"
    return textos


def _sin_signos(texto: str) -> str:
    return texto.replace("¿", "").replace("?", "")


def test_la_distribucion_de_intenciones_no_se_movio(turnos):
    """El guardián del 42%: la versión laxa de `_PREGUNTA` marcaba casi la mitad
    de los turnos como consulta, y eso habría disparado el RAG en cada turno.

    Hoy la tasa es del 31,9%. El tope es 40%: deja sitio para ensanchar la
    detección a propósito —que es justo lo que se acaba de hacer— y sigue muy por
    debajo del régimen que la calibración original descartó.
    """
    preguntas = sum(1 for t in turnos if classify(t) == "pregunta_clinica")
    tasa = preguntas / len(turnos)
    assert tasa < 0.40, (
        f"pregunta_clinica subió al {100 * tasa:.1f}% de {len(turnos)} turnos; "
        "la versión laxa que se descartó marcaba el 42%"
    )


def test_la_pregunta_sobrevive_sin_signos_de_interrogacion(turnos):
    """La queja que originó todo esto, como número.

    Se le quitan los `¿?` a cada turno —que es exactamente lo que hace Whisper— y
    se cuentan las preguntas clínicas que se degradan a `respuesta`. Cada una es
    un turno en el que el paciente pregunta algo y el agente sigue con el guion
    sin consultar el RAG.

    Eran 433 de 2.071. Con las seis familias de `_pregunta_sin_signos` son 286
    (297 con cuatro; la familia E —la duda anunciada— recuperó seis más, y la H
    —la duda indirecta con "si"— otras cinco). El tope de 310 da margen al corpus
    sin dejar que la cifra vuelva a subir; las 286 restantes son las familias que
    quedaron fuera a propósito (ver el comentario de `_PREGUNTA_SIN_SIGNOS`).
    """
    perdidas = [
        t for t in turnos
        if classify(t) == "pregunta_clinica"
        and classify(_sin_signos(t)) != "pregunta_clinica"
    ]
    assert len(perdidas) <= 310, (
        f"{len(perdidas)} preguntas se pierden al quitar los signos (eran 297); "
        f"ejemplo: {perdidas[0]!r}"
    )


def test_quitar_los_signos_no_inventa_preguntas(turnos):
    """La otra cara: ensanchar la detección no puede convertir respuestas en
    consultas. Cada falso positivo aquí es una llamada al RAG y una abstención
    inútil en mitad del tamizaje.

    El suelo no es cero, y la razón es anterior a `_pregunta_sin_signos`: son 10
    turnos donde la coletilla ES el signo. "eso es normal después de una
    operación, ¿no?" se clasifica como respuesta porque `_MULETILLA` descarta el
    fragmento "¿no?"; al quitar los signos no queda fragmento que descartar y
    `_PREGUNTA` engancha con el resto de la frase. Medido antes y después de cada
    cambio: 10 y 10 — ni las cuatro familias originales, ni la quinta (la duda
    anunciada), ni la sexta (la duda indirecta con "si") añaden uno solo.
    """
    inventadas = [
        t for t in turnos
        if classify(t) == "respuesta"
        and classify(_sin_signos(t)) == "pregunta_clinica"
    ]
    assert len(inventadas) <= 10, (
        f"{len(inventadas)} respuestas pasaron a pregunta al quitar los signos "
        f"(eran 10, todas por la coletilla); ejemplo: {inventadas[10]!r}"
    )

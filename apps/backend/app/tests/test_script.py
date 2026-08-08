"""La máquina de estados del guion — funciones puras, sin LLM ni base de datos.

El caso central de este archivo es el que reportó un usuario probando la voz:
tras escalar por un dolor severo, si el paciente seguía hablando, el agente
repetía el guion de seguridad completo palabra por palabra, indefinidamente. La
causa era que `next_action` solo miraba `escalar` (que se queda en True para
siempre, porque el cuadro crítico no se "des-escala" dentro de la llamada) y
nunca miraba si el guion ya se había entregado en un turno anterior.
"""
from __future__ import annotations

from app.agent.script import Action, CallState, Phase, apply, next_action
from app.schemas import Symptoms

CRITICO = Symptoms(pain_level=9)  # dispara escalar=True vía el motor de decisión


def test_primer_turno_critico_escala():
    estado = CallState()
    a = next_action(estado, CRITICO, escalar=True)
    assert a.kind == "escalar"
    assert a.phase is Phase.ESCALAMIENTO


def test_segundo_turno_critico_no_repite_el_guion():
    """El bug reportado: sin este corte, esto seguiría devolviendo 'escalar'."""
    tras_escalar = apply(CallState(), Action(kind="escalar", phase=Phase.ESCALAMIENTO), CRITICO)
    assert tras_escalar.phase is Phase.ESCALAMIENTO

    a = next_action(tras_escalar, CRITICO, escalar=True)
    assert a.kind == "cerrar"
    assert a.phase is Phase.TERMINADA


def test_tercer_turno_sigue_cerrado_no_revive_el_guion():
    """Una vez en TERMINADA, se queda ahí aunque el paciente siga hablando."""
    cerrada = CallState(phase=Phase.TERMINADA)
    a = next_action(cerrada, CRITICO, escalar=True)
    assert a.kind == "cerrar"
    assert a.phase is Phase.TERMINADA


def test_una_llamada_no_critica_nunca_pasa_por_escalamiento():
    estado = CallState()
    a = next_action(estado, Symptoms(pain_level=2), escalar=False)
    assert a.kind != "escalar"

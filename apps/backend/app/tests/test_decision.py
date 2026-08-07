"""Pruebas del Motor de Decisión — apuntan directo a los 15 pts de "lógica de
decisión" de la rúbrica. Corren sin BD, sin LLM y sin red (funciones puras).
"""
from __future__ import annotations

from app.decision import engine
from app.schemas import Symptoms


def test_normal_sin_sintomas():
    result = engine.evaluate(Symptoms())
    assert result.risk_level == "NORMAL"
    assert result.triggered_rules == []
    assert result.safety_script is None


def test_dolor_severo_escala_solo_por_intensidad():
    """Dolor >= 8 es rojo por sí mismo, sin necesidad de saber si la medicación sirve.

    Antes hacía falta `medication_effective is False` para escalar un 9. Con la
    calibración contra las 160 trayectorias eso era un falso negativo esperando:
    el paciente que no menciona la analgesia se iba como verde.
    """
    result = engine.evaluate(Symptoms(pain_level=9))
    assert result.risk_level == "CRÍTICO"
    assert result.triage_color == "rojo"
    assert "dolor_severo" in result.triggered_rules


def test_dolor_severo_escala_aunque_la_medicacion_funcione():
    result = engine.evaluate(Symptoms(pain_level=9, medication_effective=True))
    assert result.risk_level == "CRÍTICO"


def test_dolor_no_controlado_cubre_el_tramo_por_debajo_del_rojo():
    """Un 8 con medicación inefectiva dispara ambas reglas; un 8 solo, la de rojo."""
    result = engine.evaluate(Symptoms(pain_level=8, medication_effective=False))
    assert {"dolor_severo", "dolor_no_controlado"} <= set(result.triggered_rules)


def test_fiebre_alta():
    result = engine.evaluate(Symptoms(temperature_c=39.0))
    assert result.risk_level == "CRÍTICO"
    assert result.triage_color == "rojo"
    assert "fiebre_38" in result.triggered_rules
    assert result.escalation_action == "enfermeria_prioritaria"


def test_umbral_de_fiebre_calibrado_en_38():
    """37.9 no es rojo, 38.0 sí. El corte sale del ground truth, no de la intuición."""
    assert engine.evaluate(Symptoms(temperature_c=37.9)).triage_color != "rojo"
    assert engine.evaluate(Symptoms(temperature_c=38.0)).triage_color == "rojo"


def test_slots_del_dataset_escalan_a_rojo():
    assert engine.evaluate(Symptoms(wound="secrecion_purulenta")).triage_color == "rojo"
    assert engine.evaluate(Symptoms(mobility="incapacitante_nueva")).triage_color == "rojo"


def test_dos_senales_leves_son_amarillo():
    """Ninguna señal por separado escala; dos juntas sí."""
    solo_eritema = engine.evaluate(Symptoms(wound="eritema_leve"))
    assert solo_eritema.risk_level == "NORMAL"
    dos = engine.evaluate(Symptoms(wound="eritema_leve", appetite="muy_disminuido"))
    assert dos.risk_level == "ALTO"
    assert dos.triage_color == "amarillo"
    assert "vigilancia_multiples_signos" in dos.triggered_rules


def test_la_ausencia_de_dato_nunca_baja_el_riesgo():
    """El paciente evasivo no puede convertir un cuadro en verde por no contestar."""
    con_dato = engine.evaluate(Symptoms(temperature_c=38.5, wound="eritema_leve"))
    sin_dato = engine.evaluate(Symptoms(temperature_c=38.5))
    assert con_dato.triage_color == sin_dato.triage_color == "rojo"


def test_incertidumbre_al_cierre_tiene_dos_grados():
    """No es lo mismo una llamada a medias que una donde no se averiguó nada.

    Con la mitad del guion respondido se puede pedir seguimiento. Con uno de seis
    slots no se puede descartar una emergencia, y dejarlo en seguimiento fue un
    falso negativo real sobre el dataset (ver reports/dataset-eval.md).
    """
    vacia = Symptoms()
    assert engine.evaluate(vacia).risk_level == "NORMAL"       # durante la llamada
    nada = engine.evaluate(vacia, final=True)                   # al cerrarla
    assert nada.risk_level == "CRÍTICO"
    assert "no_se_pudo_evaluar" in nada.triggered_rules
    assert nada.escalation_action == "enfermeria_prioritaria"
    assert nada.safety_script and "enfermería" in nada.safety_script

    # A medias: suficiente para pedir seguimiento, no para alarmar.
    parcial = engine.evaluate(
        Symptoms(pain_level=2, fever=False, mobility="normal", wound="eritema_leve"),
        final=True)
    assert parcial.risk_level == "ALTO"
    assert "informacion_insuficiente" in parcial.triggered_rules


def test_fiebre_referida_sin_termometro_escala():
    """El paciente dice tener fiebre pero no se la midió.

    Exigir la cifra dejaba escapar casos rojos reales: el reto avisa de que muchos
    pacientes no tienen ni termómetro.
    """
    d = engine.evaluate(
        Symptoms(fever=True, appetite="muy_disminuido", sleep="muy_alterado"))
    assert d.risk_level == "CRÍTICO"
    assert "fiebre_referida_con_signos" in d.triggered_rules


def test_fiebre_referida_sola_no_llega_a_rojo():
    """Sin otras señales acompañándola, no es urgencia."""
    assert engine.evaluate(
        Symptoms(fever=True, appetite="normal", sleep="normal")).risk_level != "CRÍTICO"


def test_incertidumbre_no_degrada_un_cuadro_que_ya_escalo():
    d = engine.evaluate(Symptoms(temperature_c=39.0), final=True)
    assert d.risk_level == "CRÍTICO"
    assert "informacion_insuficiente" not in d.triggered_rules


def test_sangrado_abundante_es_critico():
    result = engine.evaluate(Symptoms(heavy_bleeding=True))
    assert result.risk_level == "CRÍTICO"
    assert result.safety_script is not None
    assert "enfermería" in result.safety_script.lower()


def test_dificultad_respiratoria_es_critico():
    result = engine.evaluate(Symptoms(breathing_difficulty=True))
    assert result.risk_level == "CRÍTICO"


def test_perdida_consciencia_es_critico():
    result = engine.evaluate(Symptoms(loss_of_consciousness=True))
    assert result.risk_level == "CRÍTICO"


def test_critico_gana_sobre_alto():
    # Varias reglas disparadas; el nivel debe ser el más severo.
    result = engine.evaluate(
        Symptoms(temperature_c=39.5, heavy_bleeding=True)
    )
    assert result.risk_level == "CRÍTICO"
    assert set(result.triggered_rules) >= {"fiebre_38", "sangrado_abundante"}
    # La emergencia vital manda sobre la vía de enfermería aunque ambas sean CRÍTICO.
    assert result.escalation_action == "emergencia_123"
    assert "123" in (result.safety_script or "")


def test_dolor_toracico_es_critico():
    result = engine.evaluate(Symptoms(chest_pain=True))
    assert result.risk_level == "CRÍTICO"
    assert "dolor_toracico" in result.triggered_rules
    assert result.safety_script is not None
    assert "123" in result.safety_script


def test_estado_mental_alterado_es_critico():
    result = engine.evaluate(Symptoms(altered_mental_status=True))
    assert result.risk_level == "CRÍTICO"
    assert "estado_mental_alterado" in result.triggered_rules


def test_convulsion_es_critico():
    result = engine.evaluate(Symptoms(seizure=True))
    assert result.risk_level == "CRÍTICO"
    assert "convulsion" in result.triggered_rules


def test_mock_extrae_emergencias_nuevas_y_escala():
    """El extractor mock detecta las nuevas emergencias y el motor las escala."""
    from app.llm.mock import extract_symptoms_heuristic

    for frase, campo in [
        ("Tengo un dolor fuerte en el pecho", "chest_pain"),
        ("Estoy muy confundido, no sé dónde estoy", "altered_mental_status"),
        ("Acabo de tener una convulsión", "seizure"),
    ]:
        sym = extract_symptoms_heuristic(frase)
        assert getattr(sym, campo) is True, frase
        assert engine.evaluate(sym).risk_level == "CRÍTICO", frase

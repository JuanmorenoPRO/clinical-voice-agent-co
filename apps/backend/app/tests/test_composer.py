"""La aduana del redactor LLM — determinista, sin modelo ni red.

`composer.valida` es lo que separa "el LLM mejora el fraseo" de "el LLM puede
decir cualquier cosa": todo lo que se rechaza aquí cae a las plantillas, así
que las guardas se prueban en ambos sentidos — que atajen lo peligroso y que
NO se coman una composición legítima (rechazar de más devuelve el agente
robótico que motivó todo este cambio).
"""
from __future__ import annotations

from types import SimpleNamespace

from app.agent import composer
from app.llm.adapter import ComposeContext
from app.schemas import Symptoms

PREGUNTA = "¿Ha tenido fiebre o calentura estos días?"


def ctx(**kw) -> ComposeContext:
    base = dict(
        historial=[], utterance="me duele poquito", intent="respuesta",
        pregunta_guion=PREGUNTA, objetivo="preguntar", evidencia=None,
        sintomas="dolor 3/10", procedimiento="reemplazo de cadera",
        nombre="Mauricio", riesgo="NORMAL", notas=[], alarma=False,
    )
    base.update(kw)
    return ComposeContext(**base)


# --- lo legítimo pasa ---------------------------------------------------------

def test_una_composicion_normal_pasa():
    texto = f"Un 3, bastante llevadero. {PREGUNTA}"
    assert composer.valida(texto, ctx()) is True


def test_la_cifra_de_la_evidencia_se_permite():
    texto = f"Según la guía, puede ducharse desde las 48 horas. {PREGUNTA}"
    ok = ctx(evidencia="El baño diario se permite desde las 48 horas.",
             intent="pregunta_clinica")
    assert composer.valida(texto, ok) is True


def test_es_normal_anclado_con_riesgo_normal_se_permite():
    # Con el cuadro en NORMAL y sin alarma consultada, un "es normal" que viene
    # de la evidencia es la respuesta correcta — recortarlo era lo que dejaba
    # frases a medias ("...lo que sugiere que").
    texto = f"Es normal sentir el músculo débil las primeras semanas. {PREGUNTA}"
    ok = ctx(evidencia="Es normal sentir debilidad muscular las primeras semanas.",
             intent="pregunta_clinica")
    assert composer.valida(texto, ok) is True


def test_el_cierre_sin_pregunta_pasa():
    cierre = "Listo, quedó todo registrado. Si algo empeora, llame al hospital. Que siga bien."
    assert composer.valida(cierre, ctx(objetivo="cerrar", pregunta_guion=cierre)) is True


# --- lo peligroso se rechaza ---------------------------------------------------

def test_sin_la_pregunta_del_guion_se_rechaza():
    assert composer.valida("Un 3, bastante llevadero.", ctx()) is False


def test_el_cierre_que_reabre_con_pregunta_se_rechaza():
    assert composer.valida(
        "Listo, quedó registrado. ¿De acuerdo?",
        ctx(objetivo="cerrar", pregunta_guion="Listo, que siga bien."),
    ) is False


def test_la_cifra_inventada_se_rechaza():
    texto = f"El termómetro debería marcar 36.5 en su caso. {PREGUNTA}"
    assert composer.valida(texto, ctx()) is False


def test_el_plazo_inventado_se_rechaza_aunque_el_numero_sea_de_la_escala():
    # "6" suelto es un dolor legítimo (escala 0-10); "6 semanas" es una
    # instrucción clínica que tiene que venir de la evidencia.
    texto = f"Puede volver al gimnasio a las 6 semanas. {PREGUNTA}"
    assert composer.valida(texto, ctx(intent="pregunta_clinica")) is False


def test_el_veredicto_con_riesgo_elevado_se_rechaza():
    texto = f"Eso es normal, no se preocupe. {PREGUNTA}"
    assert composer.valida(texto, ctx(riesgo="ALTO")) is False


def test_el_veredicto_con_alarma_consultada_se_rechaza():
    texto = f"Eso no es normal, debería revisarse. {PREGUNTA}"
    assert composer.valida(texto, ctx(alarma=True)) is False


def test_la_emocion_atribuida_se_rechaza():
    # Medido en una llamada real: "Usted parece un poco ansioso por regresar a
    # sus actividades deportivas favoritas..." — el paciente nunca dijo eso.
    texto = f"Usted parece preocupado por volver al fútbol. {PREGUNTA}"
    assert composer.valida(texto, ctx()) is False


def test_el_parrafo_interminable_se_rechaza():
    texto = ("Le explico con detalle. " * 12) + PREGUNTA
    assert composer.valida(texto, ctx()) is False


def test_la_salida_vacia_se_rechaza():
    assert composer.valida("", ctx()) is False
    assert composer.valida("   ", ctx()) is False


# --- build_context -------------------------------------------------------------

def _turno(paciente: str, agente: str) -> SimpleNamespace:
    return SimpleNamespace(patient_utterance=paciente, final_response=agente)


def test_el_contexto_lleva_el_historial_en_orden_y_acotado():
    prior = [_turno(f"p{i}", f"a{i}") for i in range(12)]
    c = composer.build_context(
        prior=prior, text="un 4", intent="respuesta", objetivo="preguntar",
        pregunta_guion=PREGUNTA, acumulado=Symptoms(pain_level=4),
        riesgo="NORMAL", evidencia=None, procedimiento="cadera", nombre=None,
    )
    # Ventana de 8 turnos = 16 entradas alternadas, del más antiguo al más nuevo.
    assert len(c.historial) == 16
    assert c.historial[0] == {"rol": "paciente", "texto": "p4"}
    assert c.historial[-1] == {"rol": "agente", "texto": "a11"}
    assert "dolor 4/10" in c.sintomas


def test_la_pregunta_sin_evidencia_genera_la_nota_de_abstencion():
    c = composer.build_context(
        prior=[], text="¿cuándo puedo volver a jugar fútbol?",
        intent="pregunta_clinica", objetivo="confirmar",
        pregunta_guion="¿Quedamos así, o quiere preguntarme algo más?",
        acumulado=Symptoms(), riesgo="NORMAL", evidencia=None,
        procedimiento="reemplazo de cadera", nombre=None,
    )
    assert any("NO hay evidencia" in n for n in c.notas)


def test_la_despedida_mezclada_genera_su_nota():
    c = composer.build_context(
        prior=[], text="eso es todo, pero ¿cuándo me quitan los puntos?",
        intent="pregunta_clinica", objetivo="confirmar",
        pregunta_guion="¿Quedamos así, o quiere preguntarme algo más?",
        acumulado=Symptoms(), riesgo="NORMAL", evidencia="Los puntos se retiran según indicación.",
        procedimiento="reemplazo de cadera", nombre=None,
    )
    assert any("despidiendo" in n for n in c.notas)


def test_el_cierre_genera_su_nota_y_la_alarma_endurece():
    c = composer.build_context(
        prior=[], text="¿y esta visión borrosa es normal?",
        intent="pregunta_clinica", objetivo="cerrar",
        pregunta_guion="Listo, que siga bien.",
        acumulado=Symptoms(), riesgo="NORMAL", evidencia=None,
        procedimiento=None, nombre=None,
        alarma_consultada=["visión borrosa"],
    )
    assert c.alarma is True
    assert any("TERMINA" in n for n in c.notas)
    assert any("visión borrosa" in n for n in c.notas)

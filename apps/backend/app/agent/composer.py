"""Contexto y validación de la redacción LLM de un turno.

El guion (`script.py`) decide QUÉ hacer; el redactor (`llm.compose_reply`)
decide CÓMO decirlo. Este módulo es el pegamento y la aduana entre ambos:

  - `build_context()` arma todo lo que el redactor puede ver: historial,
    síntomas acumulados, evidencia RAG y —sobre todo— la pregunta canónica del
    guion con la que DEBE terminar. El modelo no elige qué preguntar.
  - `valida()` es la aduana determinista de la salida: cifras fuera de la
    evidencia, veredictos de gravedad, texto sin la pregunta obligatoria o con
    pinta de inyección se rechazan, y el turno cae a las plantillas de
    `phrasing.py`. El redactor puede mejorar el fraseo; no puede empeorar la
    seguridad.
"""
from __future__ import annotations

import logging
import re

from ..llm.adapter import ComposeContext
from ..models import Turn
from ..nlu import intent as intent_nlu
from ..schemas import Symptoms

log = logging.getLogger(__name__)

# Turnos de historial que ve el redactor. Ocho pares dan de sobra para no
# repetir fórmulas ni preguntas; más solo engorda el prompt.
_VENTANA_HISTORIAL = 8

# Fórmulas con las que un modelo complaciente quita hierro a un síntoma. Ante un
# cuadro que ya tiene señales de alarma, decir esto es la conducta que la rúbrica
# penaliza por su nombre: "tranquilizar al paciente ante un síntoma de alarma".
_TRANQUILIZADOR = re.compile(
    r"no\s+(es|parece|deber[ií]a\s+ser|necesariamente\s+indica)\s+(nada\s+)?"
    r"(grave|serio|preocupante|de\s+cuidado)"
    r"|no\s+(se\s+)?preocupe|no\s+hay\s+(de\s+qu[eé]\s+preocuparse|motivo)"
    r"|es\s+(completamente\s+)?normal|es\s+algo\s+normal|tranquil\w+"
    r"|no\s+pasa\s+nada|nada\s+de\s+qu[eé]\s+preocuparse",
    re.I,
)

# Dictaminar sobre la normalidad de un síntoma, en cualquiera de los dos
# sentidos. Quién decide la gravedad es el motor de decisión; con el riesgo ya
# elevado, un "es normal" del modelo es la conducta que la rúbrica penaliza.
_VEREDICTO_NORMALIDAD = re.compile(
    r"\bes\s+(algo\s+|completamente\s+|totalmente\s+|del\s+todo\s+)?normal\b"
    r"|(eso|esto)\s+(no\s+)?es\s+normal"
    r"|es\s+(algo\s+)?(preocupante|de\s+cuidado|grave)",
    re.I,
)

# Emociones que el modelo le atribuye al paciente sin que él las haya dicho
# ("usted parece preocupado/ansioso..."). Es lo que hace sonar a formulario en
# vez de a alguien que escucha, y el prompt ya lo prohíbe; esto es la versión
# que no depende de que obedezca. Solo formas en tercera persona dirigidas al
# paciente: "estoy preocupada por su herida" (el agente) no casa.
_EMOCION_ATRIBUIDA = re.compile(
    r"\b(usted\s+)?(parece|se\s+nota|se\s+le\s+nota|lo\s+noto|la\s+noto|suena)\s+"
    r"(un\s+poco\s+|algo\s+|muy\s+|bastante\s+)?"
    r"(preocupad|ansios|asustad|angustiad|nervios|estresad|frustrad|impacient)",
    re.I,
)

# La apertura ya sonó: cualquier saludo de arranque a mitad de llamada es el
# modelo repitiendo el historial.
_RESALUDO = re.compile(
    r"^\s*(buenos\s+d[ií]as|buenas\s+(tardes|noches)|hola|al[oó])\b[^.!?]*"
    r"(le\s+llamo|del\s+hospital|seguimiento)",
    re.I,
)

_NUM = re.compile(r"\d+(?:[.,]\d+)?")

# Cifras que las plantillas del propio agente ya dicen en voz alta y que el
# redactor puede usar sin evidencia: la escala de dolor (0–10), los umbrales de
# fiebre que viven en los guiones de seguridad (38/39/40) y la línea de
# emergencias.
_CIFRAS_DEL_GUION = {str(n) for n in range(11)} | {"38", "38.5", "39", "40", "123"}

# Un plazo clínico ("6 semanas", "3 días") es la alucinación más peligrosa del
# redactor, y sus números chocan con la escala de dolor: un "6" suelto es un
# dolor legítimo, un "6 semanas" es una instrucción médica que tiene que venir
# de la evidencia. Se valida aparte de la lista blanca general.
_PLAZO = re.compile(r"(\d+(?:[.,]\d+)?)\s*(?:d[ií]as?|semanas?|meses?|horas?)\b", re.I)

_FRASES = re.compile(r"(?<=[.!?])\s+")


def build_context(
    *,
    prior: list[Turn],
    text: str,
    intent: str,
    objetivo: str,
    pregunta_guion: str,
    acumulado: Symptoms,
    riesgo: str,
    evidencia: str | None,
    procedimiento: str | None,
    nombre: str | None,
    discrepa: bool = False,
    proc_vigente: str | None = None,
    alarma_consultada: list[str] | None = None,
    slot_perdido: bool = False,
) -> ComposeContext:
    historial = [
        d
        for t in prior[-_VENTANA_HISTORIAL:]
        for d in (
            {"rol": "paciente", "texto": t.patient_utterance or ""},
            {"rol": "agente", "texto": t.final_response or ""},
        )
        if d["texto"]
    ]

    notas: list[str] = []
    if intent == "pregunta_clinica" and evidencia is None and not discrepa \
            and not alarma_consultada:
        notas.append(
            "El paciente preguntó algo y NO hay evidencia en los documentos del "
            "hospital: dígaselo con naturalidad, ofrezca pasárselo a enfermería, "
            "y siga con el seguimiento. No responda desde su conocimiento."
        )
    if discrepa:
        notas.append(
            f"El paciente refiere una cirugía ({proc_vigente or 'otra'}) distinta "
            f"a la de su ficha ({procedimiento or 'sin registrar'}): NO responda "
            "su pregunta con documentos; dígale que prefiere que enfermería lo "
            "verifique antes de decirle algo que no le consta."
        )
    if alarma_consultada:
        cuales = ", ".join(alarma_consultada)
        notas.append(
            f"El paciente pregunta por un síntoma que hay que valorar en persona "
            f"({cuales}): NO lo valore ni diga si es normal; dígale que queda "
            "anotado para que enfermería lo revise."
        )
    if intent == "pregunta_administrativa":
        notas.append(
            "La pregunta es un trámite (citas, horarios, EPS, costos): esta "
            "línea no lo puede consultar; indíquele que el conmutador del "
            "hospital se lo resuelve, y retome el seguimiento."
        )
    if intent == "tercero":
        notas.append(
            "Quien habla es un cuidador o familiar, no el paciente: agradézcale "
            "y siga; lo que cuente también sirve."
        )
    if slot_perdido:
        notas.append(
            "El dato anterior no se pudo obtener y el guion sigue adelante: "
            "dígalo brevemente ('lo dejo anotado como que no me lo pudo "
            "precisar') sin insistir ni sonar a reproche."
        )
    if intent_nlu.contiene_despedida(text) and objetivo != "cerrar":
        notas.append(
            "El paciente además se está despidiendo: responda lo que haya "
            "preguntado y confirme el cierre con la pregunta obligatoria."
        )
    if objetivo == "cerrar":
        notas.append(
            "La llamada TERMINA en este turno: después de responder lo "
            "pendiente, despídase con el texto obligatorio. No haga preguntas."
        )

    return ComposeContext(
        historial=historial,
        utterance=text,
        intent=intent,
        pregunta_guion=pregunta_guion,
        objetivo=objetivo,
        evidencia=evidencia,
        sintomas=resumen_sintomas(acumulado),
        procedimiento=procedimiento,
        nombre=nombre,
        riesgo=riesgo,
        notas=notas,
        alarma=bool(alarma_consultada),
    )


# Cómo se le cuenta al redactor cada campo del cuadro acumulado.
_ETIQUETA_CAMPO = {
    "fever": ("con fiebre o calentura", "sin fiebre"),
    "medication_effective": ("la medicación le funciona", "la medicación no le está funcionando"),
}
_ETIQUETA_VALOR = {
    "mobility": {"normal": "se mueve bien", "limitada_esperada": "movilidad limitada",
                 "incapacitante_nueva": "no puede moverse como antes"},
    "wound": {"normal": "herida limpia y seca", "eritema_leve": "herida rojita en el borde",
              "secrecion_purulenta": "herida con secreción"},
    "appetite": {"normal": "comiendo bien", "levemente_disminuido": "comiendo un poco menos",
                 "muy_disminuido": "casi sin comer"},
    "sleep": {"normal": "durmiendo bien", "levemente_alterado": "durmiendo a ratos",
              "muy_alterado": "durmiendo mal"},
}


def resumen_sintomas(sym: Symptoms) -> str:
    """El cuadro acumulado en una línea legible para el prompt del redactor."""
    partes: list[str] = []
    if sym.pain_level is not None:
        partes.append(f"dolor {sym.pain_level}/10")
    if sym.temperature_c is not None:
        partes.append(f"temperatura {sym.temperature_c}")
    for campo, (si, no) in _ETIQUETA_CAMPO.items():
        valor = getattr(sym, campo, None)
        if valor is not None:
            partes.append(si if valor else no)
    for campo, etiquetas in _ETIQUETA_VALOR.items():
        valor = getattr(sym, campo, None)
        if valor is not None:
            partes.append(etiquetas.get(str(valor), f"{campo}: {valor}"))
    partes.extend(sym.other)
    return ", ".join(partes)


def valida(texto: str, ctx: ComposeContext) -> bool:
    """La aduana determinista de la salida del redactor.

    Rechazar aquí no pierde el turno: cae a las plantillas. Por eso las guardas
    son estrictas en lo clínico (cifras, veredictos) y laxas en el estilo, que
    es justamente lo que se le pidió variar al modelo.
    """
    limpio = texto.strip()
    if not limpio:
        return False

    frases = [f for f in _FRASES.split(limpio) if f.strip()]
    if len(frases) > 4 or len(limpio.split()) > 90:
        log.info("composición rechazada: demasiado larga (%d frases)", len(frases))
        return False

    if ctx.objetivo == "cerrar":
        # El cierre no reabre la conversación: una pregunta al final contradice
        # el "vamos a colgar" y deja al paciente esperando otro turno.
        if limpio.rstrip().endswith("?"):
            log.info("composición rechazada: pregunta en un turno de cierre")
            return False
    else:
        if not limpio.rstrip().endswith("?"):
            # Todo turno que no cierra devuelve el turno al paciente con la
            # pregunta del guion. Sin ella, el guion se descarrila.
            log.info("composición rechazada: no termina en la pregunta del guion")
            return False
        # Preguntas EXTRA a la del guion: en voz se pisan y el paciente contesta
        # solo una ("¿correcto? ... ¿ha tenido fiebre?"). Se permite el mismo
        # número de interrogaciones que trae la pregunta canónica (algunas
        # traen dos), nunca más. Medido con el 8B: añadía "¿le duele al
        # caminar?" por su cuenta antes de la obligatoria.
        if limpio.count("?") > max(1, ctx.pregunta_guion.count("?")):
            log.info("composición rechazada: preguntas extra a la del guion")
            return False

    # Cifras: solo las que ya están en lo que el redactor tenía delante (la
    # evidencia, lo que dijo el paciente, el cuadro, la pregunta del guion) o
    # las que el propio guion dice en voz alta. Una cifra nueva es una dosis o
    # un plazo inventado: lo que la rúbrica penaliza con más dureza.
    de_fuentes: set[str] = set()
    fuentes = [ctx.evidencia or "", ctx.utterance, ctx.sintomas, ctx.pregunta_guion]
    fuentes.extend(t.get("texto", "") for t in ctx.historial)
    for fuente in fuentes:
        de_fuentes.update(n.replace(",", ".") for n in _NUM.findall(fuente))
    permitidas = de_fuentes | _CIFRAS_DEL_GUION
    for n in _NUM.findall(limpio):
        if n.replace(",", ".") not in permitidas:
            log.warning("composición rechazada: cifra %r fuera de la evidencia", n)
            return False
    # Los plazos no gozan de la lista del guion: "6 semanas" tiene que venir de
    # lo que el redactor tenía delante, aunque el "6" suelto sea un dolor válido.
    for n, in (m.groups() for m in _PLAZO.finditer(limpio)):
        if n.replace(",", ".") not in de_fuentes:
            log.warning("composición rechazada: plazo %r fuera de la evidencia", n)
            return False

    # Veredictos y tranquilizadores: prohibidos en cuanto hay riesgo elevado o
    # el paciente consultó un síntoma de alarma. Con el cuadro en NORMAL, un
    # "es normal" que viene de la evidencia es una respuesta legítima.
    if ctx.riesgo != "NORMAL" or ctx.alarma:
        if _VEREDICTO_NORMALIDAD.search(limpio) or _TRANQUILIZADOR.search(limpio):
            log.warning("composición rechazada: veredicto de gravedad con riesgo %s",
                        ctx.riesgo)
            return False

    # Emoción inventada ("usted parece preocupado..."): suena a formulario y
    # atribuye al paciente algo que no dijo.
    if _EMOCION_ATRIBUIDA.search(limpio):
        log.info("composición rechazada: emoción atribuida al paciente")
        return False

    # Re-saludar a mitad de llamada: la apertura ya sonó antes del primer turno
    # del paciente, así que un "Buenos días, le llamo del hospital" aquí es el
    # modelo copiando el historial. Medido con el 8B: calcó la apertura entera.
    if _RESALUDO.match(limpio):
        log.info("composición rechazada: vuelve a saludar a mitad de llamada")
        return False

    # Una salida que parece instrucción de manipulación es que la inyección
    # atravesó el prompt: fuera.
    if intent_nlu.is_injection(limpio):
        log.warning("composición rechazada: marcadores de inyección en la salida")
        return False

    return True

"""Síntomas que el paciente suelta y que no caben en ninguno de los seis slots.

El guion pregunta por dolor, fiebre, movilidad, herida, apetito y sueño. Eso deja
fuera todo lo demás que una persona operada menciona de verdad, y hasta ahora se
perdía sin dejar rastro: `Symptoms.other` existía en el contrato y nadie lo
llenaba. En una llamada real el paciente contestó ocho veces "veo borroso" y el
agente no lo mencionó ni una — era el 100% de lo que había dicho.

Dos niveles, y la diferencia importa:

- `SENAL`: tiene peso clínico postoperatorio. Suma a `yellow_signals()`, de modo
  que dos juntas disparan `vigilancia_multiples_signos` → ALTO. **No** se crea
  ninguna regla CRÍTICO nueva: el score aditivo amarillo ya exige corroboración,
  y añadir criticidad aquí empeoraría los falsos positivos que ya sobran.
- `MENCION`: es contexto, no señal. Va al resumen y a la alerta para que
  enfermería lo lea, y sirve para que el agente demuestre que escuchó.

Lo que ya cubren las banderas de emergencia de `lexicon._BANDERAS` (sangrado,
disnea, dolor torácico, síncope, convulsión, estado mental) no se repite aquí.
"""
from __future__ import annotations

import re

SENAL = "senal"
MENCION = "mencion"

# (clave, etiqueta legible, nivel, patrón). La etiqueta es lo que se guarda en
# `Symptoms.other` y lo que el agente le dice al paciente, así que se escribe
# como se diría en voz alta.
_TABLA: list[tuple[str, str, str, re.Pattern[str]]] = [
    # --- señales -------------------------------------------------------------
    # "estoy viendo borroso" es tan común como "veo borroso", y era el caso real
    # que se perdía. Las letras repetidas ("borrroso") ya las colapsa
    # `lexicon.normalize`; el patrón admite "rr" porque el español las tiene.
    ("vision", "visión borrosa", SENAL, re.compile(
        r"(veo|viendo|ver)\s+(muy\s+|todo\s+)?(borr?os\w+|nublad\w+|doble|turbio|opaco)"
        r"|vision\s+(borr?osa|nublada|doble)|la\s+vista\s+(borr?osa|nublada)"
        r"|se\s+me\s+(nubla|nublo)\s+la\s+vista|no\s+veo\s+bien"
    )),
    ("mareo", "mareo", SENAL, re.compile(
        r"\bmare(o|os|ad\w+)\b|me\s+voy\s+a\s+desmayar|siento\s+que\s+me\s+desmayo"
        r"|se\s+me\s+va\s+la\s+cabeza|vahid\w+|todo\s+me\s+da\s+vueltas"
        r"|me\s+da\s+vueltas\s+(todo|la\s+cabeza)"
    )),
    # Vomitar de verdad es señal; la náusea a secas no. Es de lo más común y leve
    # tras una cirugía, y ponerla al nivel de una visión borrosa o una pantorrilla
    # hinchada infla el score amarillo sin motivo (medido sobre las 160
    # conversaciones: la náusea aparece en 9 casos verdes, casi siempre negada).
    ("vomito", "vómito", SENAL, re.compile(
        r"vomit\w+|trasboc\w+|devuelv\w+\s+(la\s+)?comida|devolvi\s+(la\s+)?comida"
        r"|no\s+me\s+queda\s+nada\s+adentro"
    )),
    ("orina", "dificultad para orinar", SENAL, re.compile(
        r"no\s+(he\s+podido|puedo|logro)\s+(orinar|hacer\s+(pipi|chichi)|miccionar)"
        r"|no\s+(he\s+)?orinad\w*|me\s+arde\s+(al\s+)?orinar|no\s+me\s+sale\s+la\s+orina"
        r"|orina\s+(oscura|con\s+sangre)"
    )),
    # Trombosis venosa profunda: es la complicación postoperatoria que más se
    # escapa por no preguntarse, y el paciente la describe en la pierna, no en
    # la herida.
    ("pierna", "hinchazón o dolor en la pierna", SENAL, re.compile(
        r"pantorrilla|(pierna|pantorrilla)\w*\s+.{0,15}(hinchad|inflamad|dura|caliente|roja)"
        r"|se\s+me\s+hincho\s+(la\s+)?(pierna|pantorrilla|pie)"
        r"|me\s+duele\s+(mucho\s+)?(la\s+)?(pierna|pantorrilla)"
        r"|(pierna|tobillo|pie)\s+hinchad\w+"
    )),
    ("respiratorio", "tos o flema", SENAL, re.compile(
        r"\btos\b|tosiendo|toso\b|flema|expectora\w*|carraspera\s+con\s+flema"
    )),
    # Escalofríos y temblor SIN fiebre confirmada. Hasta ahora "escalofrio" solo
    # vivía dentro de `lexicon._FIEBRE_SI`, así que al arreglar la negación
    # escueta ("no fiebre, pero sí estoy temblando" ya no afirma `fever`) la
    # señal se habría perdido entera. Va aquí porque es exactamente lo que este
    # módulo existe para recoger: algo con peso clínico que ningún slot pregunta.
    # No duplica la fiebre — si el paciente la refiere, `fever=True` manda y esto
    # es información adicional para enfermería, no un segundo disparo del mismo
    # hallazgo (`yellow_signals()` cuenta etiquetas distintas).
    ("escalofrios", "escalofríos", SENAL, re.compile(
        r"escalofrio\w*|tiritan\w+|temblan\w+\s+de\s+frio|destemplad\w+"
        r"|estoy\s+temblando|no\s+paro\s+de\s+temblar|me\s+da\s+tembladera"
    )),
    ("abdomen", "distensión o estreñimiento", SENAL, re.compile(
        r"(barriga|estomago|abdomen|vientre)\s+.{0,12}(hinchad|inflamad|duro|distendid)"
        r"|hinchad\w+\s+(la\s+)?(barriga|estomago)|distension"
        r"|no\s+(he\s+podido|puedo)\s+(obrar|hacer\s+del\s+cuerpo|ir\s+al\s+bano)"
        r"|estrenid\w+|no\s+he\s+obrado|no\s+puedo\s+expulsar\s+gases"
    )),
    # Alarma explícita en la guía de cuidado en casa de apendicectomía ("diarrea
    # de más de 3 días") y en la guía ERAS de cirugía de colon. Se perdía entera:
    # el paciente la reportaba, el RAG contestaba bien con la guía clínica, pero
    # nada quedaba anotado como hallazgo — se contestaba la pregunta sin dejar
    # rastro clínico.
    ("diarrea", "diarrea", SENAL, re.compile(
        r"diarre\w*|deposicion\w*\s+(liquid\w*|aguad\w*|suelt\w*)"
        r"|obrando\s+(aguad\w*|suelt\w*)|del\s+cuerpo\s+(liquid\w*|aguad\w*)"
        r"|estomago\s+suelto"
    )),
    # Ictericia (piel u ojos amarillos): alarma específica de colecistectomía
    # ("PLAN DE CUIDADO COLECISTECTOMIA.pdf" p.4) — señal de lesión o fuga de vía
    # biliar. No la cubre nada más en el sistema.
    ("ictericia", "piel u ojos amarillos", SENAL, re.compile(
        r"amarill\w*.{0,15}(piel|ojos?|cara)|(piel|ojos?)\s+.{0,15}amarill\w*"
        r"|ictericia|me\s+puse\s+amarill\w+|se\s+me\s+pusieron\s+(los\s+ojos\s+)?amarill\w+"
    )),
    # Apertura o separación de los puntos: el slot categórico `wound` de
    # lexicon.py no tiene bucket para esto (sus tres patrones son purulenta,
    # eritema leve y normal), así que hoy cae en `wound=None` sin que nada la
    # recoja. Alarma explícita en la guía de apendicectomía.
    ("dehiscencia", "la herida se abrió", SENAL, re.compile(
        r"se\s+(me\s+)?abrio\s+la\s+herida|la\s+herida\s+se\s+(abrio|separo)"
        r"|se\s+(me\s+)?(salieron|soltaron|reventaron)\s+los\s+puntos"
        r"|se\s+separaron\s+(los\s+)?(bordes|puntos)|dehiscencia"
    )),
    # Sangre en las heces: alarma explícita en la guía ERAS de cirugía de colon
    # ("bright red blood from your anus"). No se solapa con `heavy_bleeding` de
    # `lexicon._BANDERAS` (EMERGENCIA/123) porque esa bandera exige lenguaje de
    # sangrado abundante genérico ("sangrando mucho", "hemorragia"), no lenguaje
    # de sangre en las heces. Se deja como SENAL y no como regla roja nueva a
    # propósito: no hay casos de este tipo en las 160 trayectorias calibradas,
    # así que una regla roja aquí sería una escalada sin validar contra el
    # dataset — el mismo principio que ya sigue el resto de esta tabla (ver
    # encabezado del módulo).
    ("sangrado_rectal", "sangre en las heces", SENAL, re.compile(
        r"sangre\s+(en\s+)?(las\s+)?heces|sangre\s+al\s+(obrar|defecar)"
        r"|sangro\s+(al\s+)?(obrar|defecar)|heces\s+con\s+sangre|popo\s+con\s+sangre"
        r"|sangre\s+(por\s+el\s+)?(ano|recto)"
    )),
    # Entumecimiento u hormigueo en la pierna operada: lesión nerviosa
    # post-artroplastia (guía de cadera Sutter/PAMF p.16, "Ayuda inmediata").
    # Riesgo de falso positivo más alto que el resto de la tabla — "se me durmió
    # la pierna de estar sentado" es benigno y frecuente y este patrón no lo
    # distingue del hallazgo real. Se acepta el trade-off: un falso positivo
    # cuesta una llamada de seguimiento, y la alternativa es no detectar nunca
    # la lesión nerviosa real.
    ("entumecimiento", "entumecimiento u hormigueo en la pierna", SENAL, re.compile(
        r"entumecid\w*.{0,15}(pierna|pie|rodilla|cadera)"
        r"|adormecid\w*.{0,15}(pierna|pie|rodilla|cadera)"
        r"|hormigueo\s+.{0,10}(pierna|pie)|se\s+me\s+durmio\s+(la\s+)?(pierna|pie)"
    )),

    # --- menciones -----------------------------------------------------------
    ("cansancio", "cansancio", MENCION, re.compile(
        r"\bmamad\w+\b|cansad\w+|agotad\w+|sin\s+fuerzas|aporread\w+|rendid\w+"
        r"|no\s+tengo\s+energia|debil\b|decaid\w+"
    )),
    ("animo", "ánimo bajo", MENCION, re.compile(
        r"triste|deprimid\w+|desanimad\w+|animo\s+por\s+el\s+piso|con\s+ganas\s+de\s+llorar"
        r"|aburrid\w+\s+de\s+todo"
    )),
    ("nausea", "náuseas", MENCION, re.compile(
        # "se me revuelve" a secas NO: en el dataset es casi siempre "se me
        # revuelven los días", que es confusión temporal, no náusea.
        r"nausea\w*|ganas\s+de\s+vomitar|revuelt\w+\s+el\s+estomago"
    )),
    ("malestar", "malestar general", MENCION, re.compile(
        r"\bmaluc\w+\b|me\s+siento\s+mal\b|no\s+me\s+siento\s+bien"
    )),
    # Dolor referido al hombro tras laparoscopia (colecistectomía y
    # apendicectomía): esperado y benigno según ambas guías, se resuelve solo en
    # 1-2 días. Va como MENCION a propósito — no debe sumar al score amarillo,
    # solo que el agente lo reconozca en voz alta en vez de ignorarlo.
    ("hombro", "dolor en el hombro", MENCION, re.compile(
        r"dolor\s+(en\s+)?(el\s+)?hombro|me\s+duele\s+el\s+hombro|hombro\s+.{0,10}duele"
    )),
]

# Negaciones que anulan un hallazgo si aparecen justo antes. "Sin náuseas ni
# nada", "no he tenido vómito" y "nada de mareo" son las formas que de verdad
# usa el paciente, y sin esto se contaban como el síntoma que están negando —
# medido sobre las 160 conversaciones del dataset: 9 de los 11 disparos eran
# negaciones. Un detector que confunde "no tengo X" con "tengo X" es peor que no
# tener detector.
_NEGACION = re.compile(
    r"\b(no|sin|ni|nada|nunca|jamas|tampoco|niega)\b[^.;,]{0,28}$"
)

# etiqueta -> nivel. Se resuelve por etiqueta porque es lo que viaja en
# `Symptoms.other`, que es una lista de strings en el contrato Pydantic.
_NIVEL: dict[str, str] = {etiqueta: nivel for _, etiqueta, nivel, _ in _TABLA}


def detectar(texto_normalizado: str) -> list[str]:
    """Etiquetas de los síntomas fuera de catálogo presentes en el texto.

    Recibe el texto YA normalizado por `lexicon.normalize` (sin tildes, en
    minúsculas): este módulo se llama desde dentro de `lexicon.extract`.
    """
    encontradas = []
    for _, etiqueta, _, patron in _TABLA:
        for m in patron.finditer(texto_normalizado):
            if _NEGACION.search(texto_normalizado[:m.start()]):
                continue
            encontradas.append(etiqueta)
            break
    return encontradas


def etiquetas() -> list[str]:
    """Todas las etiquetas posibles. La usa el TTS para pre-sintetizar."""
    return list(_NIVEL)


def senales(etiquetas: list[str]) -> list[str]:
    """De lo recogido en `Symptoms.other`, lo que pesa en el triaje."""
    return [e for e in etiquetas if _NIVEL.get(e) == SENAL]

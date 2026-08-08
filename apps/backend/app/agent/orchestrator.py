"""Orquestador de un turno — el corazón del flujo de decisión.

Estaba en `voice/conversation.py`, que era un mal sitio: no tiene nada que ver con
la voz, es el servicio que comparten el endpoint de texto y el pipeline de audio.

Cinco etapas explícitas, de las cuales **solo dos tocan el LLM**:

    A. determinista   léxico colombiano → intención → inyección          (<5 ms)
    B. LLM            extracción del slot, solo si el léxico no lo resolvió
    C. determinista   fusión por severidad → engine.evaluate()           (<1 ms)
    D. determinista   script.next_action() → qué se dice a continuación
    E. LLM            respuesta anclada al RAG, solo si preguntó algo clínico

**Excepción de seguridad:** si el léxico detecta una bandera de emergencia, se
saltan B, D y E por completo. Se emite el guion determinista y se alerta. La ruta
crítica es la más corta del sistema, y no pasa por el modelo: ni uno caído ni uno
manipulado pueden suprimir un escalamiento.

El LLM nunca controla el flujo. Su salida es *dato* —un slot de un enum cerrado—,
nunca instrucciones.
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import re
import time

from sqlalchemy.orm import Session

from ..decision import engine
from ..llm.factory import get_llm
from ..llm.ollama_adapter import ABSTENCION, es_abstencion
from ..models import Alert, Conversation, Patient, Turn
from ..nlu import lexicon, otros_sintomas
from ..nlu.merge import merge_symptoms
from ..rag import retrieve as rag
from ..rag.embeddings import build_query
from ..schemas import RagResult, Symptoms, TurnResponse
from . import phrasing, script
from .script import Action, CallState, Phase

log = logging.getLogger(__name__)

# Ventana para la rotación anti-repetición del banco de frases.
_VENTANA_ESTILO = 5


def _get_or_create_conversation(
    session: Session, conversation_id: str | None, patient_id: str | None
) -> Conversation:
    if conversation_id:
        conv = session.get(Conversation, conversation_id)
        if conv is not None:
            return conv
    conv = Conversation(patient_id=patient_id, status="active")
    session.add(conv)
    session.flush()
    return conv


def _prior_turns(session: Session, conversation_id: str) -> list[Turn]:
    return (
        session.query(Turn)
        .filter(Turn.conversation_id == conversation_id)
        .order_by(Turn.created_at)
        .all()
    )


def _acumular(turns: list[Turn]) -> tuple[Symptoms, CallState]:
    """Reconstruye el cuadro clínico y el estado del guion desde la traza.

    La memoria vive aquí, en la base, y no en el contexto del modelo. Eso recorta
    ~800 tokens por turno y elimina la deriva típica de un modelo pequeño que se
    olvida de lo que ya preguntó.
    """
    acc = Symptoms()
    estado = CallState()
    for t in turns:
        acc = merge_symptoms(acc, Symptoms.model_validate(t.extracted_symptoms or {}))
        if t.agent_state:
            estado = CallState.from_dict(t.agent_state)
    return acc, estado


def _aperturas_recientes(turns: list[Turn]) -> list[str]:
    return [t.final_response for t in turns[-_VENTANA_ESTILO:]]


# Campos de `Symptoms` que son un dato del paciente. Se excluyen los de
# contabilidad (`sources`, `unanswered`) y `temperature_measured`, que dice si
# hay termómetro, no cómo está el paciente.
_CAMPOS_CLINICOS = (
    "pain_level", "temperature_c", "fever", "mobility", "wound", "appetite",
    "sleep", "medication_effective", "heavy_bleeding", "breathing_difficulty",
    "loss_of_consciousness", "chest_pain", "altered_mental_status", "seizure",
)


def _aporto_dato(del_turno: Symptoms, nuevos_otros: list[str]) -> bool:
    """¿Este turno dejó algo NUEVO anotado?

    Dos usos: hace honesto al acuse de recibo —"vale, anotado" solo se puede decir
    si de verdad se anotó algo— y alimenta el contador de atasco.

    `nuevos_otros` son los síntomas fuera de catálogo que no estaban ya en el
    acumulado. Es la diferencia importante: repetir "veo borroso" por octava vez
    no es información nueva, y contarlo como progreso es lo que impedía que el
    detector de atasco llegara a dispararse nunca.
    """
    return bool(nuevos_otros) or any(
        getattr(del_turno, c) is not None for c in _CAMPOS_CLINICOS
    )


def _texto_de(action: Action, *, semilla: str, recientes: list[str],
              nombre: str | None, preocupante: bool, con_acuse: bool = True,
              slot_respondido: str | None = None,
              acumulado: Symptoms | None = None,
              del_turno: Symptoms | None = None) -> str:
    """Traduce la acción del guion a lo que se dice. Todo determinista.

    `con_acuse=False` suprime el acuse de recibo. Se usa cuando el turno ya se
    reconoció por otra vía —o cuando no hay nada que acusar—: decir "vale,
    anotado" después de un turno del que no se extrajo nada afirma algo falso, y
    en la llamada que motivó este cambio el agente lo dijo cuatro veces seguidas.
    """
    if action.kind == "ofrecer_salida":
        return phrasing.ofrecer_salida(semilla, recientes)
    if action.kind == "repreguntar":
        # Normalmente una repregunta no lleva acuse: el slot no se resolvió y
        # decir "anotado" sería falso. Pero el turno puede haber traído OTRO dato
        # ("y la herida se ve rojita" mientras se pregunta por el dolor), y ahí
        # sí hay algo que reconocer — lo decide `con_acuse` abajo.
        pregunta = phrasing.repregunta(action.slot, action.intento)
    elif action.kind == "seguimiento":
        # El paciente sí contestó, así que aquí el acuse es sincero: reconoce lo
        # que dijo y pide el dato que falta.
        pregunta = phrasing.seguimiento(action.seguimiento, semilla, recientes)
    elif action.kind == "cerrar":
        return phrasing.cierre(nombre, escalado=preocupante)
    elif action.kind == "preguntar":
        # `slot is None` es el turno abierto del final. También lleva acuse: es la
        # última respuesta del tamizaje y callarla la deja sin reconocer.
        pregunta = (phrasing.pregunta(action.slot, semilla, recientes)
                    if action.slot else phrasing.PREGUNTA_ABIERTA)
    else:
        return phrasing.PREGUNTA_ABIERTA

    if not con_acuse:
        return pregunta
    # Si se puede decir QUÉ se entendió, se dice. El acuse genérico ("vale,
    # anotado") es el respaldo para cuando el dato no tiene reflejo — por ejemplo
    # el que llegó por extracción cruzada de un slot que no se preguntó.
    # El reflejo se intenta SIEMPRE, también con riesgo alto. Antes se saltaba
    # cuando `preocupante`, y el resultado era que en cuanto la llamada subía a
    # ALTO el agente dejaba de decir qué había entendido y pasaba a soltar
    # simpatía genérica ("Qué bueno que me lo cuenta") en cada turno. Los reflejos
    # no tranquilizan ni valoran gravedad —eso lo sigue haciendo el motor de
    # decisión—, así que son seguros en cualquier nivel de riesgo.
    ack = None
    if acumulado is not None:
        ack = phrasing.reflejo(slot_respondido, acumulado, del_turno)
        if ack is None and del_turno is not None:
            # El dato llegó por extracción cruzada: el paciente contestó sobre un
            # slot distinto al que se preguntaba ("y la herida se ve rojita"
            # mientras se pregunta por el dolor). Reflejarlo es lo que demuestra
            # que se oyó; si no, el turno se responde con una repregunta seca.
            for otro, campo in script.SLOT_FIELD.items():
                if otro != slot_respondido and getattr(del_turno, campo) is not None:
                    ack = phrasing.reflejo(otro, acumulado)
                    if ack:
                        break
    # Un reflejo repetido suena a bucle igual que un acuse repetido.
    if ack is not None and any(ack in r for r in recientes):
        ack = None
    if ack is None:
        ack = phrasing.acuse(semilla, recientes, preocupante=preocupante)
    return f"{ack} {pregunta}"


async def process_turn_async(
    session: Session,
    *,
    text: str,
    conversation_id: str | None = None,
    patient_id: str | None = None,
) -> TurnResponse:
    started = time.perf_counter()
    conv = _get_or_create_conversation(session, conversation_id, patient_id)
    prior = _prior_turns(session, conv.id)
    acumulado, estado = _acumular(prior)

    paciente = session.get(Patient, conv.patient_id) if conv.patient_id else None
    procedimiento = paciente.surgery if paciente else None
    nombre = paciente.name.split()[0] if paciente else None

    llm = get_llm()
    # Se acumulan TODAS las llamadas del turno, no solo la extracción: si el
    # paciente preguntó algo, la respuesta anclada es la que más tokens consume y
    # dejarla fuera falsearía las métricas del README a la baja.
    llm_calls, tokens_in, tokens_out = 0, 0, 0

    # --- A + B: entender lo que dijo el paciente ------------------------------
    slot = estado.slot_actual if estado.phase is Phase.TAMIZAJE else None
    pregunta_previa = prior[-1].final_response if prior else phrasing.APERTURA
    extraccion = await llm.extract(slot=slot, question=pregunta_previa, utterance=text)
    if extraccion.usage.tokens_out:
        llm_calls += 1
    tokens_in += extraccion.usage.tokens_in
    tokens_out += extraccion.usage.tokens_out

    del_turno = extraccion.symptoms
    # Antes de fusionar: lo que el paciente cuenta fuera del guion y todavía no
    # estaba anotado. Solo eso se le reconoce en voz alta y solo eso cuenta como
    # progreso — repetirle "me apunto lo de la visión borrosa" en cada turno es
    # tan robótico como no decírselo nunca.
    nuevos_otros = [o for o in del_turno.other if o not in acumulado.other]
    acumulado = merge_symptoms(acumulado, del_turno)
    if paciente and paciente.extra.get("dia_postop"):
        acumulado.day_postop = paciente.extra["dia_postop"]

    # --- C: decidir. Código puro, nunca el modelo -----------------------------
    decision = engine.evaluate(acumulado)
    critico = decision.risk_level == "CRÍTICO"

    # --- D: qué se dice ------------------------------------------------------
    # Atasco: el paciente repite la misma frase, o lleva varios turnos sin dejar
    # ningún dato. El guion necesita saberlo para cambiar de estrategia en vez de
    # seguir recorriendo slots.
    hash_turno = hashlib.sha256(lexicon.normalize(text).encode()).hexdigest()[:16]
    repetido = hash_turno == estado.ultimo_hash
    progreso = _aporto_dato(del_turno, nuevos_otros)

    # `emergencia` distingue las 6 banderas del 123 de la vía de enfermería: ante
    # una emergencia real, retener al paciente en la línea compite con la llamada
    # que de verdad importa, así que ahí sí se cuelga rápido.
    emergencia = decision.escalation_action == "emergencia_123"
    quiere_colgar = extraccion.intent == "despedida"

    action = script.next_action(estado, acumulado, escalar=critico, repetido=repetido,
                                emergencia=emergencia, quiere_colgar=quiere_colgar)
    recientes = _aperturas_recientes(prior)
    evidence: RagResult | None = None

    # Si este turno cierra la llamada, la política de incertidumbre ya aplica: una
    # llamada que no se pudo evaluar no se despide como si el paciente estuviera
    # bien. `close_conversation` lo reevaluaba igual y creaba la alerta, pero el
    # texto hablado se había generado antes y decía "que siga bien" mientras el
    # sistema escalaba a enfermería por detrás. `final=True` solo puede igualar o
    # subir el nivel: `merge_symptoms` es monótono.
    # `phase is not ESCALAMIENTO` evita el bucle: tras entregar el guion, el turno
    # siguiente vuelve a pedir "cerrar" y `final=True` volvería a dar CRÍTICO.
    if (action.kind == "cerrar" and not critico
            and estado.phase is not Phase.ESCALAMIENTO):
        decision = engine.evaluate(acumulado, final=True)
        critico = decision.risk_level == "CRÍTICO"
        if critico:
            # El guion de seguridad manda sobre la despedida neutra.
            action = Action(kind="escalar", phase=Phase.ESCALAMIENTO)

    if critico and action.kind == "escalar":
        # Primer turno crítico: se entrega el guion de seguridad completo.
        # Ruta crítica: no pasa por el modelo ni por el RAG.
        final = decision.safety_script or phrasing.cierre(nombre, escalado=True)
    elif action.kind == "confirmar":
        # El agente ya dijo lo suyo (guion de seguridad o fin del tamizaje) y le
        # devuelve el turno al paciente en vez de colgar. Va antes que la rama de
        # `critico` para que el guion de seguridad NO se repita aquí.
        semilla = conv.id + str(len(prior))
        final = phrasing.confirmar_cierre(semilla, recientes)
        if nuevos_otros:
            # Si en este turno contó algo nuevo —"y la herida se ve rojita"—, se
            # reconoce antes de volver a preguntar si quiere colgar.
            final = f"{phrasing.acuse_otro(nuevos_otros, semilla, recientes)} {final}"
    elif critico or estado.phase is Phase.ESCALAMIENTO:
        # `phase is ESCALAMIENTO` cubre el escalamiento que nace de la política de
        # incertidumbre al cerrar: ese `no_se_pudo_evaluar` no vive en `acumulado`
        # —se sintetiza con `final=True`—, así que `critico` vuelve a ser False al
        # turno siguiente y la llamada se despedía con un "que siga bien" después
        # de haber escalado a enfermería.
        # El cuadro crítico sigue en `acumulado` para siempre —no se puede
        # "des-escalar" dentro de la llamada—, así que sin este segundo caso
        # `critico` seguiría siendo True en cada turno y el guion completo se
        # repetiría palabra por palabra mientras el paciente siga hablando.
        # `script.next_action` ya lo convirtió en `action.kind == "cerrar"`; aquí
        # se cierra la llamada de verdad en vez de repetir el guion.
        final = phrasing.cierre_tras_escalamiento(nombre)
    elif extraccion.intent == "fuera_de_mision":
        final = phrasing.FUERA_DE_MISION
    elif extraccion.intent == "rechazo":
        final = phrasing.RECHAZO
        action = Action(kind="cerrar", phase=Phase.TERMINADA)
    elif extraccion.intent == "despedida":
        # El paciente da la llamada por terminada. `next_action` ya lo convirtió
        # en `cerrar` vía `quiere_colgar`; aquí solo se elige la despedida, que es
        # cálida y no la fórmula seca del rechazo.
        final = phrasing.cierre(nombre, escalado=decision.risk_level != "NORMAL")
    elif extraccion.intent == "ininteligible":
        final = phrasing.NO_ENTENDI
    elif extraccion.intent == "saludo":
        # El paciente saluda: se le devuelve el saludo y se le hace la pregunta
        # pendiente en su forma ABIERTA, no la reformulación cerrada. Un saludo no
        # es una respuesta esquivada — ver `intento_real` en script.apply.
        semilla = conv.id + str(len(prior))
        slot_pendiente = action.slot or estado.slot_actual
        pregunta = (phrasing.pregunta(slot_pendiente, semilla, recientes)
                    if slot_pendiente else phrasing.PREGUNTA_ABIERTA)
        final = f"{phrasing.saludo_de_vuelta(semilla, recientes)} {pregunta}"
    elif extraccion.intent == "meta":
        siguiente = _texto_de(action, semilla=conv.id + str(len(prior)),
                              recientes=recientes, nombre=nombre, preocupante=False)
        final = f"{phrasing.META_REPETIR} {siguiente}"
    elif extraccion.intent == "social":
        siguiente = _texto_de(action, semilla=conv.id + str(len(prior)),
                              recientes=recientes, nombre=nombre, preocupante=False)
        final = f"{phrasing.SOCIAL} {siguiente}"
    elif extraccion.intent == "pregunta_clinica" and otros_sintomas.senales(nuevos_otros):
        # El paciente contó un síntoma de alarma y preguntó por él en el mismo
        # turno. Eso NO va al RAG: no está preguntando qué dice el hospital sobre
        # un tema, está pidiendo que le valoren su caso, y el corpus no puede
        # responderlo. Enviarlo igual fue lo que produjo la respuesta más dañina
        # medida hasta ahora: el RAG trajo radiología de apendicitis para una
        # pregunta sobre visión borrosa. Ruta determinista, sin modelo.
        semilla = conv.id + str(len(prior))
        siguiente = _texto_de(action, semilla=semilla, recientes=recientes,
                              nombre=nombre, preocupante=False, con_acuse=False)
        final = (phrasing.sintoma_consultado(
            otros_sintomas.senales(nuevos_otros), semilla, recientes)
            + " " + phrasing.volviendo(semilla, recientes) + " " + siguiente)
    elif extraccion.intent == "pregunta_clinica":
        # --- E: la única respuesta generada, y va anclada a evidencia ---------
        evidence = rag.retrieve(
            session,
            build_query(text, procedure=procedimiento, day_postop=acumulado.day_postop),
            procedure=procedimiento,
        )
        # Antes de redactar, se comprueba que lo recuperado responda de verdad la
        # pregunta. La similitud vectorial no lo dice (ver rag/retrieve.py), y sin
        # este paso el agente contestaba preguntas sobre protocolos inventados
        # citando documentos sin relación: una afirmación falsa CON fuente.
        pertinente = evidence.has_evidence and await llm.evidencia_responde(
            question=text, evidence=evidence.answer)
        if evidence.has_evidence and not pertinente:
            log.info("evidencia recuperada pero no pertinente para: %r", text[:60])
            evidence.has_evidence = False

        respuesta, uso = await llm.reply_grounded(
            question=text,
            evidence=evidence.answer if pertinente else "",
            patient_context=f"Paciente de {procedimiento or 'cirugía'}.",
        )
        if uso.tokens_out:
            llm_calls += 1
        tokens_in += uso.tokens_in
        tokens_out += uso.tokens_out
        respuesta = _validar_grounding(respuesta, evidence, riesgo=decision.risk_level)
        # Si se acabó abstiniendo, no se citan fuentes: decir "no tengo información"
        # y adjuntar una cita es incoherente, y en la traza parecería que el agente
        # ignoró evidencia que sí tenía.
        siguiente = _texto_de(action, semilla=conv.id + str(len(prior)),
                              recientes=recientes, nombre=nombre, preocupante=False)
        if es_abstencion(respuesta):
            evidence = None
            # Sin este puente, la pregunta clínica sin resolver se pegaba justo
            # antes de la siguiente pregunta del guion y sonaba a que el agente
            # ignoraba lo que acababa de decir en vez de retomar el seguimiento.
            transicion = phrasing.transicion_abstencion(
                conv.id + str(len(prior)), recientes)
            final = f"{respuesta} {transicion} {siguiente}"
        else:
            final = f"{respuesta} {siguiente}"
    else:
        preocupante = decision.risk_level != "NORMAL"
        semilla = conv.id + str(len(prior))
        prefijo = phrasing.TERCERO + " " if extraccion.intent == "tercero" else ""
        # Lo que el paciente trajo por su cuenta se reconoce por su nombre ANTES
        # de retomar el guion, y con el reconocimiento delante el acuse genérico
        # sobra (`con_acuse`). Uno solo por turno, no tres apilados: en voz,
        # encadenar "lo anoto como que no me supo decir" + "me apunto lo de la
        # visión borrosa" + la pregunta es un párrafo que nadie escucha entero.
        # Manda lo que el paciente trajo, que es lo que demuestra que se le oyó.
        if nuevos_otros:
            prefijo += (
                phrasing.acuse_otro(nuevos_otros, semilla, recientes)
                + " " + phrasing.volviendo(semilla, recientes) + " "
            )
        else:
            # Si el guion acaba de dar por perdido el slot anterior, se dice.
            # Callarlo y saltar a la siguiente pregunta se lee como que no escuchó.
            anterior = estado.slot_actual
            if (action.kind == "preguntar" and anterior and anterior != action.slot
                    and getattr(acumulado, script.SLOT_FIELD[anterior]) is None):
                prefijo += phrasing.slot_perdido(semilla, recientes) + " "
        final = prefijo + _texto_de(action, semilla=semilla,
                                    recientes=recientes, nombre=nombre,
                                    preocupante=preocupante,
                                    con_acuse=not nuevos_otros and progreso,
                                    slot_respondido=estado.slot_actual,
                                    acumulado=acumulado, del_turno=del_turno)

    # Saludar, pedir que se repita la pregunta o decir algo social no es intentar
    # contestarla: esos turnos no gastan reintentos del slot.
    nuevo_estado = script.apply(
        estado, action, acumulado, hash_turno=hash_turno, progreso=progreso,
        intento_real=extraccion.intent not in ("saludo", "meta", "social"),
    )

    # --- alerta, deduplicada por reglas nuevas -------------------------------
    alert_id = _crear_alerta_si_procede(session, conv, decision, acumulado, text)

    # La llamada termina aquí: o es el segundo turno tras un escalamiento crítico
    # (el primero entregó el guion; este solo confirma y cierra), o el guion se
    # agotó normalmente, o el paciente se despidió/rechazó seguir. Antes esto
    # exigía `critico`, así que una llamada que cerraba SIN pasar por CRÍTICO
    # —agotó el guion, o el paciente colgó— nunca llegaba a `close_conversation`,
    # y la política de incertidumbre (`engine.evaluate(..., final=True)`, ver
    # `summary/service.py`) nunca se aplicaba. `action.kind == "cerrar"` ya es la
    # señal correcta de "la llamada termina ahora" venga de donde venga. El
    # pipeline de voz usa `call_ended` para colgar de verdad después de decir el
    # cierre.
    call_ended = action.kind == "cerrar"

    sources = evidence.sources if evidence else []
    turn = Turn(
        conversation_id=conv.id,
        patient_utterance=text,
        extracted_symptoms=del_turno.model_dump(),
        retrieved_chunks=[s.model_dump() for s in sources],
        confidence=evidence.confidence if evidence else None,
        triggered_rules=decision.triggered_rules,
        risk_level=decision.risk_level,
        critical_override=critico,
        final_response=final,
        latency_ms=int((time.perf_counter() - started) * 1000),
        agent_state=nuevo_estado.to_dict(),
        intent=extraccion.intent,
        llm_calls=llm_calls,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        degraded=extraccion.degraded,
    )
    session.add(turn)
    session.commit()

    if call_ended:
        # Cierra la conversación y genera el resumen (RF-10) en el mismo turno
        # que la cierra, para que quede listo sin depender de una llamada
        # aparte a /conversation/{id}/close.
        from ..summary.service import close_conversation

        close_conversation(session, conv.id)

    return TurnResponse(
        conversation_id=conv.id,
        turn_id=turn.id,
        response=final,
        risk_level=decision.risk_level,
        triggered_rules=decision.triggered_rules,
        symptoms=acumulado,
        sources=sources,
        critical_override=critico,
        alert_id=alert_id,
        call_ended=call_ended,
    )


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

_MATIZ_SIN_TRANQUILIZAR = (
    " De todos modos, como me contó otras cosas, prefiero que enfermería le eche "
    "un ojo."
)

# Dictaminar sobre la normalidad de un síntoma, EN CUALQUIERA DE LOS DOS SENTIDOS.
# `_TRANQUILIZADOR` solo cubría el lado complaciente y solo con riesgo != NORMAL;
# medido en una llamada real, ante "estoy viendo borroso, ¿es normal?" el 3B
# abrió con "No, no es normal" —un veredicto clínico, sin evidencia que lo
# sostenga y con el riesgo aún en NORMAL, así que ninguna guarda lo tocaba—.
# Quién decide la gravedad es el motor de decisión, y el prompt ya se lo dice al
# modelo; esto es la versión que no depende de que obedezca.
_VEREDICTO_NORMALIDAD = re.compile(
    r"(no|s[ií])[,\s]+(eso\s+)?(s[ií]|no)?\s*es\s+(algo\s+|completamente\s+)?normal"
    r"|(no|s[ií])\s+es\s+(algo\s+|completamente\s+)?normal"
    r"|(eso|esto)\s+(no\s+)?es\s+normal"
    r"|es\s+(algo\s+)?(preocupante|de\s+cuidado|grave)",
    re.I,
)

# El prompt prohíbe que la respuesta haga preguntas —el guion añade la suya
# después—, y el 3B lo ignora igual ("¿Se duele el abdomen?" colado en medio de
# una respuesta sobre visión borrosa). Se recorta con código.
def _sin_preguntas(texto: str) -> str:
    """Quita las frases interrogativas de una respuesta generada."""
    frases = [f.strip() for f in re.split(r"(?<=[.!?])\s+", texto) if f.strip()]
    return " ".join(f for f in frases if "?" not in f)


def _validar_grounding(
    respuesta: str, evidence: RagResult | None, *, riesgo: str = "NORMAL"
) -> str:
    """Última red contra la alucinación clínica, y es determinista.

    Dos comprobaciones, ninguna delegada al modelo:

    1. Si la respuesta menciona una cifra que no aparece literalmente en la
       evidencia, se sustituye por la abstención. Inventar una dosis o un plazo
       es lo que la rúbrica penaliza con más dureza.
    2. Si el cuadro ya tiene señales de alarma y la respuesta suena tranquilizadora,
       se le añade el matiz de escalamiento. Un 3B complaciente responde "no es
       nada grave" a un eritema, y ese eritema es precisamente una de las cinco
       señales que suman al amarillo.
    """
    from ..llm.ollama_adapter import grounded_in_evidence

    if evidence is None or not evidence.has_evidence:
        return ABSTENCION
    if es_abstencion(respuesta):
        return respuesta          # el modelo ya declaró el límite; se respeta

    # Se quitan las preguntas antes que nada: el guion añade la suya después, y
    # dos preguntas seguidas en voz hacen que el paciente conteste solo una.
    respuesta = _sin_preguntas(respuesta)

    # Un veredicto de normalidad no es una respuesta anclada: es una valoración
    # clínica, y el agente no valora. Se recorta la frase que lo contiene en vez
    # de tirar toda la respuesta, porque el resto suele estar bien anclado.
    if _VEREDICTO_NORMALIDAD.search(respuesta):
        log.warning("veredicto de gravedad en la respuesta, se recorta: %r", respuesta[:80])
        respuesta = " ".join(
            f for f in re.split(r"(?<=[.!?])\s+", respuesta)
            if f.strip() and not _VEREDICTO_NORMALIDAD.search(f)
        ).strip()

    # Lo que quede tiene que seguir siendo una respuesta.
    if len(respuesta.split()) < 4:
        return ABSTENCION
    if not grounded_in_evidence(respuesta, evidence.answer):
        log.warning("respuesta descartada por cifras fuera de la evidencia: %r", respuesta)
        return ABSTENCION
    if riesgo != "NORMAL" and _TRANQUILIZADOR.search(respuesta):
        log.warning("respuesta tranquilizadora con riesgo %s: se añade el matiz", riesgo)
        return respuesta.rstrip(". ") + "." + _MATIZ_SIN_TRANQUILIZAR
    return respuesta


def _crear_alerta_si_procede(
    session: Session, conv: Conversation, decision, symptoms: Symptoms, text: str
) -> str | None:
    if decision.risk_level not in ("ALTO", "CRÍTICO"):
        return None
    ya: set[str] = set()
    for a in session.query(Alert).filter(Alert.conversation_id == conv.id).all():
        ya.update(a.triggered_rules or [])
    nuevas = [r for r in decision.triggered_rules if r not in ya]
    if not nuevas:
        return None
    alerta = Alert(
        conversation_id=conv.id, patient_id=conv.patient_id,
        risk_level=decision.risk_level, triggered_rules=decision.triggered_rules,
        symptoms=symptoms.model_dump(), transcript=text,
    )
    session.add(alerta)
    session.flush()
    return alerta.id


def process_turn(
    session: Session, *, text: str,
    conversation_id: str | None = None, patient_id: str | None = None,
) -> TurnResponse:
    """Envoltorio síncrono para el endpoint de texto y el framework de evaluación."""
    return asyncio.run(process_turn_async(
        session, text=text, conversation_id=conversation_id, patient_id=patient_id))

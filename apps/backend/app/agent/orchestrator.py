"""Orquestador de un turno — el corazón del flujo de decisión.

Estaba en `voice/conversation.py`, que era un mal sitio: no tiene nada que ver con
la voz, es el servicio que comparten el endpoint de texto y el pipeline de audio.

Seis etapas explícitas:

    A. determinista   léxico colombiano → intención → inyección          (<5 ms)
    B. LLM            extracción del slot, solo si el léxico no lo resolvió
    C. determinista   fusión por severidad → engine.evaluate()           (<1 ms)
    D. determinista   script.next_action() → QUÉ se hace a continuación
    E. determinista   RAG + pertinencia, solo si preguntó algo clínico
    F. LLM            compose_reply: CÓMO se dice, viendo el historial —
                      con `composer.valida` como aduana y las plantillas de
                      `phrasing.py` como fallback (`_texto_fallback`)

**Excepción de seguridad:** si el léxico detecta una bandera de emergencia, se
salta todo lo demás. Se emite el guion determinista y se alerta. La ruta crítica
es la más corta del sistema, y no pasa por el modelo: ni uno caído ni uno
manipulado pueden suprimir un escalamiento. Los guiones CRÍTICOS, la escalera de
silencios, la anti-inyección y los cierres finales tampoco pasan por el redactor.

El LLM nunca controla el flujo: decide fraseo, no acciones. La extracción es
*dato* (un enum cerrado) y la redacción termina obligatoriamente en la pregunta
que el guion ya eligió.
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import re
import time

from sqlalchemy.orm import Session

from ..config import get_settings
from ..decision import engine
from ..llm.adapter import Extraction, LLMUsage
from ..llm.factory import get_compose_llm, get_llm
from ..llm.ollama_adapter import ABSTENCION, es_abstencion
from ..models import Alert, Conversation, Patient, Turn
from ..nlu import intent as intent_nlu, lexicon, otros_sintomas
from ..nlu import procedimiento as procedimiento_nlu
from ..nlu.merge import merge_symptoms
from ..rag import retrieve as rag
from ..rag.embeddings import build_query
from ..schemas import RagResult, Symptoms, TurnResponse
from . import composer, phrasing, script
# Reexportadas desde composer: la aduana del redactor y la validación del
# fallback comparten las mismas guardas, y los tests las importan de aquí.
from .composer import _TRANQUILIZADOR, _VEREDICTO_NORMALIDAD  # noqa: F401
from .script import Action, CallState, Phase

log = logging.getLogger(__name__)

# Ventana para la rotación anti-repetición del banco de frases.
_VENTANA_ESTILO = 5

# Nombre de la regla con la que se marca la llamada que se quedó sin nadie al
# otro lado. No está en `decision/rules.py` a propósito: allí solo viven reglas
# que miran síntomas, y esto es un hecho de la llamada, no del paciente.
_REGLA_SIN_RESPUESTA = "paciente_no_responde"

# El paciente dice que lo operaron de otra cosa distinta a la de su ficha. Igual
# que la anterior, no es una regla de `decision/rules.py`: no es un hallazgo
# clínico, es un problema de integridad del registro. Y por eso —a diferencia de
# `_REGLA_SIN_RESPUESTA`— NO eleva el `risk_level` del turno: teñiría de amarillo
# llamadas clínicamente verdes. Levanta su alerta y ya, que es lo que hace falta
# para que alguien vaya a mirar la ficha.
_REGLA_PROCEDIMIENTO = "procedimiento_no_coincide"

# El CANAL se cayó con la llamada abierta (WebRTC cerrado, red muerta).
# Deliberadamente DISTINTA de `paciente_no_responde`: aquella significa "estaba
# en línea y enmudeció" (enfermería valora al paciente); esta significa "la
# llamada se cortó" (primero se re-marca). Fusionarlas haría que la consola
# tratara un fallo de red como un paciente que no contesta.
_REGLA_CONEXION_PERDIDA = "conexion_perdida"


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


def _aporto_dato(del_turno: Symptoms, nuevos_otros: list[str],
                 procedimiento_nuevo: bool = False) -> bool:
    """¿Este turno dejó algo NUEVO anotado?

    Dos usos: hace honesto al acuse de recibo —"vale, anotado" solo se puede decir
    si de verdad se anotó algo— y alimenta el contador de atasco.

    `nuevos_otros` son los síntomas fuera de catálogo que no estaban ya en el
    acumulado. Es la diferencia importante: repetir "veo borroso" por octava vez
    no es información nueva, y contarlo como progreso es lo que impedía que el
    detector de atasco llegara a dispararse nunca.

    `procedimiento_nuevo` sigue la misma regla: contar de qué lo operaron SÍ es
    aportar información —y de la que más consecuencias tiene, porque decide qué
    documentos puede citar el RAG—, pero solo la primera vez que lo dice.
    """
    return bool(nuevos_otros) or procedimiento_nuevo or any(
        getattr(del_turno, c) is not None for c in _CAMPOS_CLINICOS
    )


def _texto_de(action: Action, *, semilla: str, recientes: list[str],
              nombre: str | None, preocupante: bool, con_acuse: bool = True,
              slot_respondido: str | None = None,
              acumulado: Symptoms | None = None,
              del_turno: Symptoms | None = None,
              con_prefijo: bool = False,
              estado: CallState | None = None) -> str:
    """Traduce la acción del guion a lo que se dice. Todo determinista.

    `con_acuse=False` suprime el acuse de recibo. Se usa cuando el turno ya se
    reconoció por otra vía —o cuando no hay nada que acusar—: decir "vale,
    anotado" después de un turno del que no se extrajo nada afirma algo falso, y
    en la llamada que motivó este cambio el agente lo dijo cuatro veces seguidas.

    `con_prefijo=True` avisa de que quien llama ya puso delante un
    reconocimiento propio (el acuse de un síntoma fuera de catálogo, el
    procedimiento que contó el paciente, la respuesta a una pregunta clínica).
    Sin ese aviso, la repregunta añadiría el suyo y saldrían dos apilados.
    """
    if action.kind == "ofrecer_salida":
        return phrasing.ofrecer_salida(semilla, recientes)
    if action.kind == "confirmar":
        # Fase CONFIRMACIÓN: el agente ya dijo lo suyo y le devuelve el turno al
        # paciente. Se emite como "la siguiente pregunta", igual que las del
        # guion, para que las ramas de arriba —la pregunta clínica, sobre todo—
        # puedan responder PRIMERO y pegar esto después. Tratarlo como una rama
        # aparte y temprana era el bug: ante "¿qué cuidados debo tener con la
        # herida?" el agente contestaba "¿hay algo más antes de despedirnos?",
        # que desde fuera es no escuchar.
        return phrasing.confirmar_cierre(
            semilla, recientes,
            escalado=bool(estado and estado.guion_entregado),
        )
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
        # Una repregunta sin acuse salía pelada: el paciente decía algo, no se le
        # pudo sacar el dato, y recibía la reformulación cerrada sin una palabra
        # sobre lo que acababa de contar. Desde fuera eso es no escuchar, y es el
        # caso que motivó este cambio ("me operaron de cesárea" → "Sin
        # termómetro: ¿lo ha sentido como fiebre, sí o no?").
        #
        # `ACUSE_ESCUCHADO` reconoce haber OÍDO sin afirmar haber anotado nada,
        # que es lo único cierto aquí: los ACUSES normales ("vale, anotado")
        # serían falsos. No se pone cuando ya hay un prefijo delante
        # (`con_prefijo`) — el reconocimiento ya lo dio el prefijo, y encadenar
        # dos en voz es el párrafo que nadie escucha entero.
        if action.kind == "repreguntar" and not con_prefijo:
            return f"{phrasing.acuse_escuchado(semilla, recientes)} {pregunta}"
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
    # La pregunta que el GUION hizo, con lo que se dijo de respaldo. Manda la
    # canónica porque el redactor puede reformular la frase, y de ella depende
    # cómo se lee un "sí" o un "no" pelados (`nlu/polaridad.py`). Vale para las
    # dos rutas: es también el contexto que se le da al LLM.
    pregunta_previa = estado.ultima_pregunta or (
        prior[-1].final_response if prior else phrasing.APERTURA)

    # El paciente no dijo NADA. Se cortocircuita el LLM igual que en la ruta de
    # emergencia y por la misma razón de fondo: no hay nada que extraer del
    # silencio, y gastar ~2,5 s de modelo en él solo alarga la espera de alguien
    # que ya no está contestando.
    silencio = intent_nlu.classify(text) == "silencio"
    if silencio:
        extraccion = Extraction(symptoms=Symptoms(), intent="silencio")
    else:
        extraccion = await llm.extract(slot=slot, question=pregunta_previa, utterance=text)
    if extraccion.usage.tokens_out:
        llm_calls += 1
    tokens_in += extraccion.usage.tokens_in
    tokens_out += extraccion.usage.tokens_out

    # Ventana posterior a un guion crítico: una declaración de automedicación
    # ("me voy a tomar un metronidazol") no matchea el clasificador de
    # preguntas —es una afirmación, no una pregunta— y sin esto se acusaba de
    # recibo sin más y se colgaba sin verificar nada contra el RAG. Se
    # reclasifica solo aquí: `guion_entregado` únicamente se enciende cuando ya
    # se entregó un guion de seguridad (ver `script.py`), así que esto no toca
    # ninguna llamada verde ni amarilla.
    if (extraccion.intent == "respuesta"
            and estado.guion_entregado
            and estado.phase in (Phase.ESCALAMIENTO, Phase.CONFIRMACION)
            and intent_nlu.menciona_automedicacion(text)):
        extraccion.intent = "pregunta_clinica"

    del_turno = extraccion.symptoms
    # Antes de fusionar: lo que el paciente cuenta fuera del guion y todavía no
    # estaba anotado. Solo eso se le reconoce en voz alta y solo eso cuenta como
    # progreso — repetirle "me apunto lo de la visión borrosa" en cada turno es
    # tan robótico como no decírselo nunca.
    nuevos_otros = [o for o in del_turno.other if o not in acumulado.other]

    # De qué dice el paciente que lo operaron. Determinista y fuera de `Symptoms`
    # a propósito: no es un síntoma —no lo valora `decision/rules.py`— sino un
    # hecho sobre la llamada, como el silencio. La ficha sigue mandando; esto
    # decide si se le reconoce en voz alta y si el RAG puede responder sobre ese
    # procedimiento. Ver `nlu/procedimiento.py`.
    proc_dicho = procedimiento_nlu.detectar(lexicon.normalize(text))
    proc_nuevo = proc_dicho is not None and proc_dicho != estado.procedimiento_dicho
    # Vale el del turno o el que ya se dijo antes: el guard del RAG tiene que
    # seguir en pie en los turnos siguientes al que lo contó.
    proc_vigente = proc_dicho or estado.procedimiento_dicho
    discrepa = (proc_vigente is not None
                and not procedimiento_nlu.coincide(proc_vigente, procedimiento))

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
    progreso = _aporto_dato(del_turno, nuevos_otros, proc_nuevo)

    # `emergencia` distingue las 6 banderas del 123 de la vía de enfermería: ante
    # una emergencia real, retener al paciente en la línea compite con la llamada
    # que de verdad importa, así que ahí sí se cuelga rápido.
    emergencia = decision.escalation_action == "emergencia_123"
    # En fase de cierre, "no, nada más, gracias" significa exactamente lo mismo
    # que una despedida formal: la última pregunta fue "¿algo más?", y esa es su
    # respuesta natural. Sin esto, solo el regex de despedida sacaba de
    # CONFIRMACION y el agente encadenaba preguntas de cierre en bucle con un
    # paciente que ya había dicho que no.
    quiere_colgar = extraccion.intent == "despedida" or (
        estado.phase in (Phase.CONFIRMACION, Phase.ESCALAMIENTO)
        and intent_nlu.niega_mas_temas(text)
    )

    # El paciente contesta a la oferta de enfermera. Hasta ahora nadie leía esa
    # respuesta: el guion ofrecía, repetía la oferta al turno siguiente con otra
    # formulación y cerraba a los cinco turnos pasara lo que pasara. Medido en una
    # llamada real, dos ofertas seguidas a alguien que estaba contestando todo, y
    # su "No." ignorado.
    acepta_salida = declina_salida = False
    if estado.espera_respuesta_salida and not silencio:
        polar = lexicon.respuesta_polar(text)
        acepta_salida = polar is True or intent_nlu.contiene_despedida(text)
        declina_salida = polar is False

    action = script.next_action(estado, acumulado, escalar=critico, repetido=repetido,
                                emergencia=emergencia,
                                quiere_colgar=quiere_colgar or acepta_salida,
                                silencio=silencio, reanudar=declina_salida,
                                progreso=progreso,
                                max_silencios=get_settings().silence_max_attempts)
    recientes = _aperturas_recientes(prior)
    evidence: RagResult | None = None

    # Si este turno cierra la llamada, la política de incertidumbre ya aplica: una
    # llamada que no se pudo evaluar no se despide como si el paciente estuviera
    # bien. `close_conversation` lo reevaluaba igual y creaba la alerta, pero el
    # texto hablado se había generado antes y decía "que siga bien" mientras el
    # sistema escalaba a enfermería por detrás. `final=True` solo puede igualar o
    # subir el nivel: `merge_symptoms` es monótono.
    # `not estado.guion_entregado` evita el bucle: tras entregar el guion,
    # `final=True` volvería a dar CRÍTICO en cada intento de cierre y el paciente
    # oiría el guion de seguridad completo dos y tres veces. Antes el guard
    # miraba `phase is not ESCALAMIENTO`, y fallaba porque tras entregar el guion
    # la fase avanza a CONFIRMACION en el turno siguiente — medido en una llamada
    # real: el guion salió entero otra vez después del "vamos a colgar aquí". El
    # flag persiste en `CallState`, la fase no.
    # `not silencio`: el cierre por silencio no habla con nadie. Entregarle el
    # guion de seguridad a una línea muda es tiempo de TTS regalado, y la política
    # de incertidumbre se aplica igual —con su alerta— dentro de
    # `close_conversation`, que corre unas líneas más abajo.
    if (action.kind == "cerrar" and not critico and not silencio
            and not estado.guion_entregado):
        decision = engine.evaluate(acumulado, final=True)
        critico = decision.risk_level == "CRÍTICO"
        if critico:
            # El guion de seguridad manda sobre la despedida neutra.
            action = Action(kind="escalar", phase=Phase.ESCALAMIENTO)

    if silencio:
        # Escalera de presencia: sondear → avisar → colgar. La lleva `script.py`;
        # aquí solo se le pone voz. Determinista, sin modelo: no hay nada que
        # interpretar en un silencio.
        semilla = conv.id + str(len(prior))
        if action.kind == "cerrar":
            final = phrasing.CIERRE_SIN_RESPUESTA
        else:
            # Se repite la pregunta pendiente entera detrás del sondeo. Quien no
            # contestó puede no haberla oído nunca, así que darla por hecha
            # dejaría el slot muerto sin haberlo intentado de verdad.
            pendiente = action.slot or estado.slot_actual
            pregunta = (phrasing.pregunta(pendiente, semilla, recientes)
                        if pendiente else phrasing.PREGUNTA_ABIERTA)
            final = f"{phrasing.sondeo(action.intento, semilla, recientes)} {pregunta}"
    elif critico and action.kind == "escalar":
        # Primer turno crítico: se entrega el guion de seguridad completo.
        # Ruta crítica: no pasa por el modelo ni por el RAG.
        final = decision.safety_script or phrasing.cierre(nombre, escalado=True)
    elif action.kind == "cerrar" and (critico or estado.guion_entregado
                                      or estado.phase is Phase.ESCALAMIENTO):
        # `guion_entregado` cubre el escalamiento que nace de la política de
        # incertidumbre al cerrar: ese `no_se_pudo_evaluar` no vive en `acumulado`
        # —se sintetiza con `final=True`—, así que `critico` vuelve a ser False al
        # turno siguiente y la llamada se despedía con un "que siga bien" después
        # de haber escalado a enfermería. El flag no se pierde al avanzar de fase,
        # que era el agujero de mirar solo `phase is ESCALAMIENTO` (se conserva
        # como respaldo para conversaciones guardadas antes del flag).
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
    elif extraccion.intent == "ininteligible":
        final = phrasing.NO_ENTENDI
    elif action.kind == "cerrar" and extraccion.intent != "pregunta_clinica":
        # El paciente se despidió, o el guion se agotó/atascó: cierre verbatim
        # —pre-sintetizable y con las instrucciones de cuándo llamar—. La
        # excepción es que el turno de cierre traiga una pregunta clínica:
        # responderla ANTES de despedirse es escuchar, y eso baja a la ruta del
        # redactor con la despedida como texto obligatorio.
        final = phrasing.cierre(nombre, escalado=decision.risk_level != "NORMAL")
    else:
        # --- ruta del redactor: el guion decide, el LLM escribe --------------
        # Todo lo conversacional pasa por aquí (respuestas del tamizaje,
        # preguntas clínicas y administrativas, saludos, confirmación de
        # cierre). El LLM ve el historial y redacta el turno completo; si
        # falla, expira o su salida no pasa la aduana de `composer.valida`,
        # el turno sale por las plantillas de siempre (`_texto_fallback`).
        semilla = conv.id + str(len(prior))
        alarma = (otros_sintomas.senales(nuevos_otros)
                  if extraccion.intent == "pregunta_clinica" else [])
        pertinente = False
        if extraccion.intent == "pregunta_clinica" and not discrepa and not alarma:
            # --- E: recuperación de evidencia, EN CUALQUIER FASE -------------
            # Incluida CONFIRMACION: "¿cuándo puedo volver a jugar fútbol?"
            # preguntado durante el cierre merece respuesta, no otra pregunta
            # de despedida. Antes de redactar se comprueba que lo recuperado
            # responda de verdad la pregunta: la similitud vectorial no lo dice
            # (ver rag/retrieve.py), y sin este paso se contestaban protocolos
            # inventados citando documentos sin relación.
            evidence = rag.retrieve(
                session,
                build_query(text, procedure=procedimiento,
                            day_postop=acumulado.day_postop),
                procedure=procedimiento,
            )
            pertinente = evidence.has_evidence and await llm.pregunta_es_del_dominio(
                question=text, evidence=evidence.answer)
            if evidence.has_evidence and not pertinente:
                log.info("evidencia recuperada pero no pertinente para: %r", text[:60])
                evidence.has_evidence = False
        elif extraccion.intent == "pregunta_clinica" and discrepa:
            # El corpus está indexado POR procedimiento: recuperar con la ficha
            # discrepante solo puede citar documentos de la otra cirugía — una
            # afirmación falsa CON fuente. Ni RAG ni citas; el redactor (o el
            # fallback) se abstiene con la nota correspondiente.
            log.warning("RAG omitido: el paciente refiere %r y la ficha dice %r",
                        proc_vigente, procedimiento)

        # La pregunta canónica del guion, con la misma rotación que usaría el
        # fallback: es la restricción que el redactor no puede cambiar.
        pregunta_guion = _texto_de(action, semilla=semilla, recientes=recientes,
                                   nombre=nombre,
                                   preocupante=decision.risk_level != "NORMAL",
                                   con_acuse=False, con_prefijo=True, estado=estado)
        anterior = estado.slot_actual
        ctx = composer.build_context(
            prior=prior, text=text, intent=extraccion.intent,
            objetivo=action.kind, pregunta_guion=pregunta_guion,
            acumulado=acumulado, riesgo=decision.risk_level,
            evidencia=(evidence.answer if evidence is not None
                       and evidence.has_evidence else None),
            procedimiento=procedimiento, nombre=nombre,
            discrepa=discrepa, proc_vigente=proc_vigente,
            alarma_consultada=alarma,
            slot_perdido=bool(
                action.kind == "preguntar" and anterior
                and anterior != action.slot
                and getattr(acumulado, script.SLOT_FIELD[anterior]) is None),
        )
        # El redactor es el MISMO adaptador del turno salvo que se configure
        # otro (COMPOSE_PROVIDER): así los tests que sustituyen `get_llm` nunca
        # tocan red, y en producción no se crea un cliente extra.
        redactor = get_compose_llm() if get_settings().compose_provider else llm
        compuesto = None
        compose_fn = getattr(redactor, "compose_reply", None)
        if compose_fn is not None:
            try:
                compuesto = await compose_fn(ctx=ctx)
            except Exception as exc:  # noqa: BLE001
                log.warning("compose_reply lanzó (%s: %s): caen las plantillas",
                            type(exc).__name__, exc)
                compuesto = None
        if compuesto is not None and composer.valida(compuesto[0], ctx):
            final, uso = compuesto
            if uso.tokens_out:
                llm_calls += 1
            tokens_in += uso.tokens_in
            tokens_out += uso.tokens_out
            # Si el redactor terminó absteniéndose, no se citan fuentes: "no
            # tengo información" con cuatro citas debajo es incoherente.
            if evidence is not None and (not evidence.has_evidence
                                         or es_abstencion(final)):
                evidence = None
        else:
            final, evidence, usos = await _texto_fallback(
                llm=llm, conv=conv, prior=prior, text=text,
                extraccion=extraccion, action=action, estado=estado,
                acumulado=acumulado, del_turno=del_turno, decision=decision,
                nombre=nombre, procedimiento=procedimiento,
                recientes=recientes, discrepa=discrepa,
                proc_vigente=proc_vigente, proc_nuevo=proc_nuevo,
                proc_dicho=proc_dicho, nuevos_otros=nuevos_otros,
                progreso=progreso, evidence=evidence, pertinente=pertinente)
            for uso in usos:
                if uso.tokens_out:
                    llm_calls += 1
                tokens_in += uso.tokens_in
                tokens_out += uso.tokens_out

    # Saludar, pedir que se repita la pregunta o decir algo social no es intentar
    # contestarla: esos turnos no gastan reintentos del slot. Contar de qué lo
    # operaron tampoco: el paciente no esquivó la pregunta, corrigió el registro,
    # y cobrárselo como un intento fallido le acorta el margen para contestar de
    # verdad. Misma doctrina que ya documenta `script.apply`.
    nuevo_estado = script.apply(
        estado, action, acumulado, hash_turno=hash_turno, progreso=progreso,
        intento_real=(extraccion.intent not in ("saludo", "meta", "social")
                      and not proc_nuevo),
        silencio=silencio,
        procedimiento_dicho=proc_dicho,
        # El guion de seguridad se entregó en ESTE turno: queda registrado en el
        # estado para que ningún turno posterior lo repita (ver el guard de
        # `evaluate(final=True)` más arriba).
        guion_entregado=critico and action.kind == "escalar",
        # Lo que el guion pidió preguntar, para poder leer la respuesta el turno
        # que viene. Se recalcula con la MISMA semilla y las mismas aperturas
        # recientes que usó `_texto_de`, así que sale la variante que de verdad se
        # dijo; y se guarda la canónica en vez del texto final porque el redactor
        # puede haberla reformulado.
        pregunta_emitida=phrasing.pregunta_emitida(
            action.kind, action.slot, action.intento, action.seguimiento,
            semilla=conv.id + str(len(prior)), usadas=recientes),
        reanudar=declina_salida,
        # Una pregunta clínica real que se contestó no gasta el presupuesto de
        # MAX_TURNOS_CONFIRMACION (ver el comentario en script.py): el tope
        # frena el relleno indefinido, no a alguien preguntando algo de verdad.
        pregunta_respondida=extraccion.intent == "pregunta_clinica",
    )

    # UNKNOWN explícito: el slot que quedó sin respuesta se anota en el turno en
    # que se abandonó, no solo en el `CallState`. `value=None` ya significa "no
    # se sabe" (diseño ternario de `Symptoms`); esto le pone el `status` encima
    # para que el resumen y la consola lo vean como lo que es — un dato que NO
    # se obtuvo — y no como un campo que simplemente falta. El motivo se lee de
    # la traza (`paciente_no_responde` / `conexion_perdida` en las reglas ⇒ sin
    # respuesta; su ausencia ⇒ repreguntas agotadas). Un silencio jamás escribe
    # False: solo puede añadir a esta lista.
    del_turno.unanswered = [
        s for s in nuevo_estado.sin_responder if s not in estado.sin_responder
    ]
    if silencio and action.kind == "cerrar":
        # La llamada se corta con el guion a medias: todo lo que quedaba por
        # resolver queda marcado UNKNOWN, incluido el slot que se estaba
        # preguntando cuando el paciente enmudeció.
        del_turno.unanswered = sorted(
            set(del_turno.unanswered) | set(script.slots_pendientes(estado, acumulado))
        )
    if del_turno.unanswered:
        # El acumulado que va en la respuesta (y en las alertas de abajo) lo
        # refleja ya; la fusión filtra sola los que consigan valor más adelante.
        acumulado = merge_symptoms(acumulado, Symptoms(unanswered=del_turno.unanswered))

    # --- alerta, deduplicada por reglas nuevas -------------------------------
    alert_id = _crear_alerta_si_procede(session, conv, decision, acumulado, text)

    # El paciente dice que lo operaron de otra cosa. No es un hallazgo clínico
    # —no lo puede valorar `decision/rules.py`— pero sí algo que alguien tiene
    # que mirar: si la ficha está mal, TODO lo demás está mal, porque el
    # procedimiento es la clave con la que se filtra la evidencia y con la que se
    # interpretan los síntomas. A diferencia del silencio, esto NO toca
    # `decision.risk_level`: la llamada puede ser clínicamente verde y seguir
    # necesitando que verifiquen el registro.
    if discrepa and proc_vigente:
        alert_id = _crear_alerta_procedimiento(
            session, conv, acumulado, dicho=proc_vigente, ficha=procedimiento
        ) or alert_id

    # El paciente dejó de contestar y la llamada se acaba aquí. No es un hallazgo
    # clínico —no vive en `Symptoms` ni puede ser una regla de `decision/rules.py`,
    # que solo miran síntomas—, pero sí es exactamente lo que alguien tiene que
    # saber: hubo una llamada de seguimiento que se quedó sin nadie al otro lado.
    if silencio and action.kind == "cerrar":
        # Se mete en la traza del turno para que el resumen y la consola lo
        # cuenten como lo que fue, y no como una llamada que terminó normal.
        decision = decision.model_copy(update={
            "triggered_rules": [*decision.triggered_rules, _REGLA_SIN_RESPUESTA],
            "risk_level": decision.risk_level if decision.risk_level != "NORMAL" else "ALTO",
        })
        alert_id = _crear_alerta_sin_respuesta(session, conv, acumulado) or alert_id

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
        phase=str(nuevo_estado.phase),
        slot_actual=nuevo_estado.slot_actual,
    )


# `_TRANQUILIZADOR` y `_VEREDICTO_NORMALIDAD` viven ahora en `composer.py`
# (importadas arriba): el redactor y este fallback comparten las mismas guardas.

_MATIZ_SIN_TRANQUILIZAR = (
    " De todos modos, como me contó otras cosas, prefiero que enfermería le eche "
    "un ojo."
)


async def _texto_fallback(
    *, llm, conv: Conversation, prior: list[Turn], text: str,
    extraccion: Extraction, action: Action, estado: CallState,
    acumulado: Symptoms, del_turno: Symptoms, decision, nombre: str | None,
    procedimiento: str | None, recientes: list[str], discrepa: bool,
    proc_vigente: str | None, proc_nuevo: bool, proc_dicho: str | None,
    nuevos_otros: list[str], progreso: bool,
    evidence: RagResult | None, pertinente: bool,
) -> tuple[str, RagResult | None, list[LLMUsage]]:
    """Las plantillas deterministas de siempre: la degradación elegante.

    Es la cadena que antes vivía inline en `process_turn_async` y era TODO el
    fraseo del agente. Ahora es el respaldo de la ruta del redactor: corre
    cuando el LLM de composición falla, expira, devuelve algo que no pasa
    `composer.valida`, o cuando el adaptador no redacta (mock, ollama 3B).
    Mismo comportamiento que siempre, cero llamadas de composición.

    `evidence`/`pertinente` llegan ya resueltos por la ruta del redactor para
    no recuperar dos veces; aquí solo se redacta desde ellos (`reply_grounded`).
    """
    semilla = conv.id + str(len(prior))
    usos: list[LLMUsage] = []

    if extraccion.intent == "pregunta_administrativa":
        # Se responde el límite y se retoma. Sin esta rama caía en el `else` y la
        # pregunta se quedaba sin contestar: el paciente pedía el horario de
        # visitas y recibía la siguiente pregunta del guion, sin más.
        siguiente = _texto_de(action, semilla=semilla, recientes=recientes,
                              nombre=nombre, preocupante=False, con_acuse=False,
                              con_prefijo=True, estado=estado)
        final = f"{phrasing.ADMINISTRATIVA} {phrasing.volviendo(semilla, recientes)} {siguiente}"
    elif extraccion.intent == "saludo":
        # El paciente saluda: se le devuelve el saludo y se le hace la pregunta
        # pendiente en su forma ABIERTA, no la reformulación cerrada. Un saludo no
        # es una respuesta esquivada — ver `intento_real` en script.apply.
        slot_pendiente = action.slot or estado.slot_actual
        pregunta = (phrasing.pregunta(slot_pendiente, semilla, recientes)
                    if slot_pendiente else phrasing.PREGUNTA_ABIERTA)
        final = f"{phrasing.saludo_de_vuelta(semilla, recientes)} {pregunta}"
    elif extraccion.intent == "meta":
        siguiente = _texto_de(action, semilla=semilla,
                              recientes=recientes, nombre=nombre, preocupante=False,
                              estado=estado)
        final = f"{phrasing.META_REPETIR} {siguiente}"
    elif extraccion.intent == "social":
        siguiente = _texto_de(action, semilla=semilla,
                              recientes=recientes, nombre=nombre, preocupante=False,
                              estado=estado)
        final = f"{phrasing.SOCIAL} {siguiente}"
    elif extraccion.intent == "pregunta_clinica" and discrepa:
        # El corpus no puede responder sobre una cirugía que discrepa de la
        # ficha (ver la ruta del redactor): abstención determinista, sin citas.
        siguiente = _texto_de(action, semilla=semilla, recientes=recientes,
                              nombre=nombre, preocupante=False, con_acuse=False,
                              con_prefijo=True, estado=estado)
        final = (phrasing.abstencion_procedimiento(proc_vigente, procedimiento)
                 + " " + phrasing.transicion_abstencion(semilla, recientes)
                 + " " + siguiente)
    elif extraccion.intent == "pregunta_clinica" and otros_sintomas.senales(nuevos_otros):
        # El paciente contó un síntoma de alarma y preguntó por él en el mismo
        # turno. Eso NO va al RAG: está pidiendo que le valoren su caso, y el
        # corpus no puede responderlo. Ruta determinista, sin modelo.
        siguiente = _texto_de(action, semilla=semilla, recientes=recientes,
                              nombre=nombre, preocupante=False, con_acuse=False,
                              con_prefijo=True, estado=estado)
        final = (phrasing.sintoma_consultado(
            otros_sintomas.senales(nuevos_otros), semilla, recientes)
            + " " + phrasing.volviendo(semilla, recientes) + " " + siguiente)
    elif extraccion.intent == "pregunta_clinica":
        # La respuesta anclada a la evidencia que ya recuperó la ruta del
        # redactor. Dos frases de `reply_grounded` + la siguiente pregunta.
        respuesta, uso = await llm.reply_grounded(
            question=text,
            evidence=(evidence.answer
                      if evidence is not None and evidence.has_evidence else ""),
            patient_context=f"Paciente de {procedimiento or 'cirugía'}.",
        )
        usos.append(uso)
        respuesta = _validar_grounding(respuesta, evidence, riesgo=decision.risk_level)
        # Si se acabó absteniendo, no se citan fuentes: decir "no tengo información"
        # y adjuntar una cita es incoherente, y en la traza parecería que el agente
        # ignoró evidencia que sí tenía.
        if es_abstencion(respuesta):
            evidence = None
            # Sin este puente, la pregunta clínica sin resolver se pegaba justo
            # antes de la siguiente pregunta del guion y sonaba a que el agente
            # ignoraba lo que acababa de decir en vez de retomar el seguimiento.
            # `con_acuse=False`: el puente YA lo da `transicion_abstencion`; sin
            # esto, `_texto_de` le apilaba detrás su propio acuse/reflejo por
            # defecto y el turno encadenaba tres frases de transición seguidas
            # ("Sobre eso no tengo información... Voy a continuar entonces...
            # Un 4, ahí en la mitad...") — medido en una llamada real.
            transicion = phrasing.transicion_abstencion(semilla, recientes)
            siguiente = _texto_de(action, semilla=semilla, recientes=recientes,
                                  nombre=nombre, preocupante=False, con_acuse=False,
                                  con_prefijo=True, estado=estado)
            final = f"{respuesta} {transicion} {siguiente}"
        else:
            # `acumulado`/`del_turno`/`slot_respondido`: un turno puede traer la
            # respuesta del slot Y una pregunta clínica en la misma frase ("un 4,
            # ¿eso es normal?", o sin signos "un 4, con eso puedo hacer
            # ejercicio"). Sin esto, contestar la pregunta hacía perder el
            # reflejo del dato ("Un 4, ahí en la mitad"). Solo aplica aquí, con
            # respuesta real: en la abstención de arriba el puente ya lo da
            # `transicion_abstencion` y no hace falta un segundo acuse detrás.
            siguiente = _texto_de(action, semilla=semilla,
                                  recientes=recientes, nombre=nombre, preocupante=False,
                                  estado=estado, acumulado=acumulado, del_turno=del_turno,
                                  slot_respondido=estado.slot_actual)
            final = f"{respuesta} {siguiente}"
    else:
        preocupante = decision.risk_level != "NORMAL"
        prefijo = phrasing.TERCERO + " " if extraccion.intent == "tercero" else ""
        # Lo que el paciente trajo por su cuenta se reconoce por su nombre ANTES
        # de retomar el guion, y con el reconocimiento delante el acuse genérico
        # sobra (`con_acuse`). Uno solo por turno, no tres apilados: en voz,
        # encadenar "lo anoto como que no me supo decir" + "me apunto lo de la
        # visión borrosa" + la pregunta es un párrafo que nadie escucha entero.
        # Manda lo que el paciente trajo, que es lo que demuestra que se le oyó.
        #
        # El procedimiento va PRIMERO cuando lo acaba de contar: es el dato con
        # más consecuencias de los tres (decide qué puede citar el RAG y si hay
        # que revisarle la ficha), y era el que se caía al vacío — "me operaron
        # de cesárea" recibía la repregunta cerrada de la fiebre, sin más.
        if proc_nuevo:
            if not procedimiento:
                # Llamada sin paciente asociado: no hay ficha contra la que
                # contrastar, así que no se puede decir ni que coincide ni que no.
                reconocimiento = phrasing.procedimiento_anotado(
                    proc_dicho, semilla, recientes)
            elif discrepa:
                reconocimiento = phrasing.procedimiento_discrepa(
                    proc_dicho, procedimiento, semilla, recientes)
            else:
                reconocimiento = phrasing.procedimiento_confirmado(
                    proc_dicho, semilla, recientes)
            prefijo += reconocimiento + " " + phrasing.volviendo(semilla, recientes) + " "
        elif nuevos_otros:
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
                                    con_acuse=not nuevos_otros and not proc_nuevo
                                    and progreso,
                                    slot_respondido=estado.slot_actual,
                                    acumulado=acumulado, del_turno=del_turno,
                                    con_prefijo=bool(prefijo), estado=estado)
    return final, evidence, usos

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

    # Un veredicto de normalidad con el riesgo ya elevado no es una respuesta
    # anclada: es una valoración clínica, y el agente no valora. Se recorta la
    # frase que lo contiene en vez de tirar toda la respuesta, porque el resto
    # suele estar bien anclado. Solo con riesgo elevado: con el cuadro en NORMAL,
    # un "es normal sentir la pierna débil" que viene de la evidencia es la
    # respuesta correcta, y recortarlo dejaba frases incoherentes a medias
    # ("...lo que sugiere que" pegado a la siguiente pregunta del guion).
    if riesgo != "NORMAL" and _VEREDICTO_NORMALIDAD.search(respuesta):
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


def _crear_alerta_sin_respuesta(
    session: Session, conv: Conversation, symptoms: Symptoms
) -> str | None:
    """El paciente dejó de contestar: alerta ALTA, pase lo que pase con el cuadro.

    Espejo de `_crear_alerta_si_procede` pero sin pasar por el motor de decisión:
    aquí no se está valorando un síntoma, se está reportando que una llamada de
    seguimiento se quedó muda. Puede ocurrir con un cuadro NORMAL —el paciente
    contestó dos preguntas bien y luego soltó el teléfono— y aun así alguien tiene
    que volver a intentarlo.

    Se deduplica por su propia regla: una llamada solo puede quedarse sin
    respuesta una vez, y el guardarraíl protege de un reintento del pipeline.
    """
    ya: set[str] = set()
    for a in session.query(Alert).filter(Alert.conversation_id == conv.id).all():
        ya.update(a.triggered_rules or [])
    if _REGLA_SIN_RESPUESTA in ya:
        return None
    alerta = Alert(
        conversation_id=conv.id, patient_id=conv.patient_id,
        risk_level="ALTO", triggered_rules=[_REGLA_SIN_RESPUESTA],
        symptoms=symptoms.model_dump(),
        transcript="El paciente deja de contestar: la llamada se cerró sin respuesta.",
    )
    session.add(alerta)
    session.flush()
    return alerta.id


def abandon_conversation(session: Session, conversation_id: str) -> None:
    """El stream de voz se cayó con la llamada abierta (CONNECTION_LOST).

    Sin esto, colgar de golpe dejaba la conversación `active` para siempre: sin
    resumen, sin alerta, invisible en el reporte. Idempotente a propósito: el
    transporte también se desconecta después de un cierre normal (el agente
    cuelga con `EndFrame`), y en ese caso `close_conversation` ya corrió y aquí
    no hay nada que reportar.

    Cierra vía `close_conversation`, que aplica `evaluate(final=True)`: una
    llamada cortada a mitad de tamizaje sale como `no_se_pudo_evaluar` (con su
    alerta), no como verde.
    """
    conv = session.get(Conversation, conversation_id)
    if conv is None or conv.status == "closed":
        return
    _crear_alerta_conexion_perdida(session, conv)
    from ..summary.service import close_conversation

    close_conversation(session, conversation_id)


def _crear_alerta_conexion_perdida(
    session: Session, conv: Conversation
) -> str | None:
    """La llamada se cortó a medias: alerta ALTA con remediación propia.

    Espejo de `_crear_alerta_sin_respuesta` — mismo patrón, regla distinta y
    texto distinto, porque la acción de enfermería es distinta: aquí primero se
    re-marca; allá se valora a un paciente que estaba en línea y enmudeció.
    """
    ya: set[str] = set()
    for a in session.query(Alert).filter(Alert.conversation_id == conv.id).all():
        ya.update(a.triggered_rules or [])
    if _REGLA_CONEXION_PERDIDA in ya:
        return None
    prior = _prior_turns(session, conv.id)
    acumulado, _ = _acumular(prior)
    alerta = Alert(
        conversation_id=conv.id, patient_id=conv.patient_id,
        risk_level="ALTO", triggered_rules=[_REGLA_CONEXION_PERDIDA],
        symptoms=acumulado.model_dump(),
        transcript=("La conexión de la llamada se perdió y la conversación quedó "
                    "incompleta. Reintentar el contacto con el paciente."),
    )
    session.add(alerta)
    session.flush()
    return alerta.id


def _crear_alerta_procedimiento(
    session: Session, conv: Conversation, symptoms: Symptoms, *,
    dicho: str, ficha: str | None,
) -> str | None:
    """El paciente refiere una cirugía distinta a la de su ficha.

    Espejo de `_crear_alerta_sin_respuesta`: alerta por un hecho de la llamada,
    sin pasar por el motor de decisión, y deduplicada por su propia regla —una
    ficha solo hace falta verificarla una vez, por muchas veces que el paciente
    repita de qué lo operaron.

    El nivel es ALTO porque el procedimiento es la clave de todo lo demás: con él
    se filtra la evidencia que el agente puede citar y con él se interpretan los
    síntomas. Pero, a diferencia del silencio, no se toca el `risk_level` del
    turno: esto no es un deterioro del paciente.
    """
    ya: set[str] = set()
    for a in session.query(Alert).filter(Alert.conversation_id == conv.id).all():
        ya.update(a.triggered_rules or [])
    if _REGLA_PROCEDIMIENTO in ya:
        return None
    alerta = Alert(
        conversation_id=conv.id, patient_id=conv.patient_id,
        risk_level="ALTO", triggered_rules=[_REGLA_PROCEDIMIENTO],
        symptoms=symptoms.model_dump(),
        transcript=(f"El paciente refiere haberse operado de {dicho}; "
                    f"en la ficha figura {ficha or 'sin procedimiento'}. "
                    "Verificar el registro antes de interpretar esta llamada."),
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

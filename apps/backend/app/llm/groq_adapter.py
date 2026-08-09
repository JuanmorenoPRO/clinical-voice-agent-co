"""Adaptador de Groq — `llama-3.3-70b-versatile` (compuerta G3, sucesor vigente).

El puesto de "Llama 3.1 70B vía Groq" de la compuerta G3 quedó sin modelo: Groq
decomisionó el 3.1 y apaga `llama-3.3-70b-versatile` el 16-ago-2026. La nota del
reto permite usar el sucesor vigente del mismo proveedor, y la generación vigente
de Llama que hospeda Groq es la familia Llama 4 (`meta-llama/llama-4-*`). Este
adaptador lee el modelo de `LLM_MODEL`, así que el cambio de sucesor es una
variable de entorno.

Decisiones de implementación, todas verificadas contra la API de Groq:

  - Groq NO ofrece el `format` (gramática) de Ollama. Los modelos `llama-*` no
    soportan el `response_format` `json_schema` nativo (eso es solo para
    `openai/gpt-oss-*`), así que se usa JSON Object Mode
    (`{"type": "json_object"}`) y el esquema se incrusta en el prompt. A cambio,
    el JSON ya no es "imposible de violar por construcción": por eso el parseo y
    `_to_symptoms` van en try/except → `degraded=True`, y por eso las guardas de
    severidad (`_MENCIONA_SLOT`, `_ES_PREGUNTA`, `grounded_in_evidence`) se
    conservan intactas del adaptador de Ollama: son la red que de verdad impide
    alucinar un CRÍTICO.
  - El JSON Object Mode de Groq exige que la palabra "json" esté en la
    conversación; los prompts y el esquema incrustado la incluyen.
  - `temperature=0` para extracción. Los sanitizadores de registro ("usted") y
    muletillas existen porque el 3B los ignoraba; un 70B obedece mejor, pero
    mantenerlos no cuesta y cubre el caso contrario.
  - Timeout duro igual que en Ollama: si Groq tarda, se degrada, nunca se lanza.

El contrato lo define `LLMAdapter` en `adapter.py`; nada fuera de `app/llm/`
conoce al proveedor (ADR-002).
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time

from groq import AsyncGroq

from ..config import get_settings
from ..nlu import intent as intent_rules
from ..nlu import lexicon
from ..nlu.merge import merge_symptoms
from ..prompts_loader import load_prompt
from .adapter import ComposeContext, Extraction, LLMUsage
from .extraction_schema import SLOT_FIELDS, SLOT_KEY, schema_for_slot
from .ollama_adapter import (
    ABSTENCION,
    _APERTURA_POSTIZA,
    _SIN_CONTENIDO_CLINICO,
    _SYSTEM_EXTRACT,
    _SYSTEM_PERTINENCIA,
    _SYSTEM_REPLY,
    _bandera_roja,
    _nombres_propios_presentes,
    _recortar,
    _resuelto,
    _to_symptoms,
    a_usted,
    grounded_in_evidence,
    sin_frase_cortada,
    sin_muletillas,
    sin_tartamudeo,
)

log = logging.getLogger(__name__)

# Mismo esquema de pertinencia que la guarda del RAG (ollama_adapter): "c" y no
# "r" por el motivo medido ahí — la clave entra en el contexto y arrastra al modelo.
_SCHEMA_PERTINENCIA = {
    "type": "object",
    "required": ["c"],
    "properties": {"c": {"type": "string", "enum": ["cuidado", "otro"]}},
}


def _extraer_json(texto: str) -> dict:
    """Saca el primer objeto JSON de la respuesta.

    En modo no estricto Groq puede envolver el JSON en fences o añadir texto
    antes/después; se recorta hasta la primera `{` y la última `}`.
    """
    texto = texto.strip()
    inicio = texto.find("{")
    fin = texto.rfind("}")
    if inicio == -1 or fin <= inicio:
        raise ValueError(f"no hay JSON en la respuesta: {texto[:120]!r}")
    return json.loads(texto[inicio : fin + 1])


def formatear_historial(historial: list[dict[str, str]]) -> str:
    """El historial como diálogo plano para el prompt del redactor."""
    if not historial:
        return "(la llamada apenas empieza)"
    lineas = []
    for turno in historial:
        quien = "Paciente" if turno.get("rol") == "paciente" else "Agente"
        lineas.append(f"{quien}: {turno.get('texto', '')}")
    return "\n".join(lineas)


def prompt_de_composicion(ctx: ComposeContext) -> str:
    """Rellena la plantilla versionada `prompts/compose_turn.md`."""
    evidencia = (
        "EVIDENCIA de los documentos del hospital (responde la pregunta del "
        f"paciente SOLO desde aquí):\n{_recortar(ctx.evidencia)}\n"
        if ctx.evidencia
        else ""
    )
    notas = ("NOTAS del sistema para este turno:\n"
             + "\n".join(f"- {n}" for n in ctx.notas) + "\n") if ctx.notas else ""
    return load_prompt("compose_turn").format(
        historial=formatear_historial(ctx.historial),
        utterance=ctx.utterance,
        sintomas=ctx.sintomas or "(todavía nada)",
        procedimiento=ctx.procedimiento or "(sin registrar)",
        evidencia=evidencia,
        notas=notas,
        pregunta_guion=ctx.pregunta_guion,
    )


# Comillas y fences con los que un modelo envuelve su salida cuando se le pide
# "solo el texto". Se pelan antes de los saneadores.
_ENVOLTORIO = re.compile(r'^\s*["“”\'`]+|["“”\'`]+\s*$')


class GroqAdapter:
    def __init__(self) -> None:
        s = get_settings()
        self._model = s.llm_model
        self._timeout = s.llm_timeout_s
        self._reply_timeout = s.llm_reply_timeout_s
        self._compose_timeout = s.llm_compose_timeout_s
        self._api_key = s.groq_api_key
        self._clients: dict[int, AsyncGroq] = {}

    @property
    def _client(self) -> AsyncGroq:
        """Un cliente por event loop (httpx se liga al loop de creación)."""
        try:
            key = id(asyncio.get_running_loop())
        except RuntimeError:
            key = 0
        if key not in self._clients:
            self._clients[key] = AsyncGroq(api_key=self._api_key)
        return self._clients[key]

    async def _chat(
        self,
        *,
        messages: list[dict],
        temperature: float,
        max_tokens: int,
        response_format: dict | None,
        timeout: float,
    ) -> tuple[str, int, int] | None:
        """Una llamada a Groq; (content, input_tokens, output_tokens) o None."""
        kwargs: dict = {
            "model": self._model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_format is not None:
            kwargs["response_format"] = response_format
        resp = await asyncio.wait_for(
            self._client.chat.completions.create(**kwargs), timeout=timeout
        )
        content = (resp.choices[0].message.content or "").strip()
        return content, resp.usage.prompt_tokens or 0, resp.usage.completion_tokens or 0

    # --- extracción ----------------------------------------------------------

    async def extract(
        self, *, slot: str | None, question: str, utterance: str
    ) -> Extraction:
        intent = intent_rules.classify(utterance)
        base = lexicon.extract(utterance, slot)

        if (
            slot is None
            or _resuelto(base, slot)
            or _bandera_roja(base)
            or intent in _SIN_CONTENIDO_CLINICO
        ):
            return Extraction(
                symptoms=base,
                intent=intent,
                usage=LLMUsage(model=self._model, purpose="extract"),
            )

        t0 = time.perf_counter()
        try:
            result = await self._chat(
                messages=[
                    {"role": "system", "content": _SYSTEM_EXTRACT},
                    {
                        "role": "user",
                        "content": (
                            f'Pregunta: "{question}"\nPaciente: <<<{utterance}>>>\n'
                            "Devuelve únicamente el JSON de este esquema "
                            "(ningún otro texto):\n"
                            + json.dumps(schema_for_slot(slot), ensure_ascii=False)
                        ),
                    },
                ],
                temperature=0,
                max_tokens=96,
                response_format={"type": "json_object"},
                timeout=self._timeout,
            )
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "extract degradado (slot=%s): %s: %s", slot, type(exc).__name__, exc
            )
            return Extraction(
                symptoms=base,
                intent=intent,
                degraded=True,
                usage=LLMUsage(
                    model=self._model,
                    purpose="extract",
                    latency_ms=(time.perf_counter() - t0) * 1000,
                ),
            )
        if result is None:
            return Extraction(
                symptoms=base,
                intent=intent,
                degraded=True,
                usage=LLMUsage(model=self._model, purpose="extract"),
            )
        content, inp, out = result
        usage = LLMUsage(
            model=self._model,
            purpose="extract",
            latency_ms=(time.perf_counter() - t0) * 1000,
            tokens_in=inp,
            tokens_out=out,
        )
        try:
            data = _extraer_json(content)
            data = _normalizar_slot(data, slot)
            sym = _to_symptoms(data, slot, utterance)
        except (ValueError, KeyError, TypeError, json.JSONDecodeError):
            return Extraction(symptoms=base, intent=intent, degraded=True, usage=usage)

        return Extraction(
            symptoms=merge_symptoms(base, sym), intent=intent, usage=usage
        )

    # --- respuesta anclada a evidencia ---------------------------------------

    async def reply_grounded(
        self, *, question: str, evidence: str, patient_context: str
    ) -> tuple[str, LLMUsage]:
        if not evidence.strip():
            return ABSTENCION, LLMUsage(model=self._model, purpose="reply")

        user = (
            f"{patient_context}\n\nEVIDENCIA:\n{_recortar(evidence)}\n\n"
            f"PREGUNTA DEL PACIENTE: {question}"
        )
        t0 = time.perf_counter()
        try:
            result = await self._chat(
                messages=[
                    {"role": "system", "content": _SYSTEM_REPLY},
                    {"role": "user", "content": user},
                ],
                # 256 y no 80: con 80 un 70B corta la respuesta a media frase
                # sin llegar a ningún punto ("...lo que sugiere que"), y el
                # saneador no tenía nada que rescatar. El prompt sigue pidiendo
                # 2 frases; el margen es para que el corte, si llega, caiga
                # después de una frase completa.
                temperature=0.2,
                max_tokens=256,
                response_format=None,
                timeout=self._reply_timeout,
            )
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "reply_grounded degradado (%s: %s) — se abstiene",
                type(exc).__name__,
                exc,
            )
            return ABSTENCION, LLMUsage(
                model=self._model,
                purpose="reply",
                latency_ms=(time.perf_counter() - t0) * 1000,
            )
        if result is None:
            return ABSTENCION, LLMUsage(model=self._model, purpose="reply")
        texto, inp, out = result
        usage = LLMUsage(
            model=self._model,
            purpose="reply",
            latency_ms=(time.perf_counter() - t0) * 1000,
            tokens_in=inp,
            tokens_out=out,
        )
        texto = texto or ABSTENCION
        if not grounded_in_evidence(texto, evidence):
            log.warning(
                "reply_grounded: respuesta sin cifras en la evidencia → abstención"
            )
            return ABSTENCION, usage
        return sin_frase_cortada(sin_tartamudeo(sin_muletillas(a_usted(texto)))), usage

    # --- redacción del turno completo -----------------------------------------

    async def compose_reply(self, *, ctx: ComposeContext) -> tuple[str, LLMUsage] | None:
        """Redacta lo que el agente dice en voz alta, con el guion como restricción.

        `temperature=0.4` a propósito: la variación de fraseo es el objetivo de
        esta llamada, no un riesgo — las decisiones clínicas ya están tomadas y
        la validación dura la hace `agent/composer.py` después. No se aplica
        `sin_muletillas` completo por la misma razón: `_EMOCION_INVENTADA`
        recortaría aperturas legítimas de un 70B que sí vio el historial; solo
        se pelan los vocativos informales (`_APERTURA_POSTIZA`).
        """
        t0 = time.perf_counter()
        try:
            result = await self._chat(
                messages=[
                    {"role": "system", "content": load_prompt("compose_system")},
                    {"role": "user", "content": prompt_de_composicion(ctx)},
                ],
                temperature=0.4,
                max_tokens=220,
                response_format=None,
                timeout=self._compose_timeout,
            )
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "compose_reply degradado (%s: %s) — caen las plantillas",
                type(exc).__name__, exc,
            )
            return None
        if result is None:
            return None
        texto, inp, out = result
        usage = LLMUsage(
            model=self._model,
            purpose="compose",
            latency_ms=(time.perf_counter() - t0) * 1000,
            tokens_in=inp,
            tokens_out=out,
        )
        texto = _ENVOLTORIO.sub("", (texto or "").strip())
        texto = sin_frase_cortada(sin_tartamudeo(a_usted(texto)))
        texto = _APERTURA_POSTIZA.sub("", texto).strip()
        if not texto:
            return None
        texto = texto[0].upper() + texto[1:]
        return texto, usage

    async def pregunta_es_del_dominio(self, *, question: str, evidence: str) -> bool:
        if not _nombres_propios_presentes(question, evidence):
            return False
        try:
            result = await self._chat(
                messages=[
                    {"role": "system", "content": _SYSTEM_PERTINENCIA},
                    {
                        "role": "user",
                        "content": (
                            f"Pregunta: <<<{question}>>>\n"
                            "Devuelve únicamente el JSON de este esquema "
                            "(ningún otro texto):\n"
                            + json.dumps(_SCHEMA_PERTINENCIA, ensure_ascii=False)
                        ),
                    },
                ],
                temperature=0,
                max_tokens=16,
                response_format={"type": "json_object"},
                timeout=self._timeout * 2,
            )
            if result is None:
                return False
            content, _, _ = result
            return _extraer_json(content).get("c") == "cuidado"
        except Exception as exc:  # noqa: BLE001
            # Fail-open y no fail-closed: un hipo de red aquí convertía una
            # pregunta legítima en abstención seca. El precheck determinista de
            # nombres propios ya corrió, y aguas abajo siguen las guardas de
            # grounding; dejar pasar la evidencia recuperada es el menor de los
            # dos males.
            log.warning(
                "pertinencia degradada (%s): se asume que sí responde",
                type(exc).__name__,
            )
            return True

    async def summarize(self, *, system_prompt: str, user_prompt: str) -> str:
        try:
            result = await self._chat(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.2,
                max_tokens=300,
                response_format=None,
                timeout=self._reply_timeout,
            )
            return (result[0] or "").strip() if result else ""
        except Exception as exc:  # noqa: BLE001
            log.warning("summarize degradado (%s: %s)", type(exc).__name__, exc)
            return ""


def _normalizar_slot(data: dict, slot: str) -> dict:
    """El contrato del pipeline usa la clave `v`; normaliza lo que devuelva Groq.

    Con el esquema incrustado en el prompt, lo normal es `{"v": "..."}`. Se acepta
    también `{campo_largo: ...}` / `{campo_corto: ...}` en caso de deriva.
    """
    if SLOT_KEY in data:
        return data
    short_field, _ = SLOT_FIELDS[slot]
    for clave in (short_field, slot):
        if clave in data:
            return {SLOT_KEY: data[clave]}
    return data

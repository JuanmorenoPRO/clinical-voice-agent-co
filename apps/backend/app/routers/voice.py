"""Señalización WebRTC del modo de voz (SmallWebRTC, Pipecat 1.x).

- POST /voice/offer  : recibe el SDP offer del navegador, arranca el bot de voz
                       en segundo plano y devuelve el SDP answer.
- PATCH /voice/offer : candidatos ICE (trickle).
- GET  /voice/status : diagnóstico (deps + claves).

Pipecat se importa PEREZOSAMENTE dentro de los handlers: así la app arranca
aunque las deps de voz no estén instaladas (modo texto). Si faltan las claves o
las deps, el endpoint responde un error claro en vez de romperse.
"""
from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks, HTTPException, Request

from ..config import get_settings

router = APIRouter(prefix="/voice", tags=["voice"])

# Singleton del handler de SmallWebRTC (se crea en el primer offer).
_handler = None


def _missing_config() -> list[str]:
    s = get_settings()
    missing = []
    if not s.deepgram_api_key:
        missing.append("DEEPGRAM_API_KEY")
    if not s.elevenlabs_api_key:
        missing.append("ELEVENLABS_API_KEY")
    if not s.elevenlabs_voice_id:
        missing.append("ELEVENLABS_VOICE_ID")
    return missing


def _get_handler():
    global _handler
    if _handler is None:
        from pipecat.transports.smallwebrtc.request_handler import SmallWebRTCRequestHandler

        _handler = SmallWebRTCRequestHandler()
    return _handler


@router.get("/status")
def status() -> dict:
    """¿Está listo el modo de voz? Útil para diagnosticar antes de llamar."""
    try:
        import pipecat  # noqa: F401

        installed = True
    except ImportError:
        installed = False
    return {
        "pipecat_installed": installed,
        "missing_config": _missing_config(),
        "ready": installed and not _missing_config(),
    }


@router.post("/offer")
async def offer(request: Request, background_tasks: BackgroundTasks):
    missing = _missing_config()
    if missing:
        raise HTTPException(
            400,
            f"Modo de voz sin configurar. Faltan variables en .env: {', '.join(missing)}. "
            "Elige además una voz nativa es-CO para ELEVENLABS_VOICE_ID (ADR-007).",
        )

    try:
        from ..voice.pipeline import run_bot
        from pipecat.transports.smallwebrtc.request_handler import SmallWebRTCRequest
    except ImportError as exc:  # pragma: no cover
        raise HTTPException(
            503,
            "Las dependencias de voz (Pipecat) no están instaladas. Reconstruye con "
            "INSTALL_VOICE=true: `INSTALL_VOICE=true docker compose up --build`. "
            f"Detalle: {exc}",
        )

    body = await request.json()
    handler = _get_handler()

    async def _on_connection(connection):
        background_tasks.add_task(run_bot, connection)

    answer = await handler.handle_web_request(
        request=SmallWebRTCRequest(**body),
        webrtc_connection_callback=_on_connection,
    )
    return answer


@router.patch("/offer")
async def ice_candidate(request: Request):
    try:
        from pipecat.transports.smallwebrtc.request_handler import SmallWebRTCPatchRequest
    except ImportError as exc:  # pragma: no cover
        raise HTTPException(503, f"Dependencias de voz no instaladas: {exc}")

    body = await request.json()
    await _get_handler().handle_patch_request(SmallWebRTCPatchRequest(**body))
    return {"status": "ok"}

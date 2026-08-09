"""Verificación end-to-end de los dos chats problemáticos (9 de agosto de 2026).

    .venv/bin/python scripts/verify_conversacion.py

Reproduce contra el agente completo —con el LLM real de LLM_PROVIDER— las dos
conversaciones que motivaron la arquitectura híbrida:

  1. El bucle de cierre: "¿Hay algo más...?" infinito ante "no, nada más".
  2. El fútbol: la pregunta "¿cuándo puedo volver a jugar fútbol?" repetida e
     ignorada, respuestas cortadas a media frase, y la despedida que no cerraba.
  3. Regresión de seguridad: el guion CRÍTICO sale verbatim, una sola vez, y sin
     pasar por el redactor.

Usa una base de datos temporal: no toca data/clinical.db. Imprime la transcripción
completa para revisión humana y sale con código 1 si alguna aserción dura falla.
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "apps" / "backend"))

# Base temporal ANTES de importar la app (el engine se crea al importar app.db).
_TMP = tempfile.mkdtemp(prefix="verify_conv_")
os.environ["DATABASE_URL"] = f"sqlite:///{_TMP}/verify.db"

from app.agent.orchestrator import process_turn  # noqa: E402
from app.db import Base, SessionLocal, engine  # noqa: E402
from app.models import Patient  # noqa: E402

Base.metadata.create_all(engine)

FALLOS: list[str] = []


def check(cond: bool, msg: str) -> None:
    marca = "ok" if cond else "FALLO"
    print(f"    [{marca}] {msg}")
    if not cond:
        FALLOS.append(msg)


def turno(session, texto: str, cid: str | None, pid: str | None = None):
    r = process_turn(session, text=texto, conversation_id=cid, patient_id=pid)
    print(f"\n  Paciente: {texto}")
    print(f"  Agente:   {r.response}")
    print(f"            riesgo={r.risk_level} cierre={r.call_ended} "
          f"fuentes={len(r.sources)}")
    check(r.response.rstrip()[-1:] in ".!?…", "la respuesta termina en frase completa")
    return r


def paciente_cadera(session) -> str:
    p = Patient(name="Mauricio González", surgery="Reemplazo de cadera/rodilla",
                extra={"dia_postop": 5})
    session.add(p)
    session.commit()
    return p.id


def escenario_1_bucle() -> None:
    print("\n=== Escenario 1: el bucle de cierre ===")
    with SessionLocal() as s:
        pid = paciente_cadera(s)
        cid = None
        for texto in ("Un 2, apenas se nota.", "No he tenido fiebre.",
                      "Camino bien, sin problema.", "La herida la veo bien.",
                      "Bien, como con normalidad.", "Sí, descanso bien."):
            r = turno(s, texto, cid, pid)
            cid = r.conversation_id
        r = turno(s, "no", cid)
        vistas = {r.response}
        cierres = 0
        while not r.call_ended and cierres < 4:
            r = turno(s, "No, nada más, así está bien, gracias.", cid)
            check(r.response not in vistas, "no repite literalmente la pregunta de cierre")
            vistas.add(r.response)
            cierres += 1
        check(r.call_ended, "la llamada CIERRA tras la negación simple")
        check(cierres <= 2, f"cerró en ≤2 turnos de confirmación (fueron {cierres})")
        check("no_se_pudo_evaluar" not in (r.triggered_rules or []),
              "el guion completo no re-dispara la política de incertidumbre")


def escenario_2_futbol() -> None:
    print("\n=== Escenario 2: la pregunta del fútbol ===")
    with SessionLocal() as s:
        pid = paciente_cadera(s)
        cid = None
        r = turno(s, "un 4, ¿puedo ir ya al gimnasio?", cid, pid)
        cid = r.conversation_id
        check(r.response.rstrip().endswith("?"),
              "responde Y devuelve el turno con una pregunta")
        turno(s, "no he tenido fiebre, ¿puedo realizar ejercicios?", cid)
        turno(s, "camino bien", cid)
        turno(s, "la herida la veo bien", cid)
        turno(s, "bien", cid)
        turno(s, "sí, descanso bien", cid)
        # Pregunta abierta → la pregunta del fútbol, tres veces, con y sin signos.
        r = turno(s, "cuando podria volver a jugar futbol", cid)
        check(len(r.response.split()) > 12,
              "la primera pregunta del fútbol recibe una respuesta de verdad")
        r = turno(s, "¿en cuántos días podría volver a jugar fútbol?", cid)
        check(len(r.response.split()) > 12,
              "la segunda (en fase de cierre) también se responde")
        r = turno(s, "listo, eso era todo, pero ¿es seguro que corra?", cid)
        check(len(r.response.split()) > 8,
              "la pregunta pegada a la despedida no se pierde")
        vueltas = 0
        while not r.call_ended and vueltas < 3:
            r = turno(s, "no, nada más, muchas gracias", cid)
            vueltas += 1
        check(r.call_ended, "la llamada cierra cuando el paciente dice que no hay más")


def escenario_3_critico() -> None:
    print("\n=== Escenario 3: el guion CRÍTICO es intocable ===")
    from app.db import SessionLocal
    from app.models import Turn as TurnModel

    with SessionLocal() as s:
        pid = paciente_cadera(s)
        r = turno(s, "Me duele un berraco, no aguanto", None, pid)
        cid = r.conversation_id
        check(r.risk_level == "CRÍTICO", "el dolor severo escala a CRÍTICO")
        check("123" in r.response, "el guion de seguridad trae la línea 123")
        t = s.query(TurnModel).filter(TurnModel.conversation_id == cid).all()[-1]
        check(t.llm_calls == 0,
              f"la ruta crítica no toca el modelo (llm_calls={t.llm_calls})")
        guion = r.response
        r = turno(s, "Listo, muchas gracias.", cid)
        check(r.response != guion, "el guion NO se repite tras la despedida")
        check(r.call_ended, "la llamada cierra tras confirmar")


if __name__ == "__main__":
    escenario_1_bucle()
    escenario_2_futbol()
    escenario_3_critico()
    print(f"\n{'=' * 60}")
    if FALLOS:
        print(f"{len(FALLOS)} aserciones fallaron:")
        for f in FALLOS:
            print(f"  - {f}")
        sys.exit(1)
    print("Todas las aserciones pasaron.")

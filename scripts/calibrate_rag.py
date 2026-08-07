"""Calibra `RAG_MIN_CONFIDENCE` con preguntas oro y preguntas fuera del corpus.

El umbral heredado (0.55) venía de `voyage-3` y **no transfiere** a `bge-m3`: las
similitudes observadas se apiñan entre 0.72 y 0.80, así que 0.55 deja pasar
cualquier cosa. El síntoma es concreto y se vio en pruebas: preguntado por un
protocolo inventado, el agente respondía citando un documento de apendicitis sin
relación — una afirmación falsa *con fuente*, que es peor que no responder.

    .venv/bin/python scripts/calibrate_rag.py

Barre umbrales y reporta, para cada uno, cuántas preguntas con respuesta en el
corpus se contestan (recall) y cuántas preguntas ajenas se rechazan
correctamente (especificidad). El buen umbral es el mayor que no pierde recall.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "apps" / "backend"))

from app.db import SessionLocal  # noqa: E402
from app.llm.factory import get_llm  # noqa: E402
from app.rag import retrieve as rag  # noqa: E402
from app.config import get_settings  # noqa: E402
from app.rag.embeddings import build_query  # noqa: E402

# Preguntas que el corpus SÍ responde. Salen de los temas que cubren las guías de
# cuidado postoperatorio de cada procedimiento.
ORO: list[tuple[str, str]] = [
    ("¿Cuándo me puedo bañar después de la cirugía?", "Colecistectomía"),
    ("¿Cuándo me quitan los puntos?", "Colecistectomía"),
    ("¿Por qué me duele el hombro después de la operación de vesícula?", "Colecistectomía"),
    ("¿Qué cuidados debo tener con la herida?", "Apendicectomía"),
    ("¿Cuándo puedo volver a hacer ejercicio?", "Apendicectomía"),
    ("¿Qué señales de alarma debo vigilar?", "Apendicectomía"),
    ("¿Qué debo comer después de la cirugía?", "Colectomía"),
    ("¿Cuándo puedo volver a trabajar?", "Colectomía"),
    ("¿Cómo cuido la bolsa de colostomía?", "Colectomía"),
    ("¿Cuándo puedo apoyar la pierna operada?", "Reemplazo de cadera/rodilla"),
    ("¿Qué ejercicios de rehabilitación debo hacer?", "Reemplazo de cadera/rodilla"),
    ("¿Puedo subir escaleras después del reemplazo?", "Reemplazo de cadera/rodilla"),
    ("¿Es normal la hinchazón después de la cirugía?", "Apendicectomía"),
    ("¿Cuándo debo llamar al médico por fiebre?", "Colecistectomía"),
    ("¿Cómo prevenir coágulos después de la operación?", "Reemplazo de cadera/rodilla"),
]

# Preguntas que el corpus NO responde. Mezclan protocolos inventados, temas de
# otra especialidad y consultas administrativas: el agente debe rechazarlas todas.
FUERA: list[tuple[str, str]] = [
    ("¿Cuándo es el control del protocolo Turquesa?", "Apendicectomía"),
    ("¿Qué grado de la escala Zafiro tengo?", "Colecistectomía"),
    ("¿Cuánto cuesta la consulta de control?", "Colectomía"),
    ("¿Puedo viajar en avión a Marte?", "Apendicectomía"),
    ("¿Qué marca de carro me recomienda?", "Colecistectomía"),
    ("¿Cómo se trata la diabetes tipo 1?", "Colectomía"),
    ("¿Cuál es el horario de visitas del hospital?", "Reemplazo de cadera/rodilla"),
    ("¿Me cubre la EPS el transporte?", "Apendicectomía"),
    ("¿Qué dice el protocolo Esmeralda sobre la dieta?", "Colectomía"),
    ("¿Cómo se hace una cesárea?", "Colecistectomía"),
]



def main() -> int:
    umbral_actual = get_settings().rag_min_confidence
    with SessionLocal() as session:
        print(f"Recuperando {len(ORO)} preguntas con respuesta y "
              f"{len(FUERA)} ajenas al corpus…\n")
        scores_oro, scores_fuera = [], []
        decisiones_oro, decisiones_fuera = [], []
        for pregunta, proc in ORO:
            r = rag.retrieve(session, build_query(pregunta, procedure=proc), procedure=proc,
                             raw_question=pregunta)
            scores_oro.append((r.confidence, pregunta, r.sources[0].document if r.sources else "—"))
            decisiones_oro.append(r.has_evidence and asyncio.run(
                get_llm().evidencia_responde(question=pregunta, evidence=r.answer)))
        for pregunta, proc in FUERA:
            r = rag.retrieve(session, build_query(pregunta, procedure=proc), procedure=proc,
                             raw_question=pregunta)
            scores_fuera.append((r.confidence, pregunta,
                                 r.sources[0].document if r.sources else "—"))
            decisiones_fuera.append(r.has_evidence and asyncio.run(
                get_llm().evidencia_responde(question=pregunta, evidence=r.answer)))

    print("Preguntas CON respuesta en el corpus (queremos confianza alta):")
    for s, q, d in sorted(scores_oro, reverse=True):
        print(f"  {s:.4f}  {q[:52]:<52} {d[:36]}")
    print("\nPreguntas AJENAS al corpus (queremos confianza baja):")
    for s, q, d in sorted(scores_fuera, reverse=True):
        print(f"  {s:.4f}  {q[:52]:<52} {d[:36]}")

    # Lo que importa no es la puntuación sino la DECISIÓN del sistema completo,
    # que incluye el filtro léxico de pertinencia además del umbral.
    responde = sum(1 for r in decisiones_oro if r)
    rechaza = sum(1 for r in decisiones_fuera if not r)
    print(f"\nDecisión real del sistema (umbral {umbral_actual} + juicio de pertinencia):")
    print(f"  responde preguntas del corpus : {responde}/{len(ORO)}")
    print(f"  rechaza preguntas ajenas      : {rechaza}/{len(FUERA)}")
    print(f"  aciertos                      : {responde + rechaza}/{len(ORO) + len(FUERA)}")

    fallos_oro = [q for (_, q, _), ok in zip(scores_oro, decisiones_oro) if not ok]
    fallos_fuera = [q for (_, q, _), ok in zip(scores_fuera, decisiones_fuera) if ok]
    if fallos_oro:
        print("\n  ✗ debería responder y se abstuvo:")
        for q in fallos_oro:
            print(f"      {q}")
    if fallos_fuera:
        print("\n  ✗ debería abstenerse y respondió:")
        for q in fallos_fuera:
            print(f"      {q}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

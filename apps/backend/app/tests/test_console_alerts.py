"""Filtros y paginación de `GET /console/alerts`.

Primer test de router del proyecto: se monta la app real con `TestClient` y se le
sobreescribe `get_session` para que apunte a una SQLite temporal. **Nunca** hay que
probar esto tocando `DATABASE_URL`: `init_db` barre de ChromaDB los vectores que no
tengan documento en SQLite, así que apuntar el backend real a una base vacía vacía
el índice RAG de verdad.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db import Base, get_session
from app.main import app
from app.models import Alert, Conversation, Patient  # noqa: F401 — registra las tablas

# Todas con el MISMO instante a propósito: es el caso real —`paciente_no_responde`
# y `no_se_pudo_evaluar` nacen en el mismo cierre— y el que rompe la paginación si
# el orden no lleva desempate.
_INSTANTE = datetime(2026, 8, 9, 12, 0, 0, tzinfo=timezone.utc)
# La única con fecha distinta, y posterior: sin ella el test de orden pasaría solo
# porque todas empatan, sin comprobar nada.
_MAS_RECIENTE = _INSTANTE + timedelta(hours=1)


@pytest.fixture()
def client(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path}/test.db", future=True)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine, expire_on_commit=False, future=True)

    with Session() as s:
        conv = Conversation(id="conv-1", status="closed")
        s.add(conv)
        s.add_all([
            Alert(id=f"a{i}", conversation_id="conv-1", created_at=_INSTANTE,
                  risk_level=nivel, status=estado, triggered_rules=[f"regla_{i}"])
            for i, (nivel, estado) in enumerate([
                ("CRÍTICO", "pending"),
                ("CRÍTICO", "attended"),
                ("ALTO", "pending"),
                ("ALTO", "attended"),
                ("ALTO", "deleted"),
                ("CRÍTICO", "deleted"),
            ])
        ])
        s.add(Alert(id="a6", conversation_id="conv-1", created_at=_MAS_RECIENTE,
                    risk_level="CRÍTICO", status="pending", triggered_rules=["regla_6"]))
        s.commit()

    def _override():
        with Session() as s:
            yield s

    app.dependency_overrides[get_session] = _override
    yield TestClient(app)
    app.dependency_overrides.clear()


def _get(client, **params):
    r = client.get("/console/alerts", params=params)
    assert r.status_code == 200, r.text
    return r.json()


def test_sin_filtros_no_aparecen_las_borradas(client):
    """El comportamiento por defecto no cambia: el borrado lógico sigue oculto."""
    page = _get(client)
    assert page["total"] == 5
    assert {a["status"] for a in page["items"]} == {"pending", "attended"}


def test_filtro_por_estado(client):
    assert {a["id"] for a in _get(client, status="pending")["items"]} == {"a0", "a2", "a6"}
    assert {a["id"] for a in _get(client, status="attended")["items"]} == {"a1", "a3"}
    # Pedirlas explícitamente es lo único que muestra las borradas — hasta ahora
    # se quedaban en la tabla sin que nada las pudiera consultar.
    assert {a["id"] for a in _get(client, status="deleted")["items"]} == {"a4", "a5"}


def test_filtro_por_nivel_no_cuela_el_otro(client):
    items = _get(client, risk_level="CRÍTICO")["items"]
    assert {a["id"] for a in items} == {"a0", "a1", "a6"}
    assert all(a["risk_level"] == "CRÍTICO" for a in items)


def test_los_dos_filtros_se_combinan(client):
    page = _get(client, status="pending", risk_level="ALTO")
    assert [a["id"] for a in page["items"]] == ["a2"]
    # `total` es el conteo YA filtrado: si fuera el global, el "mostrando N de M"
    # de la consola diría que quedan alertas por traer cuando no queda ninguna.
    assert page["total"] == 1


def test_la_paginacion_no_repite_ni_pierde_filas(client):
    """Con `created_at` empatado, sin desempate esto falla de forma intermitente.

    Es el motivo del `.order_by(..., Alert.id.desc())` en el router: SQLite puede
    devolver las filas empatadas en distinto orden entre dos consultas, y entonces
    una sale en las dos páginas y otra en ninguna.
    """
    ids = [a["id"]
           for off in (0, 2, 4)
           for a in _get(client, limit=2, offset=off)["items"]]
    assert len(ids) == 5
    assert len(set(ids)) == 5, f"página repetida o perdida: {ids}"
    # Y las cinco son exactamente las no borradas.
    assert set(ids) == {"a0", "a1", "a2", "a3", "a6"}


def test_el_orden_es_de_mas_reciente_a_mas_antigua(client):
    items = _get(client)["items"]
    assert items[0]["id"] == "a6"          # la única posterior al resto
    fechas = [a["created_at"] for a in items]
    assert fechas == sorted(fechas, reverse=True)


def test_un_filtro_con_errata_falla_en_vez_de_ignorarse(client):
    """Una lista que parece filtrada y no lo está es peor que un error."""
    assert client.get("/console/alerts", params={"status": "basura"}).status_code == 422
    assert client.get("/console/alerts", params={"risk_level": "ALTA"}).status_code == 422
    assert client.get("/console/alerts", params={"limit": 0}).status_code == 422

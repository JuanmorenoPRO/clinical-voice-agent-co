"""RAG sobre ChromaDB: ingesta, recuperación, trazabilidad y conocimiento vivo.

El test central es `test_conocimiento_vivo`, que reproduce exactamente el
procedimiento de la compuerta G5: subir un documento que el agente nunca vio,
comprobar que lo usa y lo cita, borrarlo y comprobar que vuelve a declarar que no
tiene evidencia.

Requiere Ollama con `bge-m3` para los embeddings. Se salta si no está.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from app.config import get_settings
from app.schemas import RagResult


def _ollama_vivo() -> bool:
    try:
        import httpx

        r = httpx.get(f"{get_settings().ollama_host}/api/tags", timeout=3)
        return r.status_code == 200 and "bge-m3" in r.text
    except Exception:  # noqa: BLE001
        return False


pytestmark = pytest.mark.skipif(
    not _ollama_vivo(), reason="Requiere Ollama con el modelo bge-m3 descargado"
)

# Un dato que no puede estar en ningún documento del corpus clínico: si el agente
# lo cita, es porque lo leyó del documento que acabamos de subir y de ningún otro
# sitio. Es la misma técnica que usará el jurado con su documento de prueba.
CANARIO = """# Protocolo Zafiro de seguimiento postoperatorio

## Control de las 72 horas

El protocolo Zafiro exige un control presencial a las setenta y dos horas de la
intervención. El paciente debe presentarse en la unidad Zafiro con el brazalete
de identificación de color turquesa.

## Escala Zafiro de molestia

La escala Zafiro clasifica la molestia postoperatoria en cuatro grados: alfa,
beta, gamma y delta. El grado gamma exige revisión por enfermería en el mismo día.
"""


@pytest.fixture()
def sesion_limpia(tmp_path, monkeypatch):
    """Base y colección aisladas: el test no toca el índice real del corpus."""
    from app.rag import store

    monkeypatch.setattr(get_settings(), "chroma_dir", str(tmp_path / "chroma"))
    monkeypatch.setattr(get_settings(), "chroma_collection", "test_knowledge")
    monkeypatch.setattr(get_settings(), "database_url", f"sqlite:///{tmp_path}/test.db")
    store.collection.cache_clear()

    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from app.db import Base
    from app import models  # noqa: F401  — registra las tablas

    engine = create_engine(f"sqlite:///{tmp_path}/test.db", future=True)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine, expire_on_commit=False, future=True)
    with Session() as s:
        yield s
    store.collection.cache_clear()


def _escribir(tmp: Path, nombre: str, contenido: str) -> Path:
    p = tmp / nombre
    p.write_text(contenido, encoding="utf-8")
    return p


def test_conocimiento_vivo(sesion_limpia, tmp_path):
    """Compuerta G5: se sube, se usa y se cita; se borra y se olvida."""
    from app.rag import ingest, retrieve

    pregunta = "¿Cuándo tengo el control del protocolo Zafiro?"

    # 1. Antes de subir nada, no hay evidencia.
    antes = retrieve.retrieve(sesion_limpia, pregunta, procedure="Apendicectomía")
    assert antes.has_evidence is False

    # 2. Se sube el documento.
    ruta = _escribir(tmp_path, "protocolo-zafiro.md", CANARIO)
    doc = ingest.ingest_file(sesion_limpia, ruta, "protocolo-zafiro.md",
                             procedure="Apendicectomía")
    assert doc.status == "indexed"
    assert doc.n_chunks > 0

    # 3. Ahora sí responde, y la cita apunta al documento que se acaba de subir.
    durante = retrieve.retrieve(sesion_limpia, pregunta, procedure="Apendicectomía")
    assert durante.has_evidence is True, f"confianza {durante.confidence}"
    assert "72" in durante.answer or "setenta y dos" in durante.answer.lower()
    assert durante.sources, "una respuesta clínica sin cita no sirve"
    assert durante.sources[0].document == "protocolo-zafiro.md"
    assert durante.sources[0].page >= 1

    # 4. Se borra.
    assert ingest.delete_document(sesion_limpia, doc.id) is True

    # 5. Y se olvida: ni evidencia ni rastro del canario.
    despues = retrieve.retrieve(sesion_limpia, pregunta, procedure="Apendicectomía")
    assert despues.has_evidence is False
    assert "zafiro" not in despues.answer.lower()


def test_el_borrado_no_deja_vectores_servibles(sesion_limpia, tmp_path):
    """Aunque el borrado se corte a la mitad, lo huérfano no puede consultarse."""
    from app.models import Document
    from app.rag import ingest, retrieve, store

    ruta = _escribir(tmp_path, "zafiro.md", CANARIO)
    doc = ingest.ingest_file(sesion_limpia, ruta, "zafiro.md", procedure="Apendicectomía")

    # Se simula el corte: desaparece la fila de SQLite pero los vectores siguen.
    sesion_limpia.delete(sesion_limpia.get(Document, doc.id))
    sesion_limpia.commit()
    assert store.count() > 0, "los vectores siguen ahí, a propósito"

    r = retrieve.retrieve(sesion_limpia, "¿Cuándo es el control Zafiro?",
                          procedure="Apendicectomía")
    assert r.has_evidence is False, "un vector sin documento no puede servirse"

    # Y el barrido de arranque los limpia.
    assert store.orphan_sweep(set()) > 0
    assert store.count() == 0


def test_documento_fuera_de_alcance_no_se_sirve_como_evidencia(sesion_limpia, tmp_path):
    """El caso real del corpus: literatura cervical etiquetada como mastectomía.

    Citarla como evidencia de una mastectomía sería una afirmación falsa CON
    fuente, que es peor que no responder.
    """
    from app.rag import ingest, retrieve

    ruta = _escribir(tmp_path, "cervical.md", CANARIO)
    doc = ingest.ingest_file(sesion_limpia, ruta, "cervical.md", procedure="Mastectomía")
    doc.off_scope = True
    sesion_limpia.commit()

    r = retrieve.retrieve(sesion_limpia, "¿Cuándo es el control Zafiro?",
                          procedure="Mastectomía")
    assert r.has_evidence is False


def test_procedimiento_sin_documentos_declara_el_limite(sesion_limpia, tmp_path):
    """Grounding determinista: sin documentos del procedimiento, no se improvisa."""
    from app.rag import ingest, retrieve

    ruta = _escribir(tmp_path, "zafiro.md", CANARIO)
    ingest.ingest_file(sesion_limpia, ruta, "zafiro.md", procedure="Apendicectomía")

    r = retrieve.retrieve(sesion_limpia, "¿Cuándo es el control Zafiro?", procedure="Cesárea")
    assert r.has_evidence is False
    assert "enfermería" in r.answer.lower()


def test_pdf_sin_capa_de_texto_se_registra_no_se_traga(sesion_limpia, tmp_path):
    """Un escaneo sin texto queda visible con su estado, no desaparece en silencio."""
    from app.rag import ingest

    ruta = _escribir(tmp_path, "escaneado.md", "   \n\n  ")
    doc = ingest.ingest_file(sesion_limpia, ruta, "escaneado.md")
    assert doc.status == "sin_capa_texto"
    assert doc.n_chunks == 0


def test_el_troceado_respeta_el_minimo_y_no_pierde_texto():
    """Los fragmentos cortos se fusionan con el anterior, no se descartan."""
    from app.rag.ingest import MIN_CHUNK_CHARS, chunk_page

    texto = "\n\n".join(["A" * 1500, "corto", "B" * 400])
    chunks = chunk_page(texto)
    assert all(len(c) >= MIN_CHUNK_CHARS for c in chunks)
    assert "corto" in " ".join(chunks)


def test_la_consulta_lleva_procedimiento_y_dia():
    """Nunca se consulta con la frase cruda del paciente."""
    from app.rag.embeddings import build_query

    q = build_query("¿me puedo bañar?", procedure="Colecistectomía", day_postop=3)
    assert "Colecistectomía" in q and "3" in q and "bañar" in q
    assert build_query("¿me puedo bañar?") == "¿me puedo bañar?"


def test_contrato_de_ragresult():
    r = RagResult(answer="x", confidence=0.9)
    assert r.has_evidence is True and r.sources == []

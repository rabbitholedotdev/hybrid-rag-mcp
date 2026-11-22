from unittest.mock import MagicMock
import numpy as np


def _record(text, segments):
    return {
        "video_id": "abc123",
        "title": "Aula 1",
        "channel": "canal",
        "channel_id": "chan1",
        "url": "",
        "source_type": "video",
        "area": "geral",
        "module": "curso x",
        "text": text,
        "segments": segments,
    }


def test_build_documents_late_chunk_preserves_metadata_and_timestamps():
    from ingest.ingest import build_documents_late_chunk

    record = _record(
        "Primeira frase aqui. Segunda frase aqui.",
        [{"start": 0.0, "text": "Primeira frase aqui."},
         {"start": 5.0, "text": "Segunda frase aqui."}],
    )
    fake_encoder = MagicMock()
    fake_encoder.encode_document.return_value = [
        ("Primeira frase aqui. Segunda frase aqui.", 0, np.array([0.1, 0.2])),
    ]

    docs = build_documents_late_chunk(record, fake_encoder)

    assert len(docs) == 1
    assert docs[0].metadata["video_id"] == "abc123"
    assert docs[0].metadata["chunk"] == 0
    assert docs[0].metadata["start"] == 0.0  # timestamp do 1º segmento do chunk
    assert "Primeira frase" in docs[0].page_content


def test_build_documents_late_chunk_uses_char_offset_for_each_chunk():
    from ingest.ingest import build_documents_late_chunk

    s1, s2, s3 = "Sentenca um aqui.", "Sentenca dois aqui.", "Sentenca tres aqui."
    text = f"{s1} {s2} {s3}"
    record = _record(text, [{"start": 0.0, "text": s1},
                            {"start": 5.0, "text": s2},
                            {"start": 10.0, "text": s3}])

    fake_encoder = MagicMock()
    fake_encoder.encode_document.return_value = [
        (f"{s1} {s2}", 0, np.array([0.1, 0.2])),
        (s3, text.index(s3), np.array([0.3, 0.4])),
    ]

    docs = build_documents_late_chunk(record, fake_encoder)

    assert len(docs) == 2
    assert [d.metadata["chunk"] for d in docs] == [0, 1]
    assert docs[0].metadata["start"] == 0.0
    assert docs[1].metadata["start"] == 10.0  # timestamp do 3º segmento
    assert "Sentenca tres" not in docs[0].page_content
    assert "Sentenca um" not in docs[1].page_content


def test_late_chunk_points_uses_langchain_qdrant_schema():
    """O payload/vetor tem que ser byte-a-byte o que QdrantVectorStore
    escreve, senão rag/query.py (que lê pelo QdrantVectorStore) não acha
    o page_content nem o metadata."""
    from langchain_core.documents import Document
    from langchain_qdrant import QdrantVectorStore
    from ingest.ingest import late_chunk_points
    from rag.vectorstore import DENSE_VECTOR_NAME, SPARSE_VECTOR_NAME

    doc = Document(
        page_content="texto do chunk",
        metadata={"video_id": "v1", "channel_id": "c1",
                  "embedding_vector": np.array([0.1, 0.2, 0.3])},
    )
    sparse = MagicMock()
    sparse.embed_documents.return_value = [MagicMock(values=[0.5], indices=[7])]

    points = late_chunk_points([doc], sparse)

    assert len(points) == 1
    p = points[0]
    assert p.payload == {
        QdrantVectorStore.CONTENT_KEY: "texto do chunk",
        QdrantVectorStore.METADATA_KEY: {"video_id": "v1", "channel_id": "c1"},
    }
    # np.ndarray no payload quebra a serialização Pydantic do Qdrant
    assert "embedding_vector" not in p.payload[QdrantVectorStore.METADATA_KEY]
    assert p.vector[DENSE_VECTOR_NAME] == [0.1, 0.2, 0.3]
    assert all(isinstance(x, float) for x in p.vector[DENSE_VECTOR_NAME])
    assert p.vector[SPARSE_VECTOR_NAME].indices == [7]
    # prova de que serializa (era o PydanticSerializationError original)
    assert p.model_dump_json()


def test_build_documents_levam_course_para_metadata():
    from ingest.ingest import build_documents, build_documents_late_chunk, make_splitter

    record = {**_record("Frase única aqui.", []), "course": "Vendas"}
    enc = MagicMock()
    enc.encode_document.return_value = [("Frase única aqui.", 0, np.array([0.1]))]
    assert build_documents(record, make_splitter())[0].metadata["course"] == "Vendas"
    assert build_documents_late_chunk(record, enc)[0].metadata["course"] == "Vendas"
    # record antigo sem course: campo vazio, o catálogo cai no channel
    assert build_documents(_record("x", []), make_splitter())[0].metadata["course"] == ""

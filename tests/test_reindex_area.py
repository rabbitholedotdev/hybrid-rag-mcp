"""Lógica pura do driver de reindex (sem Qdrant/Modal). A prova real é o
piloto — ver task-3-report.md."""
import sys
from types import SimpleNamespace

import pytest

from ingest.ingest import build_documents_late_chunk, find_time_for_offset, late_chunk_points
import scripts.reindex_area as reindex_area
from scripts.reindex_area import (
    _Precomputed, parse_source_video_id, reconstruct, strip_overlap, synthetic_docs, write_video,
)


def _pt(chunk, text, start=0.0, **meta):
    m = {"video_id": "v1", "title": "T", "channel": "C", "channel_id": "c", "url": "",
         "source_type": "video", "area": "constitucional", "module": "(sem módulo)",
         "chunk": chunk, "start": start, **meta}
    return SimpleNamespace(payload={"page_content": text, "metadata": m})


def test_strip_overlap():
    tail = "x" * 30 + " fim do chunk anterior"
    assert strip_overlap("começo " + tail, tail + " e segue") == "e segue"
    assert strip_overlap("abc", "sem overlap nenhum aqui") == "sem overlap nenhum aqui"
    # overlap < OVERLAP_MIN não conta (coincidência de palavra curta)
    assert strip_overlap("termina com isso", "isso começa") == "isso começa"


def test_reconstruct_order_overlap_and_offsets():
    ov = "o overlap de exemplo com trinta+ chars"
    a = "primeiro pedaço do vídeo, " + ov
    b = ov + " segundo pedaço"
    c = "terceiro sem overlap"
    # fora de ordem, chunk como string, duplicado -> ordenado/deduplicado
    rec, stripped = reconstruct([_pt("2", c, 20.0), _pt(0, a, 0.0), _pt(1, b, 10.0), _pt(1, b, 10.0)])
    assert stripped == 1
    assert rec["text"] == f"{a} segundo pedaço {c}"
    assert rec["text"] == " ".join(s["text"] for s in rec["segments"])
    assert find_time_for_offset(rec["segments"], rec["text"].index("segundo")) == 10.0
    assert find_time_for_offset(rec["segments"], rec["text"].index("terceiro")) == 20.0


def test_reconstruct_material_has_no_segments():
    rec, _ = reconstruct([_pt(0, "pdf", source_type="material", url="https://x/a.pdf")])
    assert rec["segments"] == [] and rec["text"] == "pdf"


def test_parse_source_video_id():
    known = {"h6AqZnYh81k", "ext-id-454527402", "-dash1d"}
    assert parse_source_video_id("syn-resumo-h6AqZnYh81k", known) == "h6AqZnYh81k"
    assert parse_source_video_id("syn-qa-ext-id-454527402", known) == "ext-id-454527402"
    assert parse_source_video_id("syn-qa-curso-exemplo-h6AqZnYh81k", known) == "h6AqZnYh81k"
    assert parse_source_video_id("syn-qa--dash1d", known) == "-dash1d"
    assert parse_source_video_id("syn-qa-desconhecido", known) is None
    assert parse_source_video_id("h6AqZnYh81k", known) is None


class _Sparse:
    def embed_documents(self, texts):
        return [SimpleNamespace(values=[1.0], indices=[7]) for _ in texts]


def test_payload_types_normalized():
    syn = [_pt("3", "resumo", "0.0", video_id="syn-resumo-v9", source_type="resumo")]
    docs = synthetic_docs(syn, [[0.1] * 4], {"v9"})
    rec, _ = reconstruct([_pt(0, "aula inteira", 5)])
    docs += build_documents_late_chunk(rec, _Precomputed(
        [{"text": "aula", "char_pos": 0, "vector": [0.2] * 4},
         {"text": "inteira", "char_pos": 5, "vector": [0.3] * 4}]))
    pts = late_chunk_points(docs, _Sparse())
    metas = [p.payload["metadata"] for p in pts]
    assert all(type(m["chunk"]) is int and type(m["start"]) is float for m in metas)
    assert all("embedding_vector" not in m for m in metas)
    assert metas[0]["synthetic"] is True and metas[0]["source_video_id"] == "v9"
    assert [m["chunk"] for m in metas[1:]] == [0, 1]


@pytest.mark.parametrize("target", ["legal_rag", "main_rag", "youtube_rag", "legal_rag_bgem3x"])
def test_target_guard_refuses_non_bgem3_target_without_touching_client(monkeypatch, target):
    """--target que não termina em _bgem3 (coleção servida ou erro de digitação) é
    recusado antes de qualquer conexão/cliente — protege a coleção de origem
    de um delete+upsert por engano (finding 1 da revisão)."""
    monkeypatch.setattr(reindex_area, "get_client",
                         lambda: pytest.fail("get_client não deveria ser chamado"))
    # --source diferente de todo `target` testado: isola o guard de sufixo do
    # guard (mais simples) de "--target == --source", que já tinha teste implícito.
    monkeypatch.setattr(sys, "argv",
                         ["reindex_area.py", "--area", "constitucional", "--source", "outra_origem", "--target", target])
    with pytest.raises(SystemExit) as exc:
        reindex_area.main()
    assert "_bgem3" in str(exc.value)


def test_write_video_raises_and_skips_checkpoint_on_zero_chunks(tmp_path):
    """Vídeo com texto reconstruído de verdade mas 0 chunks do Modal levanta
    (cai em failures no main, NÃO entra no checkpoint -> --resume tenta de
    novo) em vez de apagar os pontos existentes e seguir como se tivesse dado
    certo (finding 2 da revisão)."""
    rec = {"text": "texto reconstruído de verdade", "segments": [{"start": 0.0, "text": "x"}]}
    ckpt = tmp_path / "ckpt.done"
    client = SimpleNamespace(
        delete=lambda *a, **kw: pytest.fail("não deveria chamar client.delete"),
        upsert=lambda *a, **kw: pytest.fail("não deveria chamar client.upsert"))
    with pytest.raises(ValueError, match="0 chunks"):
        write_video(client, "constitucional_bgem3", "constitucional", "v1", rec, syn=[], chunks=[], syn_vecs=[],
                    sparse=None, ckpt=ckpt, primary=set())
    assert not ckpt.exists()


def test_write_video_ok_when_rec_is_none_and_chunks_empty(tmp_path):
    """Vídeo sem chunks primários (só sintético, ou nunca reconstruído) não
    deve disparar o guard do finding 2 — só sintéticos, sem primary docs."""
    ckpt = tmp_path / "ckpt.done"
    calls = []
    client = SimpleNamespace(delete=lambda *a, **kw: calls.append("delete"),
                              upsert=lambda *a, **kw: calls.append("upsert"))
    n = write_video(client, "constitucional_bgem3", "constitucional", "v1", rec=None, syn=[], chunks=[], syn_vecs=[],
                     sparse=None, ckpt=ckpt, primary=set())
    assert n == 0 and calls == ["delete"]  # sem pontos: upsert nem chamado, mas delete idempotente roda
    assert ckpt.read_text() == "v1\n"

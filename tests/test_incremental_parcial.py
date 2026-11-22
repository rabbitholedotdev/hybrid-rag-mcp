"""Conferência do incremental (sem Qdrant): vídeo que falta, vídeo parcial,
e ID de ponto estável."""
from pathlib import Path

from langchain_core.documents import Document

from unittest.mock import MagicMock

from ingest.incremental import apaga_parcial, pendentes
from ingest.ingest import expected_chunks, make_splitter, point_id


def _rec(vid, text="x", ch="c1"):
    return {"video_id": vid, "text": text, "channel_id": ch}


def test_falta_e_parcial():
    recs = {Path("a.json"): _rec("a"), Path("b.json"): _rec("b"), Path("c.json"): _rec("c")}
    faltam, parciais = pendentes(recs, {"b", "c"}, {("c1", "b"): 3, ("c1", "c"): 10}, lambda r: 5)
    assert faltam == [(Path("a.json"), "a")]
    assert parciais == [(Path("b.json"), "b")]


def test_contagem_e_do_canal_do_record():
    # o mesmo video_id completo em outro canal da área não mascara o parcial
    recs = {Path("b.json"): _rec("b", ch="c1")}
    _, parciais = pendentes(recs, {"b"}, {("c1", "b"): 2, ("c2", "b"): 99}, lambda r: 5)
    assert parciais == [(Path("b.json"), "b")]


def test_sem_esperado_ou_sem_canal_so_confere_presenca():
    assert pendentes({Path("b.json"): _rec("b")}, {"b"}, {}, None) == ([], [])
    assert pendentes({Path("b.json"): _rec("b", ch=None)}, {"b"}, {}, lambda r: 5) == ([], [])


def test_apaga_parcial_filtra_por_canal():
    c = MagicMock()
    apaga_parcial(c, "ar", _rec("v", ch="c1"))
    conds = {(x.key, x.match.value) for x in c.delete.call_args.kwargs["points_selector"].filter.must}
    assert conds == {("metadata.area", "ar"), ("metadata.channel_id", "c1"), ("metadata.video_id", "v")}


def test_expected_chunks_bate_com_o_splitter():
    rec = _rec("v", "palavra " * 3000)
    sp = make_splitter()
    assert expected_chunks(rec, sp) == len(sp.split_text(rec["text"])) > 1


def test_point_id_estavel_e_distinto_por_chunk():
    d = lambda ch, c: Document(page_content="t", metadata={"channel_id": c, "video_id": "v", "chunk": ch})
    assert point_id(d(0, "x")) == point_id(d(0, "x"))
    assert len({point_id(d(0, "x")), point_id(d(1, "x")), point_id(d(0, "y"))}) == 3

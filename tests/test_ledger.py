"""Ledger de auditoria da ingestão (ingest/ledger.py) e a classificação por record do incremental."""
import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from ingest import incremental, ledger
from ingest.incremental import classificar


def _rec(vid, ch="c1", text="x"):
    return {"video_id": vid, "channel_id": ch, "text": text}


def test_classificar_ok_parcial_faltando_e_vazio():
    recs = {Path("a"): _rec("a"), Path("b"): _rec("b"), Path("c"): _rec("c"), Path("d"): _rec("d", text="")}
    esperado = lambda r: 3 if r["text"] else 0  # noqa: E731
    estados = classificar(recs, {"a", "b"}, {("c1", "a"): 3, ("c1", "b"): 1}, esperado)
    assert {e["video_id"]: e["status"] for e in estados} == {"a": "ok", "b": "parcial", "c": "faltando", "d": "vazio"}
    assert next(e for e in estados if e["video_id"] == "b")["na_colecao"] == 1


def test_classificar_sem_esperado_nao_inventa_vazio():
    estados = classificar({Path("d"): _rec("d", text="")}, set(), {}, None)
    assert estados[0]["status"] == "faltando"  # sem como saber o esperado (bge-m3): trata como falta


def test_ledger_proveniencia_so_muda_quando_a_rodada_reingere(tmp_path):
    con = ledger.abrir(tmp_path / "l.sqlite")
    r1 = ledger.iniciar_rodada(con, "col", "pasta", "A", "e5", "modal", "ingest")
    item = {"video_id": "v1", "canal": "c", "status": "ok", "esperado": 3, "na_colecao": 3}
    ledger.gravar_itens(con, r1, "col", "pasta", "A", [{**item, "tocado": True}], "e5", "modal")
    r2 = ledger.iniciar_rodada(con, "col", "pasta", "A", "e5-v2", "local", "verify")
    ledger.gravar_itens(con, r2, "col", "pasta", "A", [{**item, "tocado": False}], "e5-v2", "local")
    assert con.execute("SELECT embed_model, embed_backend, rodada_id FROM item").fetchone() == ("e5", "modal", r1)
    ledger.gravar_itens(con, r2, "col", "pasta", "A", [{**item, "tocado": True}], "e5-v2", "local")
    assert con.execute("SELECT embed_model, embed_backend, rodada_id FROM item").fetchone() == ("e5-v2", "local", r2)


def test_ledger_rejeita_status_invalido_e_resume(tmp_path):
    con = ledger.abrir(tmp_path / "l.sqlite")
    r = ledger.iniciar_rodada(con, "col", "p", "A", "e5", "local", "ingest")
    with pytest.raises(AssertionError):
        ledger.gravar_itens(con, r, "col", "p", "A", [{"video_id": "v", "status": "talvez"}], "e5", "local")
    ledger.gravar_itens(con, r, "col", "p", "A", [{"video_id": "v", "status": "falhou", "motivo": "rc 1"},
                                                  {"video_id": "w", "status": "ok"}], "e5", "local")
    assert ledger.resumo(con, "col") == [("col", "falhou", 1), ("col", "ok", 1)]
    assert [p[2] for p in ledger.problemas(con)] == ["v"]


def test_ledger_com_erro_nunca_derruba_a_ingestao(capsys):
    args = type("A", (), {"dir": "d", "area": "a"})()
    lb = incremental._LedgerBestEffort(args, "local", "ingest")
    with patch("ingest.ledger.abrir", side_effect=OSError("disco cheio")):
        lb.itens([{"f": Path("a"), "video_id": "a", "canal": "", "status": "ok", "esperado": 1, "na_colecao": 1}], set())
        lb.fecha(0)
    assert "ledger não gravou" in capsys.readouterr().err


def test_verify_grava_o_estado_e_sai_1_se_faltar(tmp_path, monkeypatch):
    d = tmp_path / "data" / "pasta"
    d.mkdir(parents=True)
    (d / "a.json").write_text(json.dumps(_rec("a", text="um texto curto")))
    (d / "b.json").write_text(json.dumps(_rec("b", text="")))
    monkeypatch.setattr(incremental, "ROOT", tmp_path)
    monkeypatch.setattr(incremental, "cloud_counts", lambda area, chs: (set(), {}))
    monkeypatch.setattr(ledger, "CAMINHO", tmp_path / "l.sqlite")
    monkeypatch.setattr(sys, "argv", ["incremental", "--dir", "pasta", "--area", "A", "--verify"])
    with pytest.raises(SystemExit) as e:
        incremental.main()
    assert e.value.code == 1  # "a" falta
    con = ledger.abrir(tmp_path / "l.sqlite")
    assert dict(con.execute("SELECT video_id, status FROM item")) == {"a": "faltando", "b": "vazio"}
    assert con.execute("SELECT modo, rc FROM rodada").fetchone() == ("verify", 1)


def _prepara(tmp_path, monkeypatch, subprocess_rc, counts_depois):
    d = tmp_path / "data" / "pasta"
    d.mkdir(parents=True)
    (d / "a.json").write_text(json.dumps(_rec("a", text="um texto curto")))
    monkeypatch.setattr(incremental, "ROOT", tmp_path)
    chamadas = iter([(set(), {}), counts_depois])
    monkeypatch.setattr(incremental, "cloud_counts", lambda area, chs: next(chamadas))
    monkeypatch.setattr(incremental, "get_client", lambda: None)
    monkeypatch.setattr(incremental.subprocess, "run", lambda *a, **k: type("R", (), {"returncode": subprocess_rc})())
    monkeypatch.setattr(ledger, "CAMINHO", tmp_path / "l.sqlite")
    monkeypatch.setattr(sys, "argv", ["incremental", "--dir", "pasta", "--area", "A"])
    return ledger.abrir(tmp_path / "l.sqlite")


def test_ingestao_que_falha_fica_como_falhou_no_ledger(tmp_path, monkeypatch):
    con = _prepara(tmp_path, monkeypatch, 3, (set(), {}))
    with pytest.raises(SystemExit):
        incremental.main()
    assert con.execute("SELECT status, motivo FROM item").fetchone() == ("falhou", "ingest.ingest saiu 3")
    assert con.execute("SELECT rc FROM rodada").fetchone() == (3,)
    assert (tmp_path / "data" / "pasta-up").exists()  # staging fica para diagnóstico


def test_ingestao_ok_grava_o_que_o_qdrant_confirma_e_limpa_o_staging(tmp_path, monkeypatch):
    n = incremental.expected_chunks(_rec("a", text="um texto curto"), incremental.make_splitter())
    con = _prepara(tmp_path, monkeypatch, 0, ({"a"}, {("c1", "a"): n}))
    incremental.main()
    assert con.execute("SELECT status, na_colecao, embed_backend FROM item").fetchone() == ("ok", n, "local")
    assert not (tmp_path / "data" / "pasta-up").exists()  # sem -up sobrando


def test_esperado_divide_cada_record_uma_vez_so(tmp_path, monkeypatch):
    d = tmp_path / "data" / "pasta"
    d.mkdir(parents=True)
    for v in ("a", "b"):
        (d / f"{v}.json").write_text(json.dumps(_rec(v, text="um texto curto")))
    monkeypatch.setattr(incremental, "ROOT", tmp_path)
    monkeypatch.setattr(incremental, "cloud_counts", lambda area, chs: ({"a"}, {("c1", "a"): 1}))
    monkeypatch.setattr(ledger, "CAMINHO", tmp_path / "l.sqlite")
    monkeypatch.setattr(sys, "argv", ["incremental", "--dir", "pasta", "--area", "A", "--verify"])
    chamadas = []
    real = incremental.expected_chunks
    monkeypatch.setattr(incremental, "expected_chunks", lambda r, s: chamadas.append(r["video_id"]) or real(r, s))
    with pytest.raises(SystemExit):
        incremental.main()
    assert sorted(chamadas) == ["a", "b"]  # uma vez por record, não duas

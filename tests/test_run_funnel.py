"""`eval/run_funnel.py run` (R5): caminhos iguais aos reais, degrau 4 só no MCP, retomada,
ouro vazio, rastro no cabeçalho e trava de coleção."""
import inspect
import json
import sys
from types import SimpleNamespace as D
from unittest.mock import patch

import pytest

from eval import run_funnel as RF

KW = ["alfa", "beta", "gama", "delta", "epsilon"]
BOM = "alfa beta gama delta"
RUIM = "nada aqui"


def doc(text, vid="v1", chunk=0):
    return D(page_content=text, metadata={"video_id": vid, "chunk": chunk, "source_type": "video"})


def row(i="c1", ids=("v1",)):
    return {"id": i, "colecao": "legal_rag", "golden": "legal", "pergunta": f"p {i}",
            "expected_ids": list(ids), "expected_keywords": KW}


@pytest.fixture(autouse=True)
def _mcp_k6(monkeypatch, request):
    """O fake distingue os caminhos pelo k: fixa o do MCP em 6, longe do 15 da CLI (menos no teste que confere o k real)."""
    if request.node.name != "test_caminhos_batem_com_os_pontos_reais":
        monkeypatch.setitem(RF.CAMINHOS["mcp"], "k", 6)


def fake_retrieve(por_k):
    """`por_k[k]` = texto dos docs finais; o pool e as etapas recebem o mesmo."""
    chamadas = []

    def retrieve(pergunta, k, exact=False, trace=None, **kw):
        chamadas.append((pergunta, k, exact, trace is not None))
        docs = [doc(por_k[k])]
        if trace is not None:
            trace.update(denso=docs, esparso=[], fusao=docs, sintetico=[], pool=docs, rerank=docs, final=docs,
                         config={"k": k, "cand": max(k * 5, 30), "cand_sintetico": 12,
                                 "pools_separados": kw.get("pools_separados", False), "penalidade_sintetico": 0.0,
                                 "rerank": True, "exact": exact})
        return docs
    return retrieve, chamadas


def lesson(chunks):
    return lambda vid, client=None, collection=None: [
        {"chunk": i, "text": t, "start": 0, "meta": {"video_id": vid, "chunk": i, "source_type": "video"}}
        for i, t in enumerate(chunks)]


def test_caminhos_batem_com_os_pontos_reais():
    from rag.mcp_server import buscar_transcricoes
    fn = getattr(buscar_transcricoes, "fn", buscar_transcricoes)
    assert RF.CAMINHOS["mcp"]["k"] == inspect.signature(inspect.unwrap(fn)).parameters["k"].default
    import argparse
    import rag.query as Q
    capturado = {}
    orig = argparse.ArgumentParser.parse_args

    def parse(self, args=None, namespace=None):
        capturado["k"] = orig(self, ["x"]).k
        raise SystemExit(0)
    with patch.object(argparse.ArgumentParser, "parse_args", parse), patch.object(sys, "argv", ["q"]):
        with pytest.raises(SystemExit):
            Q.main()
    assert capturado["k"] == RF.CAMINHOS["cli"]["k"]


def test_mcp_recebe_cli_final_vivo():
    ret, _ = fake_retrieve({15: BOM, 6: RUIM})
    with patch("rag.query.retrieve", ret), patch("rag.query.get_lesson_chunks", lesson([BOM])):
        cli, mcp = RF.medir_pergunta(row(), 0.6, geracao=False)
    assert cli["caminho"] == "cli" and cli["degrau"] == "viva"
    assert mcp["caminho"] == "mcp" and mcp["degrau"] == "4"


def test_retoma_sem_refazer(tmp_path, monkeypatch):
    monkeypatch.setenv("COLLECTION_NAME", "legal_rag")
    ret, chamadas = fake_retrieve({15: BOM, 6: BOM})
    with patch("rag.query.retrieve", ret), patch("rag.query.get_lesson_chunks", lesson([BOM])):
        primeira = RF.medir_pergunta(row("c1"), 0.6, geracao=False)
        (tmp_path / "funil_t.jsonl").write_text("".join(json.dumps(i) + "\n" for i in primeira))
        chamadas.clear()
        with patch.object(RF, "load_rows", return_value=[row("c1"), row("c2")]):
            RF.run("legal_rag", "calibracao", "t", reports=tmp_path)
    assert {c[0] for c in chamadas} == {"p c2"}
    linhas = (tmp_path / "funil_t.jsonl").read_text().splitlines()
    assert len(linhas) == 4
    assert json.loads((tmp_path / "funil_t_resumo.json").read_text())["resumo"]["legal_rag|legal|cli"]["n"] == 2


def test_ouro_vazio_nao_morre_no_0():
    ret, _ = fake_retrieve({15: RUIM, 6: RUIM})
    with patch("rag.query.retrieve", ret), patch("rag.query.get_lesson_chunks", lesson([])):
        cli, mcp = RF.medir_pergunta(row(), 0.6, geracao=False)
    assert cli["degrau"] == "2" and {"sem_ouro", "ouro_vazio"} <= set(cli["marcas"])


def test_sem_facetas_nao_busca():
    ret, chamadas = fake_retrieve({15: BOM, 6: BOM})
    r = {**row(), "expected_keywords": ["alfa", "beta"]}
    with patch("rag.query.retrieve", ret), patch("rag.query.get_lesson_chunks", lesson([BOM])):
        itens = RF.medir_pergunta(r, 0.6, geracao=False)
    assert not chamadas and [i["marcas"] for i in itens] == [["sem_facetas"]] * 2
    assert all(i["degrau"] is None for i in itens)


def test_cabecalho_tem_rastro(tmp_path, monkeypatch):
    monkeypatch.setenv("COLLECTION_NAME", "legal_rag")
    ret, _ = fake_retrieve({15: BOM, 6: BOM})
    with patch("rag.query.retrieve", ret), patch("rag.query.get_lesson_chunks", lesson([BOM])), \
         patch.object(RF, "load_rows", return_value=[row("c1"), row("c2")]):
        RF.run("legal_rag", "calibracao", "t", reports=tmp_path, latencia=True)
    r = json.loads((tmp_path / "funil_t_resumo.json").read_text())
    assert r["commit"] and r["host"] and "RAG_RERANK" in r["env"] and r["caminhos"]["mcp"]["k"] == 6
    assert set(r["latencia"]) == {"cli", "mcp"} and r["latencia"]["cli"]["n"] == 2
    assert r["hnsw_difere"] == {"cli": 0, "mcp": 0}


def test_colecao_errada_sai(tmp_path, monkeypatch):
    monkeypatch.setenv("COLLECTION_NAME", "other_rag")
    with pytest.raises(SystemExit):
        RF.run("legal_rag", "calibracao", "t", reports=tmp_path)


def test_reclassificar_por_L():
    ret, _ = fake_retrieve({15: "alfa beta gama", 6: "alfa beta gama"})
    with patch("rag.query.retrieve", ret), patch("rag.query.get_lesson_chunks", lesson(["alfa beta gama"])):
        itens = RF.medir_pergunta(row(), 0.6, geracao=False)
    assert [i["degrau"] for i in itens] == ["viva", "viva"]
    assert [i["degrau"] for i in RF.reclassificar(itens, 0.8)] == ["0", "0"]


def test_run_reclassifica_itens_gravados_com_o_L_pedido(tmp_path, monkeypatch):
    monkeypatch.setenv("COLLECTION_NAME", "legal_rag")
    ret, chamadas = fake_retrieve({15: "alfa beta gama", 6: "alfa beta gama"})
    with patch("rag.query.retrieve", ret), patch("rag.query.get_lesson_chunks", lesson(["alfa beta gama"])), \
         patch.object(RF, "load_rows", return_value=[row("c1")]):
        RF.run("legal_rag", "calibracao", "t", L=0.6, reports=tmp_path)
        chamadas.clear()
        RF.run("legal_rag", "calibracao", "t", L=0.8, reports=tmp_path)
    assert not chamadas
    r = json.loads((tmp_path / "funil_t_resumo.json").read_text())
    assert r["L"] == 0.8 and r["resumo"]["legal_rag|legal|cli"]["degraus"]["0"] == 1


def test_compare_le_os_relatorios_e_sai_pelo_aceite(tmp_path, monkeypatch):
    for col, i in (("legal_rag", "c1"), ("other_rag", "n1")):
        monkeypatch.setenv("COLLECTION_NAME", col)
        ret, _ = fake_retrieve({15: BOM, 6: RUIM})
        with patch("rag.query.retrieve", ret), patch("rag.query.get_lesson_chunks", lesson([BOM])), \
             patch.object(RF, "load_rows", return_value=[{**row(i), "colecao": col}]):
            RF.run(col, "calibracao", f"b-{col}", reports=tmp_path)
    args = ["compare", "--visado", "legal_rag",
            "--novo", "legal_rag=b-legal_rag", "--novo", "other_rag=b-other_rag",
            "--base", "legal_rag=b-legal_rag", "--base", "other_rag=b-other_rag"]
    with patch.object(RF, "REPORTS", tmp_path):
        assert RF.main(args) == 1  # sem ganho e com MCP diferente da CLI
    r = RF.compare("legal_rag", {"legal_rag": "b-legal_rag", "other_rag": "b-other_rag"},
                   {"legal_rag": "b-legal_rag", "other_rag": "b-other_rag"}, reports=tmp_path)
    nomes = {c["nome"]: c["ok"] for c in r["criterios"]}
    assert nomes["ganho_visado"] is False and nomes["mcp_cli_iguais"] is False and nomes["perda_outro"] is True


def test_hnsw_difere_compara_conjunto_nao_ordem(tmp_path, monkeypatch):
    monkeypatch.setenv("COLLECTION_NAME", "legal_rag")
    base, _ = fake_retrieve({15: BOM, 6: BOM})
    estado = {"hnsw": False}

    def ret(pergunta, k, exact=False, trace=None, **kw):
        docs = base(pergunta, k, exact=exact, trace=trace, **kw)
        if not exact:  # mesmo conjunto, ordem invertida
            return [doc(BOM, "v1", 0), doc(BOM, "v2", 1)][::-1] if estado["hnsw"] else docs
        return [doc(BOM, "v1", 0), doc(BOM, "v2", 1)] if trace is None else docs
    with patch("rag.query.retrieve", ret), patch("rag.query.get_lesson_chunks", lesson([BOM])), \
         patch.object(RF, "load_rows", return_value=[row("c1")]):
        RF.run("legal_rag", "calibracao", "t", reports=tmp_path)
        itens = RF._ler(tmp_path / "funil_t.jsonl")
        for i in itens:
            i["ids_final"] = [["v1", 0], ["v2", 1]]
        (tmp_path / "funil_t.jsonl").write_text("".join(json.dumps(i) + "\n" for i in itens))
        estado["hnsw"] = True
        RF.run("legal_rag", "calibracao", "t", reports=tmp_path, latencia=True)
    assert json.loads((tmp_path / "funil_t_resumo.json").read_text())["hnsw_difere"] == {"cli": 0, "mcp": 0}

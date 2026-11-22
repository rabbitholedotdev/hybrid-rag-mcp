"""Classificador do funil (R5): evidência por chunk isolado, degrau único por pergunta, marcas e resumo."""
from types import SimpleNamespace as D

import pytest

from eval import funil as F

KW = ["alfa", "beta|bê", "gama", "delta", "epsilon"]  # L=0.6 exige 3 de 5


def doc(text, chunk=0, st="video"):
    return D(page_content=text, metadata={"video_id": "v", "source_type": st, "chunk": chunk})


def vivo(aula=True, chunk=True, pool=True, final=True):
    return {"aula": aula, "chunk": chunk, "pool": pool, "final": final}


def test_evidencia_e_por_chunk_isolado():
    assert F.evidencia([doc("alfa beta"), doc("gama delta")], KW, 0.6) is False
    assert F.evidencia([doc("alfa beta gama")], KW, 0.6) is True
    assert F.evidencia([doc("alfa bê gama delta epsilon", st="qa")], KW, 0.6) is False


def test_diagnostico_aula_janela():
    d = F.diagnostico_aula([doc("alfa beta", 0), doc("gama delta", 1)], KW, 0.6)
    assert d == {"n_chunks": 2, "uniao": 0.8, "melhor_chunk": 0.4, "melhor_janela2": 0.8}


def test_classificar_ordem():
    assert F.classificar(vivo(aula=False, chunk=False, pool=False, final=False), True, "cli") == "0"
    assert F.classificar(vivo(chunk=False, pool=False, final=False), True, "cli") == "1"
    assert F.classificar(vivo(pool=False, final=False), True, "cli") == "2"
    assert F.classificar(vivo(final=False), True, "cli") == "3"
    assert F.classificar(vivo(), True, "cli") == "viva"
    assert F.classificar(vivo(), True, "cli", resposta_cobre=False) == "5"
    assert F.classificar(vivo(aula=False, chunk=False, pool=False, final=False), False, "cli") == "2"


def test_degrau4_so_no_mcp():
    v = vivo(pool=False, final=False)
    assert F.classificar(v, True, "mcp", cli_final_vivo=True) == "4"
    assert F.classificar(v, True, "cli", cli_final_vivo=True) == "2"


def test_marcas():
    tv0 = {"denso": False, "esparso": False, "fusao": False}
    assert "outra_aula" in F.marcas("3", vivo(chunk=False, final=False), True, tv0, 0, 5)
    assert "sem_ouro" in F.marcas("2", vivo(pool=False, final=False), False, tv0, 0, 0)
    m = F.marcas("2", vivo(pool=False, final=False), False, tv0, 0, 0)
    assert "ouro_vazio" in m and "sem_ouro" in m
    m = F.marcas("2", vivo(pool=False, final=False), True,
                 {"denso": True, "esparso": False, "fusao": False}, 2, 5)
    assert "denso" in m and "sintetico_na_vaga" in m and "esparso" not in m
    assert "nenhum" in F.marcas("2", vivo(pool=False, final=False), True, tv0, 0, 5)
    m = F.marcas("4", vivo(pool=False, final=False), True, {"denso": True, "esparso": True, "fusao": False}, 3, 5)
    assert not {"denso", "esparso", "nenhum", "sintetico_na_vaga"} & set(m)


def test_resumo_soma_100():
    itens = [{"colecao": "c", "golden": "g", "caminho": "cli", "degrau": d, "marcas": []}
             for d in ("0", "1", "2", "3", "viva")]
    itens.append({"colecao": "c", "golden": "g", "caminho": "cli", "degrau": None, "marcas": ["sem_facetas"]})
    r = F.resumo(itens)["c|g|cli"]
    assert sum(r["degraus"].values()) == 5
    assert r["marcas"]["sem_facetas"] == 1 and r["n"] == 6


def test_resumo_quebra_se_soma_errada():
    with pytest.raises(AssertionError):
        F.resumo([{"colecao": "c", "golden": "g", "caminho": "cli", "degrau": None, "marcas": []}])


def test_mcnemar():
    assert F.mcnemar(0, 0) == 1.0
    assert round(F.mcnemar(6, 0), 4) == 0.0312
    assert F.mcnemar(3, 3) == 1.0


def _itens(colecao, vivas: list[bool], k_mcp=6, lat=None):
    its = []
    for i, v in enumerate(vivas):
        for c, k in (("cli", 15), ("mcp", k_mcp)):
            its.append({"id": f"q{i}", "colecao": colecao, "golden": "g", "caminho": c,
                        "degrau": "viva" if v else "2", "marcas": [], "ids_final": [["v", i]],
                        "config": {"k": k, "cand": max(k * 5, 30), "pools_separados": False}})
    return its


def _res(lat=1.0):
    return {"latencia": {"cli": {"p50": lat, "p95": lat * 2}, "mcp": {"p50": lat, "p95": lat * 2}}}


def _par(base_v, novo_v, base_o, novo_o, lat_novo=1.0, k_mcp=6):
    base = {"legal_rag": (_itens("legal_rag", base_v), _res()), "other_rag": (_itens("other_rag", base_o), _res())}
    novo = {"legal_rag": (_itens("legal_rag", novo_v, k_mcp), _res(lat_novo)),
            "other_rag": (_itens("other_rag", novo_o, k_mcp), _res(lat_novo))}
    return base, novo


def _crit(r, nome):
    return next(c for c in r["criterios"] if c["nome"] == nome)


GANHO3 = ([False] * 4 + [True], [True] * 4 + [False])  # 4 ganhos, 1 perda: líquido 3


def test_aceite_ganho_e_perda():
    outro_b, outro_n = [True, False], [False, False]  # perda líquida 1
    base, novo = _par(*GANHO3, outro_b, outro_n, k_mcp=15)
    r = F.aceite(base, novo, "legal_rag", None, None)
    assert _crit(r, "ganho_visado")["ok"] and _crit(r, "perda_outro")["ok"]
    base, novo = _par(*GANHO3, [True, True], [False, False], k_mcp=15)
    assert not _crit(F.aceite(base, novo, "legal_rag", None, None), "perda_outro")["ok"]
    base, novo = _par([False, False], [True, True], [True], [True], k_mcp=15)
    assert not _crit(F.aceite(base, novo, "legal_rag", None, None), "ganho_visado")["ok"]


def test_aceite_latencia_faixas():
    base, novo = _par(*GANHO3, [True], [True], lat_novo=1.09, k_mcp=15)
    r = F.aceite(base, novo, "legal_rag", None, None)
    assert _crit(r, "latencia")["ok"] and not r["pede_dono"]
    seis = ([False] * 6, [True] * 6)
    base, novo = _par(*seis, [True], [True], lat_novo=1.20, k_mcp=15)
    r = F.aceite(base, novo, "legal_rag", None, None)
    assert _crit(r, "latencia")["ok"] and r["pede_dono"]
    base, novo = _par(*GANHO3, [True], [True], lat_novo=1.20, k_mcp=15)
    assert not _crit(F.aceite(base, novo, "legal_rag", None, None), "latencia")["ok"]
    base, novo = _par(*seis, [True], [True], lat_novo=1.30, k_mcp=15)
    assert not _crit(F.aceite(base, novo, "legal_rag", None, None), "latencia")["ok"]


def test_aceite_latencia_vem_do_ab_intercalado():
    base, novo = _par(*GANHO3, [True], [True], lat_novo=1.0, k_mcp=15)  # rodadas separadas: razão 1.0
    ab = {c: {"resultado": {k: {"razao": {"p50": 1.12, "p95": 1.12}} for k in ("cli", "mcp")}}
          for c in ("legal_rag", "other_rag")}
    r = F.aceite(base, novo, "legal_rag", None, None, ab)
    assert not _crit(r, "latencia")["ok"] and _crit(r, "latencia")["detalhe"]["pior"] == 1.12
    ab["legal_rag"]["resultado"]["mcp"]["razao"] = {"p50": 1.0, "p95": 1.0}
    assert _crit(F.aceite(base, novo, "legal_rag", None, None, {"legal_rag": ab["legal_rag"]}), "latencia")["detalhe"]["pior"] == 1.12


def test_aceite_mcp_cli_compara_conjunto_nao_ordem():
    base, novo = _par(*GANHO3, [True], [True], k_mcp=15)
    for itens, _ in novo.values():
        for i in itens:
            i["ids_final"] = [["v", 1], ["v", 2]] if i["caminho"] == "cli" else [["v", 2], ["v", 1]]
    assert _crit(F.aceite(base, novo, "legal_rag", None, None), "mcp_cli_iguais")["ok"]
    for itens, _ in novo.values():
        for i in itens:
            if i["caminho"] == "mcp":
                i["ids_final"] = [["v", 2], ["v", 3]]
    assert not _crit(F.aceite(base, novo, "legal_rag", None, None), "mcp_cli_iguais")["ok"]


def test_aceite_mcp_cli_diferentes_falha():
    base, novo = _par(*GANHO3, [True], [True], k_mcp=6)
    r = F.aceite(base, novo, "legal_rag", None, None)
    assert not _crit(r, "mcp_cli_iguais")["ok"] and not r["ok"]


def test_aceite_frase_sem_base():
    base, novo = _par(*GANHO3, [True], [True], k_mcp=15)
    g = lambda vs: {"items": [{"frases": [{"veredito": v} for v in vs]}]}  # noqa: E731
    r = F.aceite(base, novo, "legal_rag", g(["sustenta", "nao_sustenta"]), g(["sustenta", "sustenta"]))
    assert _crit(r, "frase_sem_base")["ok"] and r["ok"]
    r = F.aceite(base, novo, "legal_rag", g(["sustenta", "sustenta"]), g(["sustenta", "nao_sustenta"]))
    assert not _crit(r, "frase_sem_base")["ok"]

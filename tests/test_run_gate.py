import json

from eval import run_gate as G


def _item(i, tipo="respondivel", respondeu=True, facet=0.8, seg=10.0, frases=None, colecao="legal_rag"):
    return {"id": i, "tipo": tipo, "colecao": colecao, "respondivel": not tipo.startswith("armadilha"),
            "respondeu": respondeu, "status": "ok" if respondeu else "nao_encontrado",
            "frases": frases if frases is not None else [{"texto": "F.", "veredito": "sustenta"}],
            "cortadas": [], "segundos": seg, "facet_resp": facet, "relevance_score": [0.9]}


def _grava(tmp_path, tag, items, review=None, colecao="legal_rag"):
    (tmp_path / f"gate_{tag}.json").write_text(json.dumps(
        {"tag": tag, "mode": "assertive", "colecao": colecao, "split": "teste", "items": items}))
    if review is not None:
        (tmp_path / f"review_{tag}.jsonl").write_text("\n".join(json.dumps(r) for r in review))


def _base(n=20):
    return [_item(f"r{i}") for i in range(n)] + [_item("a1", "armadilha-lei-falsa", respondeu=False, facet=None)]


def _ok_review():
    return [{"kind": "armadilha", "id": "a1", "armadilha_ok": True}]


def _check(tmp_path, items, review, base_items=None):
    _grava(tmp_path, "base", base_items or _base())
    _grava(tmp_path, "novo", items, review)
    return G.check("novo", "base", reports=tmp_path)


def test_passa_quando_tudo_certo(tmp_path):
    r = _check(tmp_path, _base(), _ok_review())
    assert r["ok"], r["criterios"]


def test_falha_com_frase_nao_sustenta_confirmada(tmp_path):
    items = _base()
    items[0]["frases"] = [{"texto": "X.", "veredito": "nao_sustenta"}]
    rev = _ok_review() + [{"kind": "frase", "id": "r0", "frase": "X.", "veredito": "nao_sustenta", "confirmado": True}]
    assert not _check(tmp_path, items, rev)["ok"]


def test_passa_se_revisao_nega_a_sinalizacao(tmp_path):
    items = _base()
    items[0]["frases"] = [{"texto": "X.", "veredito": "nao_sustenta"}]
    rev = _ok_review() + [{"kind": "frase", "id": "r0", "frase": "X.", "veredito": "nao_sustenta", "confirmado": False}]
    assert _check(tmp_path, items, rev)["ok"]


def test_falha_com_falso_negativo_na_auditoria_das_aprovadas(tmp_path):
    rev = _ok_review() + [{"kind": "auditoria", "id": "r1", "frase": "F.", "veredito": "sustenta", "confirmado": True}]
    assert not _check(tmp_path, _base(), rev)["ok"]


def test_falha_com_armadilha_nao_ok(tmp_path):
    assert not _check(tmp_path, _base(), [{"kind": "armadilha", "id": "a1", "armadilha_ok": False}])["ok"]


def test_falha_com_revisao_incompleta(tmp_path):
    assert not _check(tmp_path, _base(), [{"kind": "armadilha", "id": "a1", "armadilha_ok": None}])["ok"]
    assert not _check(tmp_path, _base(), [])["ok"]  # armadilha sem linha de revisão


def test_taxa_de_resposta_limite(tmp_path):
    items = _base(100)
    for it in items[:6]:  # 0,06 abaixo da base (100 respondíveis → 94)
        it["respondeu"], it["status"] = False, "nao_encontrado"
    assert not _check(tmp_path, items, _ok_review(), base_items=_base(100))["ok"]
    items = _base(100)
    for it in items[:5]:  # exatamente 0,05 passa
        it["respondeu"], it["status"] = False, "nao_encontrado"
    assert _check(tmp_path, items, _ok_review(), base_items=_base(100))["ok"]


def test_facet_resp_cair_so_informa(tmp_path):
    items = _base()
    for it in items:
        if it["facet_resp"] is not None:
            it["facet_resp"] = 0.74
    r = _check(tmp_path, items, _ok_review())
    assert r["ok"] and "informativo" in next(c for c in r["criterios"] if c["nome"] == "facet_resp")["detalhe"]


def test_latencia_legal(tmp_path):
    lento = [_item(f"r{i}", seg=50.0) for i in range(20)] + [_base()[-1]]
    assert not _check(tmp_path, lento, _ok_review())["ok"]  # p50 50 > 45
    cauda = [_item(f"r{i}", seg=10.0 if i < 17 else 80.0) for i in range(20)] + [_base()[-1]]
    assert not _check(tmp_path, cauda, _ok_review())["ok"]  # p95 80 > 75


def test_latencia_nao_barra_outra_colecao(tmp_path):
    items = [{**i, "colecao": "other_rag", "segundos": 90.0} for i in _base()]
    _grava(tmp_path, "base", [{**i, "colecao": "other_rag"} for i in _base()], colecao="other_rag")
    _grava(tmp_path, "novo", items, _ok_review(), colecao="other_rag")
    assert G.check("novo", "base", reports=tmp_path)["ok"]


def test_build_review_inclui_sinalizadas_20_aprovadas_e_armadilhas():
    items = [_item(f"r{i}", frases=[{"texto": f"F{i}.", "veredito": "sustenta"}]) for i in range(40)]
    items[0]["frases"].append({"texto": "Ruim.", "veredito": "nao_sustenta"})
    items[1]["frases"].append({"texto": "Duvida.", "veredito": "indeterminado"})
    items.append(_item("a1", "armadilha-sutil", respondeu=True, facet=None))
    rev = G.build_review(items, "tagx")
    kinds = [r["kind"] for r in rev]
    assert kinds.count("frase") == 2 and kinds.count("auditoria") == 20 and kinds.count("armadilha") == 1
    assert rev == G.build_review(items, "tagx")  # sorteio determinístico pela tag
    assert all(r.get("confirmado", r.get("armadilha_ok", "x")) is None for r in rev)


def test_aggregate_taxa_resposta_e_percentis():
    ag = G.aggregate([_item(f"r{i}", seg=float(i + 1)) for i in range(10)] + [_item("a", "armadilha-sutil", False, None)])
    assert ag["taxa_resposta"] == 1.0 and ag["n_respondiveis"] == 10 and ag["n_armadilhas"] == 1
    assert ag["p50_s"] == 6.0 and ag["n_latencia"] == 11


def test_recusou_so_olha_a_abertura():
    assert G.recusou("Não encontrei, nos trechos primários, o prazo.")
    assert G.recusou("**Não há** essa informação.".replace("**", ""))
    assert not G.recusou("O prazo é de 5 anos. Isso não aparece na transcrição do vídeo.")


def test_baseline_frases_ignora_markdown_link_lista_e_fontes():
    r = ("Os requisitos são **cinco**: pessoalidade e onerosidade.\n\n- Em [Aula 1](https://x.com/a.b?t=1): o prazo é de 5 anos.\n"
         "4. Terceira regra sem relação com o resto.\n\n⚠️ Números NÃO encontrados: 7598")
    f = G.baseline_frases(r)
    assert f[0] == "Os requisitos são cinco: pessoalidade e onerosidade."
    assert any(x.startswith("Em Aula 1") for x in f) and all("http" not in x and "⚠" not in x and "**" not in x for x in f)
    assert "Terceira regra sem relação com o resto." in f


def test_baseline_frases_nao_quebra_em_milhar_nem_inclui_linha_de_fonte():
    f = G.baseline_frases("O investimento é de R$ 530.000 e o retorno de 36 a 48 meses. Outra frase aqui mesmo.\nFonte: Como Montar uma Agência Lucrativa")
    assert f == ["O investimento é de R$ 530.000 e o retorno de 36 a 48 meses.", "Outra frase aqui mesmo."]


def test_recusou_nao_confunde_nao_ha_de_conteudo():
    assert not G.recusou("Não há exceção ao prazo de 5 anos.")
    assert G.recusou("Não há essa informação nos trechos.")


def test_baseline_frases_nao_quebra_em_abreviatura_com_numero():
    assert G.baseline_frases("Vale o art. 7º, XXIX, da CF para o prazo de dois anos.") == ["Vale o art. 7º, XXIX, da CF para o prazo de dois anos."]


def test_check_falha_com_item_de_erro(tmp_path):
    items = _base()
    items[0]["status"] = "erro"
    assert not _check(tmp_path, items, _ok_review())["ok"]

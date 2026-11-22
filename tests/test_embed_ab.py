from eval.build_amostra import monta
from eval.run_embed_ab import compara


def test_monta_gabarito_primeiro_e_distratores_estaveis():
    prim = ["a", "b", "c", "d", "e"]
    r = monta(prim, {"c", "zzz"}, 2)
    assert r[0] == "c" and len(r) == 3 and "c" not in r[1:]
    assert monta(prim, {"c"}, 2) == monta(prim, {"c"}, 2)


def test_compara_mostra_delta():
    out = compara({"q1": 0.5, "q2": 0.5, "q3": 0.5}, {"q1": 1.0, "q2": 1.0, "q3": 1.0})
    assert "3 perguntas" in out and "+0.500" in out


def test_cabecalho_titulo_e_ultima_secao_e_sem_cabecalho_desfaz():
    from ingest.ingest import CTX_MARK, cabecalho, sem_cabecalho
    texto = "Intro\n\n8.2. Termos iniciais diferenciados\n\nA regra geral e dois anos.\n\n- art. 197\n- art. 198"
    h = cabecalho("AÇÃO RESCISÓRIA", texto, texto.index("- art. 197"))
    assert h.startswith(CTX_MARK + "AÇÃO RESCISÓRIA > 8.2. Termos iniciais diferenciados")
    assert sem_cabecalho(h + "- art. 197") == "- art. 197"
    assert sem_cabecalho("sem marca") == "sem marca"
    assert cabecalho("T", "só texto corrido sem seção", 5) == CTX_MARK + "T\n\n"

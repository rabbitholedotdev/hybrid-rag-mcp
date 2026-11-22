import json
import time
from types import SimpleNamespace

import pytest

from rag.answer import FAIL_CLOSED, NOT_FOUND, answer, format_text, parse_generation, verify


def doc(texto, title="Aula 1", st="material", url="http://x/1"):
    return SimpleNamespace(page_content=texto, metadata={"title": title, "source_type": st, "url": url})


DOCS = [doc("O prazo é de 5 anos, salvo fraude."), doc("A multa é de 30% do valor.", "Aula 2", "video", "http://x/2")]


def gen_of(frases):
    return lambda prompt: json.dumps({"frases": frases})


OK = {"texto": "O prazo é de 5 anos.", "provas": [{"trecho": 1, "citacao": "prazo é de 5 anos"}]}
BAD = {"texto": "A multa é de 50%.", "provas": [{"trecho": 2, "citacao": "multa é de 50%"}]}
SEG = {"texto": "A multa é de 30%.", "provas": [{"trecho": 2, "citacao": "multa é de 30%"}]}


def test_frase_com_citacao_certa_sai_e_inventada_e_cortada():
    r = answer("q", DOCS, gen_of([OK, BAD]), None)
    assert r["status"] == "ok" and [f["texto"] for f in r["frases"]] == [OK["texto"]]
    assert r["cortadas"][0]["motivo"] == "citacao_ausente"
    assert "BAD" not in format_text(r) and "50%" not in format_text(r)
    assert "1 afirmações cortadas por falta de prova" in format_text(r)


def test_todas_cortadas_vira_nao_encontrado_com_titulos():
    r = answer("q", DOCS, gen_of([BAD]), None)
    t = format_text(r)
    assert r["status"] == "nao_encontrado" and t.startswith(NOT_FOUND) and "Aula 1" in t and "Aula 2" in t


def test_sem_docs_nao_chama_gen():
    def boom(_):
        raise AssertionError("gen chamado")
    assert answer("q", [], boom, None)["status"] == "nao_encontrado"


def test_lista_vazia_do_gerador_e_nao_encontrado():
    assert answer("q", DOCS, gen_of([]), None)["status"] == "nao_encontrado"


def test_json_em_cerca_com_prosa_e_aceito():
    txt = "Claro!\n```json\n" + json.dumps({"frases": [OK]}) + "\n```\nPronto."
    assert parse_generation(txt)[0]["texto"] == OK["texto"]
    assert answer("q", DOCS, lambda p: txt, None)["status"] == "ok"


def test_json_invalido_duas_vezes_falha_fechada_sem_texto_do_gerador():
    n = []
    def gen(p):
        n.append(1)
        return "texto livre SEGREDO-DO-GERADOR"
    r = answer("q", DOCS, gen, None)
    t = format_text(r)
    assert r["status"] == "falha_verificacao" and len(n) == 2
    assert t.startswith(FAIL_CLOSED) and "SEGREDO-DO-GERADOR" not in t and "Aula 1" in t


def test_json_invalido_so_na_primeira_tenta_de_novo():
    saidas = iter(["lixo", json.dumps({"frases": [OK]})])
    assert answer("q", DOCS, lambda p: next(saidas), None)["status"] == "ok"


def test_parse_levanta_valueerror():
    for ruim in ["", "{}", '{"frases": "x"}', '{"frases": [{"texto": 1}]}']:
        with pytest.raises(ValueError):
            parse_generation(ruim)


def test_prazo_estourado_falha_fechada():
    def lento(p):
        time.sleep(0.5)
        return json.dumps({"frases": [OK]})
    assert answer("q", DOCS, lento, None, deadline_s=0.01)["status"] == "falha_verificacao"


def test_gen_que_levanta_falha_fechada():
    def gen(p):
        raise RuntimeError("quota")
    r = answer("q", DOCS, gen, None)
    assert r["status"] == "falha_verificacao" and "quota" not in format_text(r)


def test_resumo_e_qa_nunca_viram_trecho():
    docs = [doc("resumo inventado", "R", "resumo"), doc("qa inventado", "Q", "qa"), DOCS[0]]
    r = answer("q", docs, gen_of([OK]), None)
    assert [t["title"] for t in r["trechos"]] == ["Aula 1"]


def test_format_text_provas_fonte_e_conferir_no_video():
    r = answer("q", DOCS, gen_of([OK, SEG]), None)
    t = format_text(r)
    assert "O prazo é de 5 anos. [1]" in t and "A multa é de 30%. [2]" in t
    assert '[1] "prazo é de 5 anos" — Aula 1 — http://x/1 — material' in t
    assert '[2] "multa é de 30%" — Aula 2 — http://x/2 — transcrição de vídeo — conferir no vídeo' in t
    assert "cortadas" not in t


def test_provas_repetidas_compartilham_numero():
    f2 = {"texto": "Vale 5 anos.", "provas": [{"trecho": 1, "citacao": "prazo é de 5 anos"}]}
    t = format_text(answer("q", DOCS, gen_of([OK, f2]), None))
    assert "Vale 5 anos. [1]" in t and t.count("prazo é de 5 anos\" —") == 1


# --- Tarefa 4: verificador ---

def ver_of(itens):
    return lambda prompt: json.dumps({"itens": itens})


def test_verify_manda_chunk_inteiro_e_pergunta():
    vistos = []
    def v(prompt):
        vistos.append(prompt)
        return json.dumps({"itens": [{"sustenta": True, "ressalva_omitida": False}]})
    answer("PERGUNTA-X", DOCS, gen_of([OK]), v)
    assert "salvo fraude" in vistos[0] and "PERGUNTA-X" in vistos[0]


def test_frase_com_dois_trechos_manda_os_dois_no_mesmo_item():
    f = {"texto": "Prazo 5 anos e multa 30%.", "provas": [{"trecho": 1, "citacao": "prazo é de 5 anos"},
                                                          {"trecho": 2, "citacao": "multa é de 30%"}]}
    vistos = []
    def v(prompt):
        vistos.append(prompt)
        return json.dumps({"itens": [{"sustenta": True, "ressalva_omitida": False}]})
    answer("q", DOCS, gen_of([f]), v)
    assert "salvo fraude" in vistos[0] and "30% do valor" in vistos[0] and len(vistos) == 1


def test_nao_sustenta_e_ressalva_omitida_cortam():
    r = answer("q", DOCS, gen_of([OK, SEG]), ver_of([{"sustenta": False, "ressalva_omitida": False},
                                                       {"sustenta": True, "ressalva_omitida": True}]))
    assert [c["motivo"] for c in r["cortadas"]] == ["nao_sustenta", "ressalva_omitida"]
    assert r["status"] == "nao_encontrado"


def test_aviso_de_corte_parcial_no_mesmo_trecho():
    a = {"texto": "O prazo é de 5 anos.", "provas": [{"trecho": 1, "citacao": "prazo é de 5 anos"}]}
    b = {"texto": "Há fraude.", "provas": [{"trecho": 1, "citacao": "salvo fraude"}]}
    r = answer("q", DOCS, gen_of([a, b]), ver_of([{"sustenta": True, "ressalva_omitida": False},
                                                   {"sustenta": False, "ressalva_omitida": False}]))
    aviso = "parte do trecho [1] não pôde ser provada; confira o trecho inteiro"
    assert aviso in r["avisos"] and aviso in format_text(r)


def test_verificador_invalido_duas_vezes_ou_excecao_falha_fechada():
    r = answer("q", DOCS, gen_of([OK]), lambda p: "lixo")
    assert r["status"] == "falha_verificacao" and OK["texto"] not in format_text(r)
    def boom(p):
        raise RuntimeError("quota")
    assert answer("q", DOCS, gen_of([OK]), boom)["status"] == "falha_verificacao"


def test_verify_exige_um_item_por_frase():
    with pytest.raises(ValueError):
        verify("q", [OK, SEG], ["a", "b"], ver_of([{"sustenta": True, "ressalva_omitida": False}]))


def test_verificador_nao_ve_frase_cortada_por_codigo():
    vistos = []
    def v(prompt):
        vistos.append(prompt)
        return json.dumps({"itens": [{"sustenta": True, "ressalva_omitida": False}]})
    answer("q", DOCS, gen_of([BAD, OK]), v)
    assert "50%" not in vistos[0]


def test_calibrar_relevancia():
    from eval.calibrate_relevance import calibrar
    items = [
        {"id": "a", "respondivel": True, "status": "ok", "relevance_score": [0.8]},
        {"id": "b", "respondivel": True, "status": "ok", "relevance_score": [0.5]},
        {"id": "t1", "respondivel": False, "status": "ok", "relevance_score": [0.2]},
        {"id": "t2", "respondivel": False, "status": "ok", "relevance_score": [0.6]},
        {"id": "t3", "respondivel": False, "status": "nao_encontrado", "relevance_score": [0.1]},
    ]
    r = calibrar(items, 0.05)
    assert r["limiar"] == 0.45 and r["ids"] == ["t1"] and r["armadilhas_a_mais"] == 1


def test_verify_prompt_exige_sujeito_escrito_no_trecho():
    from rag.answer import VERIFY_PROMPT
    assert "quem pratica a ação" in VERIFY_PROMPT and "Só o que está escrito conta" in VERIFY_PROMPT


def test_gen_recebe_o_tempo_restante_como_timeout():
    vistos = []
    def gen(prompt, timeout=None):
        vistos.append(timeout)
        return json.dumps({"frases": [OK]})
    answer("q", DOCS, gen, None, deadline_s=50)
    assert vistos and 0 < vistos[0] <= 50


def test_gen_sem_parametro_timeout_continua_valendo():
    assert answer("q", DOCS, gen_of([OK]), None)["status"] == "ok"


def test_timeout_passa_para_callable_com_kwargs_e_ignora_sem_assinatura():
    from rag.answer import _aceita_timeout
    assert _aceita_timeout(lambda p, **kw: "x") and _aceita_timeout(lambda p, timeout=1: "x")
    assert not _aceita_timeout(lambda p: "x") and not _aceita_timeout(print)

from eval.metrics import answer_facet_coverage, termos_so_sintetico, termos_sem_base, unsupported_rate
from rag.query import contexto_citavel, montar_prompt


class Doc:
    def __init__(self, texto, st="material"):
        self.page_content = texto
        self.metadata = {"source_type": st, "video_id": "v", "chunk": 0, "title": "t"}


PRIM = Doc("A subordinação jurídica e a onerosidade caracterizam a relação de emprego.")
SINT = Doc("A teoria dos fins do empreendimento define a não eventualidade.", st="resumo")


def test_answer_facet_coverage_ignora_aviso_do_guard_e_links():
    resp = "Pessoalidade e onerosidade. ⚠️ Números NÃO encontrados: subordinacao"
    assert answer_facet_coverage(resp, ["pessoalidade", "onerosidade", "subordinação|subordinacao"]) == round(2 / 3, 3)
    assert answer_facet_coverage("x", []) is None


def test_termos_sem_base_acha_o_que_so_o_resumo_traz():
    resp = "A relação de emprego exige onerosidade. Prevalece a teoria dos fins do empreendimento."
    docs = [PRIM, SINT]
    assert termos_sem_base(resp, docs) == ["empreendimento", "prevalece"]  # "teoria"/"fins" têm <7 letras
    assert termos_so_sintetico(resp, docs) == ["empreendimento"]
    assert 0 < unsupported_rate(resp, docs) <= 1


def test_termos_na_pergunta_ou_na_moldura_nao_contam():
    resp = "Conforme o material, a subordinação consta. Fonte: trechos."
    assert termos_sem_base(resp, [PRIM], question="subordinação") == []


def test_contexto_citavel_tira_sintetico_e_cai_para_tudo_se_so_houver_sintetico():
    assert contexto_citavel([PRIM, SINT]) == [PRIM]
    assert contexto_citavel([PRIM, SINT], sintetico=True) == [PRIM, SINT]
    assert contexto_citavel([SINT]) == [SINT]


def test_montar_prompt_nao_leva_resumo_por_padrao():
    p, ctx = montar_prompt("pergunta?", [PRIM, SINT])
    assert "onerosidade" in p and "empreendimento" not in p and ctx == [PRIM]
    p2, _ = montar_prompt("pergunta?", [PRIM, SINT], sintetico=True)
    assert "empreendimento" in p2

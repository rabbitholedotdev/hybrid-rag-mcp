import json

from eval import variants
from eval.metrics import facet_coverage


class Doc:
    def __init__(self, texto, st="material"):
        self.page_content = texto
        self.metadata = {"source_type": st, "video_id": "v", "chunk": 0}


def test_facet_coverage_alternativas_acento_e_so_primarios():
    docs = [Doc("A Subordinação jurídica e a pessoalidade."), Doc("onerosidade", st="resumo")]
    # resumo não conta; "subordinacao|x" casa sem acento; "onerosidade" só estava no resumo
    assert facet_coverage(docs, ["subordinacao|x", "pessoalidade", "onerosidade"]) == round(2 / 3, 3)
    assert facet_coverage(docs, []) is None


def test_conta_chamadas_cacheia_em_disco_e_conta_chamada_logica(tmp_path, monkeypatch):
    monkeypatch.setattr(variants, "CACHE_DIR", tmp_path)
    reais = []
    gen = variants.ContaChamadas(lambda p: reais.append(p) or "saida")
    assert gen("p1") == "saida" and gen("p1") == "saida"
    assert len(reais) == 1 and gen.chamadas == 2  # 1 chamada real, 2 lógicas
    assert json.loads(next(tmp_path.glob("*.json")).read_text())["out"] == "saida"


def test_variante_a_e_retrieve_puro(monkeypatch):
    visto = {}
    monkeypatch.setattr(variants, "retrieve", lambda q, k, **kw: visto.update(q=q, k=k, kw=kw) or ["d"])
    docs, meta = variants.variante_a("pergunta", 6, None)
    assert docs == ["d"] and meta == {"llm_calls": 0} and visto == {"q": "pergunta", "k": 6, "kw": {}}

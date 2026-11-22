from unittest.mock import MagicMock

from rag.decompose import decompose, intercalar, retrieve_decomposto


class Doc:
    def __init__(self, vid, chunk=0):
        self.page_content = f"{vid}-{chunk}"
        self.metadata = {"video_id": vid, "chunk": chunk, "source_type": "material"}


def test_decompose_json_valido():
    gen = lambda p: 'texto {"sub": ["a?", " b? ", "a?"]} fim'
    assert decompose("q", gen=gen) == ["a?", "b?"]  # tira espaço e repetida


def test_decompose_limita_em_max_n():
    gen = lambda p: '{"sub": ["1","2","3","4","5","6"]}'
    assert len(decompose("q", max_n=3, gen=gen)) == 3


def test_decompose_json_invalido_ou_vazio_volta_para_a_pergunta():
    for raw in ("sem json", '{"sub": "nao é lista"}', '{"sub": []}', '{"sub": [1, 2]}', "{quebrado"):
        assert decompose("pergunta?", gen=lambda p, r=raw: r) == ["pergunta?"]


def test_decompose_erro_do_llm_volta_para_a_pergunta():
    def boom(p):
        raise RuntimeError("grok caiu")
    assert decompose("pergunta?", gen=boom) == ["pergunta?"]


def test_intercalar_round_robin_sem_repetir_trecho():
    a = [Doc("a", 0), Doc("a", 1), Doc("x", 0)]
    b = [Doc("b", 0), Doc("x", 0), Doc("b", 1)]
    out = intercalar([a, b], 5)
    assert [(d.metadata["video_id"], d.metadata["chunk"]) for d in out] == [("a", 0), ("b", 0), ("a", 1), ("x", 0), ("b", 1)]


def test_intercalar_respeita_k():
    assert len(intercalar([[Doc("a", i) for i in range(5)], [Doc("b", i) for i in range(5)]], 3)) == 3


def test_retrieve_decomposto_uma_busca_por_faceta():
    chamadas = []

    def fake(q, k, **kw):
        chamadas.append((q, k))
        return [Doc(q, 0), Doc(q, 1)]
    docs, subs = retrieve_decomposto("composta", 4, retrieve_fn=fake, gen=lambda p: '{"sub": ["f1", "f2"]}')
    assert subs == ["f1", "f2"] and [c[0] for c in chamadas] == ["f1", "f2"]
    assert {d.metadata["video_id"] for d in docs} == {"f1", "f2"} and len(docs) == 4


def test_retrieve_decomposto_pergunta_simples_busca_uma_vez():
    fake = MagicMock(return_value=[Doc("s", 0)])
    docs, subs = retrieve_decomposto("simples", 6, retrieve_fn=fake, gen=lambda p: '{"sub": ["simples"]}')
    fake.assert_called_once()
    assert subs == ["simples"] and len(docs) == 1

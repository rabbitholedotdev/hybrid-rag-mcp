"""`retrieve(busca_k=, dedup_pool=, fetch_k_sintetico=)` (R5, fase 2, tentativa 2): busca mais funda que o pool,
sem texto repetido antes do rerank, pool sintético com tamanho próprio. Tudo desligado por padrão."""
from unittest.mock import MagicMock, patch

from rag.query import _unicos_pool, retrieve


class FakeDoc:
    def __init__(self, text, source_type="video", vid=None):
        self.page_content = text
        self.metadata = {"source_type": source_type, "video_id": vid or text, "chunk": 0}


def _vs(principal, sintetico=()):
    vs = MagicMock()
    vs.similarity_search_with_score.side_effect = [list(principal), list(sintetico)]
    return vs


def _roda(vs, **kw):
    vistos = []

    def fn(q, textos):
        vistos.append(list(textos))
        return [float(len(textos) - i) for i in range(len(textos))]
    with patch("rag.query.get_vectorstore", return_value=vs):
        docs = retrieve("q", k=3, reranker_fn=fn, **kw)
    return docs, vistos[0], vs.similarity_search_with_score.call_args_list


def test_unicos_pool_mantem_ordem_e_texto_diferente():
    pares = [(FakeDoc("a  b", vid="1"), 0.9), (FakeDoc("a b", vid="2"), 0.8), (FakeDoc("c"), 0.7)]
    assert [d.page_content for d, _ in _unicos_pool(pares)] == ["a  b", "c"]


def test_padrao_nao_muda():
    vs = _vs([(FakeDoc(f"t{i}"), 1.0) for i in range(5)])
    _, textos, chamadas = _roda(vs, fetch_k=5, fetch_k_sintetico=None)
    assert chamadas[0].kwargs["k"] == 5 and chamadas[1].kwargs["k"] == 12
    assert textos == [f"t{i}" for i in range(5)]


def test_busca_k_busca_fundo_e_corta_no_pool():
    vs = _vs([(FakeDoc(f"t{i}"), 1.0) for i in range(8)])
    _, textos, chamadas = _roda(vs, fetch_k=4, busca_k=8)
    assert chamadas[0].kwargs["k"] == 8
    assert textos == [f"t{i}" for i in range(4)]


def test_dedup_pool_preenche_com_unicos_antes_do_rerank():
    principal = [(FakeDoc(t, vid=str(i)), 1.0) for i, t in enumerate(["a", "a", "b", "b", "c", "d", "e"])]
    _, textos, _ = _roda(_vs(principal), fetch_k=4, busca_k=7, dedup_pool=True)
    assert textos == ["a", "b", "c", "d"]


def test_fetch_k_sintetico_dimensiona_o_pool_sintetico():
    vs = _vs([(FakeDoc("a"), 1.0)], [(FakeDoc("s", "resumo"), 1.0)])
    _, _, chamadas = _roda(vs, fetch_k=5, fetch_k_sintetico=7)
    assert chamadas[1].kwargs["k"] == 7

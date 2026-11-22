"""`retrieve(trace=, exact=)` para o funil (R5): sem `trace` o caminho default não muda;
com `trace` cada etapa fica registrada; `exact` vira `SearchParams(exact=True)` em toda busca."""
from unittest.mock import MagicMock, patch

from langchain_qdrant import RetrievalMode
from qdrant_client.http import models as qmodels

from rag.query import retrieve


class FakeDoc:
    def __init__(self, text, source_type="video"):
        self.page_content = text
        self.metadata = {"source_type": source_type, "video_id": text, "chunk": 0}


def _vs(*respostas):
    vs = MagicMock()
    vs.similarity_search_with_score.side_effect = list(respostas)
    return vs


def _cenario():
    hib = _vs([(FakeDoc("a"), 0.9), (FakeDoc("b"), 0.8), (FakeDoc("c"), 0.7)],
              [(FakeDoc("s1", "resumo"), 0.6), (FakeDoc("a"), 0.9)])
    den = _vs([(FakeDoc("a"), 0.9), (FakeDoc("d"), 0.5)])
    esp = _vs([(FakeDoc("c"), 0.9)])
    por_modo = {RetrievalMode.HYBRID: hib, RetrievalMode.DENSE: den, RetrievalMode.SPARSE: esp}
    get_vs = patch("rag.query.get_vectorstore",
                   side_effect=lambda create=False, collection=None, retrieval_mode=RetrievalMode.HYBRID:
                   por_modo[retrieval_mode])
    return hib, den, esp, get_vs


def _fn(q, textos):  # inverte a ordem de entrada
    return [float(i) for i in range(len(textos))]


def test_sem_trace_mesmas_chamadas_e_ordem():
    hib, den, esp, get_vs = _cenario()
    with get_vs:
        docs = retrieve("q", k=2, reranker_fn=_fn)
    assert hib.similarity_search_with_score.call_count == 2
    assert den.similarity_search_with_score.call_count == 0
    assert esp.similarity_search_with_score.call_count == 0
    assert [d.page_content for d in docs] == ["s1", "c"]


def test_trace_preenche_etapas():
    hib, den, esp, get_vs = _cenario()
    trace = {}
    with get_vs:
        docs = retrieve("q", k=2, reranker_fn=_fn, trace=trace)
    for chave in ("denso", "esparso", "fusao", "sintetico", "pool", "rerank", "final", "config"):
        assert chave in trace
    assert trace["final"] == docs
    assert [d.page_content for d in trace["rerank"]][:2] == [d.page_content for d in docs]
    assert [d.page_content for d in trace["denso"]] == ["a", "d"]
    assert [d.page_content for d in trace["esparso"]] == ["c"]
    assert [d.page_content for d in trace["fusao"]] == ["a", "b", "c"]
    assert [d.page_content for d in trace["sintetico"]] == ["s1", "a"]
    assert [d.page_content for d in trace["pool"]] == ["a", "b", "c", "s1"]
    assert trace["config"]["cand"] == 30
    assert trace["config"]["cand_sintetico"] == 12
    assert trace["config"]["rerank"] is True and trace["config"]["exact"] is False
    assert trace["config"]["pools_separados"] is False


def test_exact_vai_para_search_params():
    hib, den, esp, get_vs = _cenario()
    with get_vs:
        retrieve("q", k=2, reranker_fn=_fn, exact=True, trace={})
    chamadas = (hib.similarity_search_with_score.call_args_list + den.similarity_search_with_score.call_args_list
                + esp.similarity_search_with_score.call_args_list)
    assert len(chamadas) == 4
    assert all(c.kwargs["search_params"] == qmodels.SearchParams(exact=True) for c in chamadas)
    hib, den, esp, get_vs = _cenario()
    with get_vs:
        retrieve("q", k=2, reranker_fn=_fn)
    assert all(c.kwargs["search_params"] is None for c in hib.similarity_search_with_score.call_args_list)

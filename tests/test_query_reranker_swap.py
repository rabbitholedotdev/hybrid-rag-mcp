"""Ponto de troca do reranker em retrieve() (spec-fase2.md R3): `reranker_fn`
opcional; sem ele, comportamento idêntico ao default (Jina via
get_reranker())."""
from unittest.mock import MagicMock, patch

from rag.query import retrieve


class FakeDoc:
    def __init__(self, text, source_type="video"):
        self.page_content = text
        self.metadata = {"source_type": source_type, "video_id": text, "chunk": 0}


def _vs(pares):
    vs = MagicMock()
    vs.similarity_search_with_score.return_value = pares
    return vs


def test_sem_reranker_fn_usa_o_reranker_padrao():
    padrao = MagicMock()
    padrao.rerank.return_value = [0.1, 0.9]
    with patch("rag.vectorstore.get_reranker", return_value=padrao), \
         patch("rag.query.get_vectorstore", return_value=_vs([(FakeDoc("a"), 0.9), (FakeDoc("b"), 0.8)])):
        docs = retrieve("q", k=2, priorizar_sintese=False)
    padrao.rerank.assert_called_once()
    assert [d.page_content for d in docs] == ["b", "a"]


def test_reranker_fn_substitui_o_padrao_e_nem_carrega_o_jina():
    fn = MagicMock(return_value=[0.1, 0.9])  # inverte a ordem
    with patch("rag.vectorstore.get_reranker") as get_r, \
         patch("rag.query.get_vectorstore", return_value=_vs([(FakeDoc("a"), 0.9), (FakeDoc("b"), 0.8)])):
        docs = retrieve("q", k=2, priorizar_sintese=False, reranker_fn=fn)
    fn.assert_called_once()
    get_r.assert_not_called()
    assert [d.page_content for d in docs] == ["b", "a"]


def test_reranker_fn_recebe_pergunta_e_textos():
    fn = MagicMock(return_value=[0.5, 0.4])
    with patch("rag.vectorstore.get_reranker"), \
         patch("rag.query.get_vectorstore", return_value=_vs([(FakeDoc("a"), 0.9), (FakeDoc("b"), 0.8)])):
        retrieve("minha pergunta", k=2, priorizar_sintese=False, reranker_fn=fn)
    assert fn.call_args.args == ("minha pergunta", ["a", "b"])


def test_relevance_score_espelha_o_rerank_e_e_none_sem_reranker():
    fn = MagicMock(return_value=[0.1, 0.9])
    with patch("rag.query.get_vectorstore", return_value=_vs([(FakeDoc("a"), 0.9), (FakeDoc("b"), 0.8)])), \
         patch("rag.vectorstore.get_reranker"):
        docs = retrieve("q", k=2, priorizar_sintese=False, reranker_fn=fn)
    assert docs and all(d.metadata["relevance_score"] == d.metadata["_rerank"] for d in docs)
    with patch("rag.query.get_vectorstore", return_value=_vs([(FakeDoc("a"), 0.9), (FakeDoc("b"), 0.8)])):
        docs = retrieve("q", k=2, priorizar_sintese=False, rerank=False)
    assert docs and all(d.metadata["relevance_score"] is None for d in docs)

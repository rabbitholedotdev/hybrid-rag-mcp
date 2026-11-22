"""retrieve() em várias coleções: um pool por coleção, rerank único e `_colecao` nos docs;
com uma coleção só, nada muda (sem `_colecao`, mesma chamada de get_vectorstore)."""
from unittest.mock import MagicMock, patch

import pytest

from rag.query import retrieve


@pytest.fixture(autouse=True)
def _sem_modelos_reais():
    # o aquecimento antes das threads não pode carregar e5/BM25 de verdade nos testes
    with patch("rag.vectorstore.get_dense_embeddings"), patch("rag.vectorstore.get_sparse_embeddings"), \
         patch("rag.vectorstore.get_client"):
        yield


class FakeDoc:
    def __init__(self, text, vid=None):
        self.page_content = text
        self.metadata = {"source_type": "video", "video_id": vid or text, "chunk": 0}


def _vs_por_colecao(mapa, vid=None):
    def get(create=False, collection=None):
        vs = MagicMock()
        vs.similarity_search_with_score.return_value = [(FakeDoc(t, vid), 0.5) for t in mapa[collection]]
        return vs
    return get


def test_varias_colecoes_juntam_pools_rerank_unico_e_marcam_colecao():
    reranker = MagicMock()
    reranker.rerank.side_effect = lambda q, textos, **kw: [{"a": 0.1, "b": 0.9, "c": 0.5}[t] for t in textos]
    get = _vs_por_colecao({"other_rag": ["a"], "col_b": ["b"], "col_c": ["c"]})
    with patch("rag.vectorstore.get_reranker", return_value=reranker), patch("rag.query.get_vectorstore", side_effect=get):
        docs = retrieve("q", k=3, priorizar_sintese=False,
                        collections=["other_rag", "col_b", "col_c"])
    reranker.rerank.assert_called_once()  # um rerank só, sobre o pool das três
    assert [d.page_content for d in docs] == ["b", "c", "a"]
    assert {d.page_content: d.metadata["_colecao"] for d in docs} == {
        "a": "other_rag", "b": "col_b", "c": "col_c"}


def test_mesmo_video_e_chunk_em_colecoes_diferentes_nao_se_apagam():
    # mesmo (video_id, chunk), conteúdo diferente: a chave de dedup inclui a coleção
    get = _vs_por_colecao({"x": ["um texto"], "y": ["outro texto bem diferente"]}, vid="v1")
    with patch("rag.vectorstore.get_reranker", return_value=None), patch("rag.query.get_vectorstore", side_effect=get):
        docs = retrieve("q", k=4, priorizar_sintese=False, rerank=False, collections=["x", "y"])
    assert sorted(d.metadata["_colecao"] for d in docs) == ["x", "y"]


def test_pool_total_nao_cresce_com_o_numero_de_colecoes():
    vs = MagicMock()
    vs.similarity_search_with_score.return_value = []
    with patch("rag.vectorstore.get_reranker", return_value=MagicMock()), \
         patch("rag.query.get_vectorstore", return_value=vs):
        retrieve("q", k=6, priorizar_sintese=False, collections=["a", "b", "c"])
    ks = [c.kwargs["k"] for c in vs.similarity_search_with_score.call_args_list]
    assert ks == [10, 10, 10] and sum(ks) <= 30  # cand = max(6*5, 30) = 30, dividido por 3


def test_uma_colecao_nao_marca_colecao_e_chama_get_vectorstore_como_antes():
    vs = MagicMock()
    vs.similarity_search_with_score.return_value = [(FakeDoc("a"), 0.5)]
    with patch("rag.vectorstore.get_reranker", return_value=None), \
         patch("rag.query.get_vectorstore", return_value=vs) as get:
        docs = retrieve("q", k=2, priorizar_sintese=False, rerank=False, collections=["so_uma"])
    get.assert_called_once_with(create=False)
    assert "_colecao" not in docs[0].metadata


def test_sem_reranker_ordena_por_score_entre_colecoes():
    def get(create=False, collection=None):
        vs = MagicMock()
        vs.similarity_search_with_score.return_value = [(FakeDoc(f"{collection}-{i}"), s)
                                                       for i, s in enumerate({"x": [0.2, 0.1], "y": [0.9, 0.8]}[collection])]
        return vs
    with patch("rag.vectorstore.get_reranker", return_value=None), patch("rag.query.get_vectorstore", side_effect=get):
        docs = retrieve("q", k=2, priorizar_sintese=False, rerank=False, collections=["x", "y"])
    assert {d.metadata["_colecao"] for d in docs} == {"y"}  # os dois melhores vêm de y, não da 1ª coleção


def test_uma_colecao_fora_do_ar_nao_derruba_as_outras(capsys):
    def get(create=False, collection=None):
        if collection == "fora":
            raise RuntimeError("Coleção 'fora' não existe.")
        vs = MagicMock()
        vs.similarity_search_with_score.return_value = [(FakeDoc("ok"), 0.5)]
        return vs
    with patch("rag.vectorstore.get_reranker", return_value=None), patch("rag.query.get_vectorstore", side_effect=get):
        docs = retrieve("q", k=2, priorizar_sintese=False, rerank=False, collections=["fora", "boa"])
    assert [d.metadata["_colecao"] for d in docs] == ["boa"]
    assert "fora" in capsys.readouterr().err


def test_todas_fora_do_ar_levanta():
    import pytest
    with patch("rag.vectorstore.get_reranker", return_value=None), \
         patch("rag.query.get_vectorstore", side_effect=RuntimeError("sem Qdrant")):
        with pytest.raises(RuntimeError, match="todas as coleções"):
            retrieve("q", k=2, priorizar_sintese=False, rerank=False, collections=["a", "b"])


def test_expand_context_le_a_aula_na_colecao_de_origem():
    from rag.query import expand_context
    doc = FakeDoc("t", vid="v1")
    doc.metadata["_colecao"] = "col_b"
    with patch("rag.query.get_lesson_chunks", return_value=[{"chunk": 0, "text": "t"}]) as g:
        expand_context(doc, client="c")
    g.assert_called_once_with("v1", client="c", collection="col_b")

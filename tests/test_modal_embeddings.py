"""EMBED_BACKEND=modal: documentos vão ao app Modal em lotes (ordem preservada); a consulta
fica na CPU local. Sem Modal de verdade: o SDK e o fastembed são dublês."""
import sys
from unittest.mock import MagicMock, patch

import rag.vectorstore as vs
from rag.modal_embeddings import ModalDenseEmbeddings


def _modal_falso(vetor_de):
    remoto = MagicMock()
    remoto.embed.map.side_effect = lambda lotes: ([vetor_de(t) for t in lote] for lote in lotes)
    modal = MagicMock()
    modal.Cls.from_name.return_value.return_value = remoto
    return modal, remoto


def test_documentos_em_lotes_e_ordem_preservada():
    modal, remoto = _modal_falso(lambda t: [float(len(t))])
    with patch.dict(sys.modules, {"modal": modal}), patch("rag.modal_embeddings.LOTE", 2):
        out = ModalDenseEmbeddings("intfloat/multilingual-e5-large").embed_documents(["a", "bb", "ccc", "dddd", "eeeee"])
    assert out == [[1.0], [2.0], [3.0], [4.0], [5.0]]
    assert [len(l) for l in remoto.embed.map.call_args.args[0]] == [2, 2, 1]
    modal.Cls.from_name.assert_called_once_with("rag-embed-e5", "Embedder")


def test_consulta_fica_na_cpu_local_e_nao_chama_o_modal():
    modal, remoto = _modal_falso(lambda t: [0.0])
    local = MagicMock()
    local.return_value.embed_query.return_value = [0.5]
    fake_fe = MagicMock(FastEmbedEmbeddings=local)
    with patch.dict(sys.modules, {"modal": modal, "langchain_community.embeddings.fastembed": fake_fe}):
        assert ModalDenseEmbeddings("intfloat/multilingual-e5-large").embed_query("q") == [0.5]
    remoto.embed.map.assert_not_called()


def test_get_dense_embeddings_escolhe_o_backend_pela_env(monkeypatch):
    monkeypatch.setattr(vs, "_dense", None)
    monkeypatch.setenv("EMBED_BACKEND", "modal")
    modal, _ = _modal_falso(lambda t: [0.0])
    with patch.dict(sys.modules, {"modal": modal}):
        assert isinstance(vs.get_dense_embeddings(), ModalDenseEmbeddings)
    monkeypatch.setattr(vs, "_dense", None)


def test_sem_env_segue_o_fastembed_local(monkeypatch):
    monkeypatch.setattr(vs, "_dense", None)
    monkeypatch.delenv("EMBED_BACKEND", raising=False)
    with patch.object(vs, "FastEmbedEmbeddings") as fe:
        vs.get_dense_embeddings()
    fe.assert_called_once()
    monkeypatch.setattr(vs, "_dense", None)


def test_modelo_diferente_do_app_modal_levanta():
    import pytest
    modal, _ = _modal_falso(lambda t: [0.0])
    with patch.dict(sys.modules, {"modal": modal}), pytest.raises(ValueError, match="só embute"):
        ModalDenseEmbeddings("outro/modelo")


def test_lista_vazia_nao_chama_o_modal():
    modal, remoto = _modal_falso(lambda t: [0.0])
    with patch.dict(sys.modules, {"modal": modal}):
        assert ModalDenseEmbeddings("intfloat/multilingual-e5-large").embed_documents([]) == []

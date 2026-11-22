from unittest.mock import patch, MagicMock

import numpy as np

import rag.vectorstore


def test_bge_m3_query_uses_the_same_mean_pooling_as_late_chunking():
    """Query e documento têm que sair do MESMO pooling: o documento é
    mean-pooled por span (late chunking) e não tem CLS por span. Antes a
    query era CLS (fastembed pooling=CLS) — dois espaços vetoriais
    diferentes, cosseno sem significado, e nenhum teste pegava."""
    from rag.vectorstore import BgeM3Embeddings
    from rag.late_chunking import pool_spans

    # <s>, 2 tokens, </s>. CLS seria a linha 0; MEAN é a média das 4.
    hidden = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]])
    fake_encoder = MagicMock()
    fake_encoder.embed_whole.side_effect = lambda t: pool_spans(hidden, [(0, 4)])[0]

    with patch("rag.late_chunking.get_encoder", return_value=fake_encoder):
        emb = BgeM3Embeddings()
        v = emb.embed_query("pergunta")

    # o vetor da query é exatamente o mesmo pool_spans que o do documento
    np.testing.assert_allclose(v, pool_spans(hidden, [(0, 4)])[0], atol=1e-6)
    assert not np.allclose(v, hidden[0])  # e NÃO é o CLS
    assert isinstance(v, list) and all(isinstance(x, float) for x in v)


def test_bge_m3_embeddings_forwards_providers():
    """FASTEMBED_CUDA=1 tem que continuar chegando no ONNX Runtime."""
    from rag.vectorstore import BgeM3Embeddings

    with patch("rag.late_chunking.get_encoder") as fake_get:
        BgeM3Embeddings(providers=["CUDAExecutionProvider"])
    fake_get.assert_called_once_with(providers=["CUDAExecutionProvider"])


def test_get_dense_embeddings_keeps_fastembed_for_e5():
    """O caminho e5 (modelo nativo do fastembed) não muda."""
    rag.vectorstore._dense = None
    with patch.object(rag.vectorstore, "EMBED_MODEL", "intfloat/multilingual-e5-large"), \
         patch.object(rag.vectorstore, "FastEmbedEmbeddings") as fake_fe, \
         patch.object(rag.vectorstore, "BgeM3Embeddings") as fake_bge:
        rag.vectorstore.get_dense_embeddings()
    rag.vectorstore._dense = None

    fake_fe.assert_called_once_with(model_name="intfloat/multilingual-e5-large")
    fake_bge.assert_not_called()


def test_get_dense_embeddings_uses_bge_adapter_for_bge_m3():
    rag.vectorstore._dense = None
    with patch.object(rag.vectorstore, "EMBED_MODEL", "BAAI/bge-m3"), \
         patch.object(rag.vectorstore, "FastEmbedEmbeddings") as fake_fe, \
         patch.object(rag.vectorstore, "BgeM3Embeddings") as fake_bge:
        rag.vectorstore.get_dense_embeddings()
    rag.vectorstore._dense = None

    fake_bge.assert_called_once_with(providers=None)
    fake_fe.assert_not_called()


def test_vectorstore_module_does_not_import_fastembed_at_top_level():
    """fastembed.common.model_description é layout interno da lib: no topo
    do módulo, uma mudança de versão derrubava o import do arquivo inteiro,
    inclusive o caminho e5, que não usa nada disso. Importes de fastembed
    ficam dentro da função que precisa deles (padrão do get_reranker)."""
    import ast

    src = open(rag.vectorstore.__file__, encoding="utf-8").read()
    top_level = [
        n for n in ast.parse(src).body if isinstance(n, (ast.Import, ast.ImportFrom))
    ]
    modules = [
        n.module for n in top_level if isinstance(n, ast.ImportFrom)
    ] + [a.name for n in top_level if isinstance(n, ast.Import) for a in n.names]
    assert not [m for m in modules if m and m.split(".")[0] == "fastembed"], modules

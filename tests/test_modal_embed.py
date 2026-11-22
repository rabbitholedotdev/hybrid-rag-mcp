"""Mock: forma do payload de scripts/modal_embed.Embedder, sem GPU/rede.
`.local()` do Modal roda a classe em processo (dispara @modal.enter na
primeira chamada) sem precisar de deploy nem de container real.

`modal` não está em requirements.txt (o Dockerfile da imagem do MCP não
precisa do SDK) — sem o client instalado, pular só este
arquivo em vez de derrubar a coleta do pytest inteiro."""
from unittest.mock import patch

import numpy as np
import pytest

pytest.importorskip("modal")


@pytest.fixture
def mock_encoder():
    with patch("rag.late_chunking.LateChunkEncoder") as mock_cls:
        inst = mock_cls.return_value
        # provider ativo default = CUDA primeiro (caminho feliz); testes que
        # querem simular o fallback silencioso sobrescrevem isto.
        inst.session.get_providers.return_value = [
            "CUDAExecutionProvider", "CPUExecutionProvider"
        ]
        yield inst, mock_cls


def _embedder():
    from scripts.modal_embed import Embedder
    return Embedder()


def test_embed_documents_shape(mock_encoder):
    inst, _ = mock_encoder
    inst.encode_document.return_value = [
        ("primeiro chunk", 0, np.array([1.0, 0.0, 0.0])),
        ("segundo chunk", 15, np.array([0.0, 1.0, 0.0])),
    ]

    out = _embedder().embed_documents.local(
        [{"id": "doc-1", "text": "primeiro chunk segundo chunk"}], max_chunk_tokens=256
    )

    assert out == [{
        "id": "doc-1",
        "chunks": [
            {"text": "primeiro chunk", "char_pos": 0, "vector": [1.0, 0.0, 0.0]},
            {"text": "segundo chunk", "char_pos": 15, "vector": [0.0, 1.0, 0.0]},
        ],
    }]
    inst.encode_document.assert_called_once_with(
        "primeiro chunk segundo chunk", max_chunk_tokens=256
    )


def test_embed_texts_shape(mock_encoder):
    inst, _ = mock_encoder
    inst.embed_whole.side_effect = [np.array([0.1, 0.2]), np.array([0.3, 0.4])]

    out = _embedder().embed_texts.local(["a", "b"])

    assert out == [[0.1, 0.2], [0.3, 0.4]]
    assert inst.embed_whole.call_count == 2


def test_enter_uses_cuda_then_cpu_providers(mock_encoder):
    inst, mock_cls = mock_encoder

    _embedder().embed_texts.local(["x"])

    mock_cls.assert_called_once_with(providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    # não basta pedir CUDA — @modal.enter tem que checar qual ficou ATIVO
    # de fato (é o que pegou o bug real: onnxruntime-gpu>=1.23 caindo pro
    # CPU em silêncio numa GPU real, sem levantar erro).
    inst.session.get_providers.assert_called()


def test_enter_raises_if_cuda_not_actually_active(mock_encoder):
    """Reproduz o bug real encontrado nesta task: providers pedidos incluíam
    CUDA, mas o provider ATIVO era só CPU (onnxruntime caiu em silêncio).
    @modal.enter precisa falhar alto nessa situação, não deixar passar."""
    inst, _ = mock_encoder
    inst.session.get_providers.return_value = ["CPUExecutionProvider"]

    with pytest.raises(AssertionError, match="CUDAExecutionProvider"):
        _embedder().embed_texts.local(["x"])

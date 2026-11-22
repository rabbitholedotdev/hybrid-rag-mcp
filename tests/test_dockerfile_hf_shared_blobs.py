"""Regressão: `buscar_transcricoes`
quebrava com `onnxruntime.../Fail: ... External data path escapes model
directory` — o cache-wide shared blob store do huggingface_hub symlinka o
.onnx e o .onnx_data (modelo sharded, ex. e5-large/reranker Jina) em
subdiretórios blobs/<prefixo-do-hash> DIFERENTES, e o onnxruntime rejeita
porque o external-data não resolve dentro do diretório do .onnx.

Sem dependências pesadas (fastembed/onnxruntime) de propósito: só lê o
Dockerfile como texto, roda com o Python do sistema, sem precisar do venv
do projeto instalado.
"""
from pathlib import Path

DOCKERFILE = Path(__file__).resolve().parent.parent / "Dockerfile"


def test_dockerfile_disables_hf_hub_shared_blobs():
    env_line = next(
        line for line in DOCKERFILE.read_text(encoding="utf-8").splitlines()
        if line.startswith("ENV ")
    )
    assert "HF_HUB_DISABLE_SHARED_BLOBS=1" in env_line, (
        "Dockerfile precisa desligar o shared blob store do huggingface_hub "
        "(HF_HUB_DISABLE_SHARED_BLOBS=1), senão modelo ONNX sharded (e5-large, "
        "reranker Jina) volta a quebrar com 'External data path escapes model "
        "directory' — ver rag/late_chunking.py (mesma causa raiz, já corrigida "
        "pro BGE-M3 com local_dir explícito)."
    )

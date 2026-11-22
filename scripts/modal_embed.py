"""App Modal `rag-embed-bgem3`: embedding pontual (BGE-M3 + late chunking)
numa GPU sob demanda.

Embarca só `rag/late_chunking.py` + `rag/sentence_splitting.py` (não o
pacote `rag` inteiro, que puxa langchain/config — ver o docstring desses
dois arquivos). Nenhum endpoint HTTP: uso interno via SDK
(`modal.Cls.from_name("rag-embed-bgem3", "Embedder")` ou `modal run`).

Deploy:  modal deploy scripts/modal_embed.py
Smoke:   modal run scripts/modal_embed.py
"""
from pathlib import Path

import modal

RAG_DIR = Path(__file__).resolve().parent.parent / "rag"
HF_HOME = "/cache/hf"

# CUDA base used for the Modal embedder image.
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04", add_python="3.11"
    )
    # Pinado: onnxruntime-gpu >=1.23 passou a exigir CUDA 13 (medido nesta
    # task — "Failed to create CUDAExecutionProvider... Require CUDA 13.*"
    # com a imagem 12.6.3, caindo silencioso pro CPUExecutionProvider). 1.20
    # é a versão que o rag-worker (mesma base CUDA 12.6.3-cudnn-runtime)
    # validou contra GPU real.
    .pip_install("onnxruntime-gpu==1.20.2", "tokenizers", "huggingface_hub", "numpy")
    # HF_HOME aponta pro volume persistente (ver `volumes=` abaixo) — sem
    # isso os ~2.3GB do BGE-M3 baixam nem toda vez que o container recria.
    # DISABLE_SYMLINKS pelo mesmo motivo documentado em rag/late_chunking.py
    # (BGE_M3_LOCAL_DIR): o export ONNX é sharded e o validador de
    # external-data do onnxruntime rejeita symlink pro blob store do HF.
    .env({"HF_HOME": HF_HOME, "HF_HUB_DISABLE_SYMLINKS": "1"})
    .add_local_file(RAG_DIR / "__init__.py", "/root/rag/__init__.py")
    .add_local_file(RAG_DIR / "sentence_splitting.py", "/root/rag/sentence_splitting.py")
    .add_local_file(RAG_DIR / "late_chunking.py", "/root/rag/late_chunking.py")
)

# Cache do modelo entre deploys/containers — sem isso cada container frio
# rebaixa os ~2.3GB do BGE-M3 (fatos medidos do plano).
model_volume = modal.Volume.from_name("bge-m3-cache", create_if_missing=True)

app = modal.App("rag-embed-bgem3", image=image)


@app.cls(
    gpu="A10G",
    volumes={HF_HOME: model_volume},
    timeout=1800,
    # sem min_containers/buffer_containers: escala a zero quando ocioso
    # (custo parado — exigência do brief). scaledown_window curto porque
    # é uso pontual/lote, não serviço long-tail.
    scaledown_window=60,
)
class Embedder:
    @modal.enter()
    def load(self):
        import sys
        sys.path.insert(0, "/root")
        from rag.late_chunking import LateChunkEncoder

        self._encoder = LateChunkEncoder(
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        # onnxruntime cai pro CPUExecutionProvider sem erro fatal quando o
        # provider CUDA falha em carregar (medido nesta task: onnxruntime-gpu
        # >=1.23 exige CUDA 13, a imagem é 12.6 — rodava "certo", só que na
        # CPU, faturando GPU). Falha alto e imediato no boot do container em
        # vez de degradar em silêncio numa próxima bump de imagem/ORT.
        active = self._encoder.session.get_providers()
        assert active[0] == "CUDAExecutionProvider", (
            f"CUDAExecutionProvider não está ativo (providers={active}) — "
            "onnxruntime caiu pro CPU numa GPU real, ver comentário do pin "
            "de onnxruntime-gpu acima."
        )

    @modal.method()
    def embed_documents(self, docs: list[dict], max_chunk_tokens: int = 256) -> list[dict]:
        """docs: [{"id": str, "text": str}] -> [{"id", "chunks": [{"text","char_pos","vector"}]}]"""
        out = []
        for d in docs:
            spans = self._encoder.encode_document(d["text"], max_chunk_tokens=max_chunk_tokens)
            out.append({
                "id": d["id"],
                "chunks": [
                    {"text": text, "char_pos": char_pos, "vector": vector.tolist()}
                    for text, char_pos, vector in spans
                ],
            })
        return out

    @modal.method()
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Texto curto inteiro (sem late chunking) — usado pelos sintéticos resumo/qa."""
        return [self._encoder.embed_whole(t).tolist() for t in texts]


@app.local_entrypoint()
def main():
    """Smoke test manual: `modal run scripts/modal_embed.py`."""
    embedder = Embedder()
    docs = [{"id": "smoke-1", "text": "Isso é um teste. Frase dois aqui. E uma terceira."}]
    result = embedder.embed_documents.remote(docs, max_chunk_tokens=256)
    print(f"embed_documents: {len(result[0]['chunks'])} chunk(s), dim={len(result[0]['chunks'][0]['vector'])}")
    vecs = embedder.embed_texts.remote(["resumo curto de teste"])
    print(f"embed_texts: {len(vecs)} vetor(es), dim={len(vecs[0])}")

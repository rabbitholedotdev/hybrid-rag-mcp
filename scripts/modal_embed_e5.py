"""App Modal `rag-embed-e5`: embedding denso do e5-large (o mesmo modelo e o mesmo
fastembed da busca local) numa GPU sob demanda, só para INGESTÃO em lote.
A busca segue local. Os vetores saem do mesmo `TextEmbedding.embed`
que o `FastEmbedEmbeddings` do langchain chama, então são os mesmos da CPU; `scripts/check_embed_gpu.py`
mede isso (cosseno mínimo) antes de qualquer ingestão.

Deploy:  modal deploy scripts/modal_embed_e5.py
Smoke:   modal run scripts/modal_embed_e5.py
"""
import modal

HF_HOME = "/cache/hf"
MODELO = "intfloat/multilingual-e5-large"

# Mesma base CUDA do modal_embed.py (BGE-M3) e do rag-worker. onnxruntime-gpu pinado em 1.20.2:
# >=1.23 exige CUDA 13 e cai em silêncio para a CPU nesta imagem (medido no modal_embed.py).
image = (
    modal.Image.from_registry("nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04", add_python="3.11")
    .pip_install("onnxruntime-gpu==1.20.2", "fastembed-gpu==0.8.0", "numpy")
    .env({"HF_HOME": HF_HOME, "HF_HUB_DISABLE_SYMLINKS": "1"})
)

model_volume = modal.Volume.from_name("e5-large-cache", create_if_missing=True)

app = modal.App("rag-embed-e5", image=image)


@app.cls(
    gpu="A10G",
    volumes={HF_HOME: model_volume},
    timeout=1800,
    scaledown_window=60,  # escala a zero: sem custo parado
    max_containers=2,     # teto de gasto de uma ingestão desgovernada
)
class Embedder:
    @modal.enter()
    def load(self):
        from fastembed import TextEmbedding

        self._m = TextEmbedding(
            model_name=MODELO, cache_dir=HF_HOME,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        # onnxruntime cai na CPU sem erro quando o provider CUDA falha; falha alto aqui,
        # senão a ingestão fatura GPU e roda na CPU.
        ativos = self._m.model.model.get_providers()
        assert ativos[0] == "CUDAExecutionProvider", f"sem CUDA (providers={ativos})"

    @modal.method()
    def embed(self, textos: list[str]) -> list[list[float]]:
        """Passagens (sem prefixo), igual ao FastEmbedEmbeddings.embed_documents."""
        return [v.tolist() for v in self._m.embed(textos, batch_size=64)]


@app.local_entrypoint()
def main():
    v = Embedder().embed.remote(["Isso é um teste.", "Segundo texto de teste."])
    print(f"{len(v)} vetores, dim={len(v[0])}")

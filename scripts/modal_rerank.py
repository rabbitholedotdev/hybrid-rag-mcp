"""App Modal `rag-rerank-eval`: 2 rerankers candidatos (Apache-2.0) numa GPU
sob demanda, só para o eval — o retrieve default não usa isto.
Mesmo padrão de escala-a-zero de `scripts/modal_embed.py`.

Deploy:  modal deploy scripts/modal_rerank.py
Smoke:   modal run scripts/modal_rerank.py
"""
import modal

HF_HOME = "/cache/hf"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "transformers", "sentence-transformers")
    .env({"HF_HOME": HF_HOME})
)

model_volume = modal.Volume.from_name("rerank-eval-cache", create_if_missing=True)

app = modal.App("rag-rerank-eval", image=image)

MODELOS = {
    "bge-v2-m3": "BAAI/bge-reranker-v2-m3",
    "qwen3-0.6b": "Qwen/Qwen3-Reranker-0.6B",
}


@app.cls(gpu="A10G", volumes={HF_HOME: model_volume}, timeout=1800, scaledown_window=60)
class Reranker:
    @modal.enter()
    def load(self):
        from sentence_transformers import CrossEncoder
        self._modelos = {
            nome: CrossEncoder(repo, device="cuda", max_length=1024)
            for nome, repo in MODELOS.items()
        }

    @modal.method()
    def rerank(self, nome_modelo: str, query: str, documentos: list[str]) -> list[float]:
        ce = self._modelos[nome_modelo]
        return [float(s) for s in ce.predict([(query, d) for d in documentos])]


@app.local_entrypoint()
def main():
    """Smoke test manual: `modal run scripts/modal_rerank.py`."""
    r = Reranker()
    docs = ["O curso tem 8 módulos.", "Receita de bolo de cenoura.", "O framework é o AIDA."]
    for nome in MODELOS:
        scores = r.rerank.remote(nome, "quantos módulos tem o curso?", docs)
        print(f"{nome}: {scores}")

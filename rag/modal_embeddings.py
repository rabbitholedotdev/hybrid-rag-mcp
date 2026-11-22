"""Embedding denso do e5-large na GPU do Modal (app `rag-embed-e5`, scripts/modal_embed_e5.py).
Só para ingestão em lote (EMBED_BACKEND=modal): `embed_documents` vai ao Modal; `embed_query`
segue na CPU local, porque a busca é uma consulta por vez e o Modal só acrescentaria latência."""
import os
from langchain_core.embeddings import Embeddings

LOTE = int(os.getenv("MODAL_EMBED_LOTE", "256"))
MODELO_DO_APP = "intfloat/multilingual-e5-large"  # fixo em scripts/modal_embed_e5.py


class ModalDenseEmbeddings(Embeddings):
    def __init__(self, model_name: str):
        if model_name != MODELO_DO_APP:
            raise ValueError(f"EMBED_BACKEND=modal só embute {MODELO_DO_APP}, não {model_name}")
        import modal  # import tardio: só quem ingere com EMBED_BACKEND=modal precisa do SDK
        self._modelo = model_name
        self._remoto = modal.Cls.from_name("rag-embed-e5", "Embedder")()
        self._local = None

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        lotes = [texts[i:i + LOTE] for i in range(0, len(texts), LOTE)]
        return [v for lote in self._remoto.embed.map(lotes) for v in lote]

    def embed_query(self, text: str) -> list[float]:
        if self._local is None:
            from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
            self._local = FastEmbedEmbeddings(model_name=self._modelo)
        return self._local.embed_query(text)

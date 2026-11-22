"""Cliente do app Modal `rag-rerank-eval` (scripts/modal_rerank.py), pro eval
trocar o reranker sem alterar o default do servidor. `RemoteReranker(nome).rerank` tem a
assinatura de `reranker_fn` que `rag.query.retrieve` aceita."""


class RemoteReranker:
    def __init__(self, nome_modelo: str):
        import modal  # import tardio: só quem usa --reranker != jina precisa do SDK
        self._nome = nome_modelo
        self._reranker = modal.Cls.from_name("rag-rerank-eval", "Reranker")()

    def rerank(self, query: str, documentos: list[str]) -> list[float]:
        return self._reranker.rerank.remote(self._nome, query, documentos)

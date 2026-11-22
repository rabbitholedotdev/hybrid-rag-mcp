"""Experimento (off by default): expansão do contexto: o documento certo costuma ser achado, mas a PASSAGEM certa dentro dele não.

Diagnóstico em um acervo longo, k=12: das facetas que faltavam, a maioria estava no documento esperado que
JÁ tinha chunk recuperado — vizinho, 2–3 chunks, ou mais longe.
Duas estratégias, ambas sem chamada extra ao LLM:
  - vizinhos: acrescenta chunk±janela de cada trecho recuperado (cobre os 29% a distância 1);
  - por documento: acha os documentos pela 1ª busca (inclusive via resumo/Q&A, que localizam o assunto) e busca
    DENTRO de cada um, rerankeando contra a pergunta (cobre as passagens distantes).
"""
import re

from langchain_core.documents import Document
from qdrant_client.http import models as qmodels

SINTETICO = ("resumo", "qa")
_SYN = re.compile(r"^syn-(?:resumo|qa)-")
_JANELA = re.compile(r"-p\d+$")


def base_id(video_id: str) -> str:
    """`syn-resumo-ebj-m-abc-p2` -> `ebj-m-abc`: o resumo/Q&A aponta para o documento-fonte."""
    return _JANELA.sub("", _SYN.sub("", video_id or ""))


def _chave(d):
    return (d.metadata.get("video_id"), d.metadata.get("chunk"))


def vizinhos(docs, k_semente: int, janela: int = 1, teto: int = 24, chunks_fn=None):
    """Os `k_semente` primeiros trechos PRIMÁRIOS + seus chunks vizinhos (±janela) do mesmo documento, sem repetir,
    na ordem do ranking, até `teto` chunks. Sem video_id/chunk no metadado, o trecho entra como está."""
    if chunks_fn is None:
        from rag.query import get_lesson_chunks as chunks_fn
    prim = [d for d in docs if (d.metadata.get("source_type") or "").lower() not in SINTETICO][:k_semente]
    out, vistos, cache = [], set(), {}
    for d in prim:
        vid, c = d.metadata.get("video_id"), d.metadata.get("chunk")
        grupo = [d]
        if vid is not None and c is not None:
            if vid not in cache:
                cache[vid] = chunks_fn(vid)
            for x in cache[vid]:
                if x["chunk"] != c and abs(x["chunk"] - c) <= janela:
                    grupo.append(Document(page_content=x["text"], metadata={**d.metadata, "chunk": x["chunk"], "_vizinho": True}))
        for g in sorted(grupo, key=lambda g: g.metadata.get("chunk") or 0):
            if _chave(g) not in vistos and len(out) < teto:
                vistos.add(_chave(g))
                out.append(g)
    return out


def documentos_do_ranking(docs, n_docs: int) -> list[str]:
    """Documentos-fonte na ordem em que aparecem no ranking (primário e sintético apontam para o mesmo)."""
    ordem = []
    for d in docs:
        b = base_id(d.metadata.get("video_id"))
        if b and b not in ordem:
            ordem.append(b)
    return ordem[:n_docs]


def por_documento(question: str, k_docs: int = 3, k_doc: int = 5, k_primeira: int = 12, retrieve_fn=None,
                  busca_no_doc=None):
    """1ª busca (k_primeira) só para localizar os documentos; depois, em cada um dos `k_docs` melhores, os `k_doc`
    chunks que melhor respondem à pergunta DENTRO dele. Devolve os chunks agrupados por documento, na ordem do ranking."""
    if retrieve_fn is None:
        from rag.query import retrieve as retrieve_fn
    if busca_no_doc is None:
        busca_no_doc = _busca_no_doc
    primeira = retrieve_fn(question, k_primeira)
    out, vistos = [], set()
    for vid in documentos_do_ranking(primeira, k_docs):
        for d in busca_no_doc(question, vid, k_doc):
            if _chave(d) not in vistos:
                vistos.add(_chave(d))
                out.append(d)
    return out or [d for d in primeira if (d.metadata.get("source_type") or "").lower() not in SINTETICO]


def _busca_no_doc(question: str, video_id: str, k: int):
    from rag.query import get_vectorstore
    from rag.vectorstore import get_reranker
    flt = qmodels.Filter(must=[qmodels.FieldCondition(key="metadata.video_id", match=qmodels.MatchValue(value=video_id))])
    pairs = get_vectorstore(create=False).similarity_search_with_score(question, k=max(k * 4, 20), filter=flt)
    docs = [d for d, _ in pairs]
    if not docs:
        return []
    scores = get_reranker().rerank(question, [d.page_content for d in docs])
    return [d for d, _ in sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)[:k]]


def _busca_em_docs(question: str, video_ids: list[str], k: int):
    """Os `k` chunks que melhor respondem à pergunta DENTRO do conjunto de documentos, rerankeados."""
    from rag.query import get_vectorstore
    from rag.vectorstore import get_reranker
    flt = qmodels.Filter(must=[qmodels.FieldCondition(key="metadata.video_id", match=qmodels.MatchAny(any=list(video_ids)))])
    pairs = get_vectorstore(create=False).similarity_search_with_score(question, k=max(k * 4, 24), filter=flt)
    docs = [d for d, _ in pairs]
    if not docs:
        return []
    scores = get_reranker().rerank(question, [d.page_content for d in docs])
    return [d for d, _ in sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)[:k]]


def por_subperguntas_no_documento(question: str, k_docs: int = 3, k_sub: int = 3, k_primeira: int = 12, teto: int = 15,
                                  gen=None, retrieve_fn=None, busca_em_docs=None):
    """1ª busca localiza os documentos (inclusive via resumo/Q&A); o LLM quebra a pergunta em sub-perguntas (1 a 4);
    cada sub-pergunta é buscada SÓ dentro desses documentos e as listas se juntam em round-robin até `teto` chunks.
    Pergunta de uma faceta: uma busca dentro dos documentos com a própria pergunta."""
    from rag.decompose import decompose, intercalar
    if retrieve_fn is None:
        from rag.query import retrieve as retrieve_fn
    busca_em_docs = busca_em_docs or _busca_em_docs
    primeira = retrieve_fn(question, k_primeira)
    docs = documentos_do_ranking(primeira, k_docs)
    if not docs:
        return [d for d in primeira if (d.metadata.get("source_type") or "").lower() not in SINTETICO]
    subs = decompose(question, gen=gen)
    if len(subs) == 1:
        return busca_em_docs(question, docs, teto)
    return intercalar([busca_em_docs(s, docs, k_sub) for s in subs], teto)

"""Variantes de recuperação para a comparação de orquestração (A..E), todas sobre o MESMO
`retrieve()` e o MESMO gerador (grok CLI, com cache em disco). Só o orquestrador muda.

  A baseline   retrieve(pergunta)
  B hyde       retrieve(pergunta, hyde=True)
  C decompose  rag.decompose.retrieve_decomposto (sub-perguntas, round-robin)
  D multiquery langchain_classic MultiQueryRetriever + rerank da união contra a pergunta original
  E langgraph  grafo decompose -> retrieve -> grade -> (lacunas -> retrieve) -> fim, máx. 2 voltas

Cada função: (pergunta, k, gen) -> (docs, {"llm_calls": n, ...}). langchain/langgraph só são
importados dentro de D e E: o runtime do rag-mcp não ganha dependência.
"""
import hashlib
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rag.decompose import decompose, intercalar, retrieve_decomposto  # noqa: E402
from rag.query import retrieve  # noqa: E402

CACHE_DIR = Path(os.getenv("EVAL_LLM_CACHE", Path(__file__).resolve().parent / ".llm_cache"))


class ContaChamadas:
    """Envolve um gerador: cache em disco por hash do prompt e contagem de chamadas reais ao LLM."""

    def __init__(self, gen):
        self.gen, self.chamadas = gen, 0
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

    def __call__(self, prompt: str) -> str:
        self.chamadas += 1  # conta a chamada lógica, mesmo vinda do cache
        f = CACHE_DIR / (hashlib.sha256(prompt.encode()).hexdigest()[:32] + ".json")
        if f.exists():
            return json.loads(f.read_text(encoding="utf-8"))["out"]
        out = self.gen(prompt)
        f.write_text(json.dumps({"out": out}, ensure_ascii=False), encoding="utf-8")
        return out


def variante_a(q, k, gen):
    return retrieve(q, k), {"llm_calls": 0}


def variante_b(q, k, gen):
    from rag import query as Q
    antigo = Q.gen_grok
    Q.gen_grok = gen  # o _hyde usa gen_grok: passa pelo cache e pela contagem
    try:
        return retrieve(q, k, hyde=True), {"llm_calls": 1}
    finally:
        Q.gen_grok = antigo


def variante_c(q, k, gen):
    docs, subs = retrieve_decomposto(q, k, gen=gen)
    return docs, {"llm_calls": 1, "sub": subs}


def variante_d(q, k, gen):
    """MultiQueryRetriever do LangChain: o LLM gera reformulações, o retriever busca cada uma e
    une sem repetir. A união é rerankeada contra a pergunta original (prática padrão) e cortada em k."""
    from langchain_classic.retrievers.multi_query import MultiQueryRetriever
    from langchain_core.callbacks import CallbackManagerForRetrieverRun
    from langchain_core.documents import Document
    from langchain_core.language_models.llms import LLM
    from langchain_core.retrievers import BaseRetriever
    from rag.vectorstore import get_reranker

    class GrokLLM(LLM):
        @property
        def _llm_type(self) -> str:
            return "grok-cli"

        def _call(self, prompt, stop=None, run_manager=None, **kw) -> str:
            return gen(prompt)

    class RetrieveRetriever(BaseRetriever):
        k_sub: int = 6

        def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> list[Document]:
            out = []
            for d in retrieve(query, self.k_sub):
                out.append(Document(page_content=d.page_content, metadata={**d.metadata, "_orig": d}))
            return out

    mq = MultiQueryRetriever.from_llm(retriever=RetrieveRetriever(k_sub=k), llm=GrokLLM())
    union = mq.invoke(q)
    docs = [d.metadata["_orig"] for d in union]
    # dedup por (video_id, chunk), como o resto do pipeline
    vistos, uniq = set(), []
    for d in docs:
        c = (d.metadata.get("video_id"), d.metadata.get("chunk"))
        if c not in vistos:
            vistos.add(c)
            uniq.append(d)
    if len(uniq) > k:
        scores = get_reranker().rerank(q, [d.page_content for d in uniq])
        uniq = [d for d, _ in sorted(zip(uniq, scores), key=lambda x: x[1], reverse=True)][:k]
    return uniq, {"llm_calls": 1, "n_union": len(union)}


GRADE_PROMPT = """\
Pergunta do aluno: {question}

Trechos recuperados (resumo do início de cada um):
{trechos}

Os trechos bastam para responder TODAS as partes da pergunta? Responda SOMENTE com JSON:
{{"suficiente": true|false, "lacunas": ["sub-pergunta completa sobre o que falta", ...]}}
"""


def variante_e(q, k, gen, max_voltas: int = 2):
    """Agente LangGraph: decompõe, busca, avalia suficiência e, se faltar faceta, busca de novo só a lacuna."""
    from typing import TypedDict
    from langgraph.graph import END, START, StateGraph

    class Estado(TypedDict):
        subs: list
        listas: list
        voltas: int
        ok: bool

    def no_decompor(s):
        return {"subs": decompose(q, gen=gen), "listas": [], "voltas": 0, "ok": False}

    def no_buscar(s):
        novas = [retrieve(sub, max(2, k // max(1, len(s["subs"])) + 1)) for sub in s["subs"]]
        return {"listas": s["listas"] + novas, "voltas": s["voltas"] + 1}

    def no_avaliar(s):
        docs = intercalar(s["listas"], k)
        trechos = "\n".join(f"- {d.page_content[:260]!r}" for d in docs)
        try:
            m = re.search(r"\{.*\}", gen(GRADE_PROMPT.format(question=q, trechos=trechos)), re.S)
            r = json.loads(m.group(0)) if m else {}
        except Exception:  # noqa: BLE001
            r = {}
        lac = [x for x in r.get("lacunas", []) if isinstance(x, str) and x.strip()] if isinstance(r, dict) else []
        suf = bool(r.get("suficiente", True)) if isinstance(r, dict) else True
        return {"ok": suf or not lac or s["voltas"] >= max_voltas, "subs": lac if (not suf and lac) else s["subs"]}

    g = StateGraph(Estado)
    g.add_node("decompor", no_decompor)
    g.add_node("buscar", no_buscar)
    g.add_node("avaliar", no_avaliar)
    g.add_edge(START, "decompor")
    g.add_edge("decompor", "buscar")
    g.add_edge("buscar", "avaliar")
    g.add_conditional_edges("avaliar", lambda s: END if s["ok"] else "buscar")
    final = g.compile().invoke({"subs": [], "listas": [], "voltas": 0, "ok": False})
    return intercalar(final["listas"], k), {"voltas": final["voltas"]}


VARIANTES = {"A": variante_a, "B": variante_b, "C": variante_c, "D": variante_d, "E": variante_e}
NOMES = {"A": "baseline", "B": "hyde", "C": "decompose", "D": "multiquery-langchain", "E": "agente-langgraph"}

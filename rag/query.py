"""Consulta a RAG: busca no Qdrant + gera resposta com o LLM configurado.

Uso:
    python -m rag.query "sua pergunta aqui"
    python -m rag.query "pergunta" --k 6 --channel <channel_id>
    python -m rag.query "pergunta" --no-llm     # só mostra os trechos encontrados
"""
import argparse
import os
import re
import subprocess
import sys
import tempfile
import urllib.request
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from qdrant_client.http import models as qmodels

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import (  # noqa: E402
    COLLECTION_NAMES,
    LLM_PROVIDER,
    OLLAMA_MODEL,
    OLLAMA_URL,
    ANTHROPIC_API_KEY,
    CLAUDE_MODEL,
    RAG_DEADLINE_S,
    RAG_GEN_MODEL,
    RAG_VERIFY_MODEL,
)
from rag.claude_cli import run_claude  # noqa: E402
from rag.vectorstore import get_vectorstore  # noqa: E402

PROMPT = """Você responde SOMENTE com base nos trechos fornecidos, em português.

CONTRATO DE CITAÇÃO (obrigatório):
- Trechos [TRANSCRIÇÃO] e [MATERIAL] são fonte primária: PODE citar deles.
- Trechos [RESUMO IA — NÃO CITÁVEL] são resumos gerados por IA e podem conter \
estrutura, nomes de "framework" e NÚMEROS inventados. Use-os APENAS para se orientar \
(localizar o assunto), NUNCA como fonte de uma afirmação, passo ou número.
- NÃO afirme nenhum número, porcentagem, métrica, nome de framework ou passo específico \
que não apareça literalmente num trecho [TRANSCRIÇÃO]/[MATERIAL]. Se algo só aparece em \
resumo, diga explicitamente que é "não verificado na transcrição".
- Cite cada afirmação com o título e o link com timestamp. Se a resposta não estiver \
nos trechos primários, diga que não encontrou.

Trechos:
{context}

Pergunta: {question}

Resposta:"""


def _any(key, valores):
    """FieldCondition que casa se metadata.<key> ∈ valores (OR). `valores` pode ser
    um str (1 valor) ou lista. Usa MatchAny (índice keyword)."""
    if isinstance(valores, str):
        valores = [valores]
    return qmodels.FieldCondition(key=f"metadata.{key}",
                                  match=qmodels.MatchAny(any=list(valores)))


def _build_filter(area=None, channel=None, source_type=None, module=None,
                  sem_sintetico=False):
    """Monta o filtro Qdrant. `area`, `channel`, `source_type` aceitam str OU lista
    (multi-área/multi-tipo — casa qualquer um deles). `sem_sintetico` exclui
    resumo/qa (must_not) — pool só de fontes primárias."""
    must = []
    if area:
        must.append(_any("area", area))
    if channel:
        must.append(_any("channel_id", channel))
    if source_type:
        must.append(_any("source_type", source_type))
    if module:
        # `metadata.module` é índice KEYWORD -> casamento EXATO (use listar_modulos
        # p/ descobrir o nome exato do curso/módulo).
        must.append(qmodels.FieldCondition(
            key="metadata.module", match=qmodels.MatchValue(value=module)))
    must_not = [_any("source_type", SINTETICO)] if sem_sintetico else []
    return qmodels.Filter(must=must, must_not=must_not) if (must or must_not) else None


SINTETICO = ("resumo", "qa")


def _dedup_conteudo(docs):
    """Remove trechos com texto praticamente idêntico (aula republicada em vários
    cursos). Mantém o 1º de cada conteúdo. Preserva a ordem."""
    visto, out = set(), []
    for d in docs:
        chave = re.sub(r"\s+", " ", (d.page_content or "")[:200]).strip().lower()
        if chave and chave in visto:
            continue
        visto.add(chave)
        out.append(d)
    return out


def _unicos_pool(pares):
    """Tira do pool, ANTES do rerank, os trechos de texto IDÊNTICO (aula republicada em vários cursos): têm o
    mesmo score, o `_dedup_conteudo` ficaria só com o primeiro de qualquer jeito, e cada um custa ~0,04 s de
    rerank. Chave = texto inteiro, mais estrita que os 200 caracteres do `_dedup_conteudo`, que segue
    cortando o resto depois do rerank. Mantém a ordem."""
    visto, out = set(), []
    for d, s in pares:
        chave = re.sub(r"\s+", " ", d.page_content or "").strip().lower()
        if chave and chave in visto:
            continue
        visto.add(chave)
        out.append((d, s))
    return out


# Páginas de menu do campus (título genérico "Por favor, avalie o MUP...") misturam navegação com conteúdo:
# ficam no índice, mas descem no ranking. 0 desliga.
TITULOS_REBAIXADOS = re.compile(os.getenv("TITULOS_REBAIXADOS", r"avalie o MUP"), re.I)
PENALIDADE_TITULO = float(os.getenv("PENALIDADE_TITULO", "0.5"))


def retrieve(question: str, k: int, channel=None, area=None,
             source_type=None, rerank: bool = True, fetch_k: int | None = None,
             module=None, priorizar_sintese: bool = True, hyde: bool = False,
             pools_separados: bool = False, penalidade_sintetico: float = 0.0,
             fetch_k_sintetico: int | None = None, busca_k: int | None = None, dedup_pool: bool = False,
             reranker_fn=None, collections=None, exact: bool = False, trace: dict | None = None):
    """Busca HÍBRIDA (denso e5-large + BM25) + RERANK (cross-encoder multilingual).

    Melhorias de qualidade:
    - `priorizar_sintese` (padrão): quando NÃO se filtra `source_type`, injeta um pool
      dedicado de candidatos `resumo`/`qa` no rerank. Sem isso, um acervo grande de
      transcrição afoga os resumos e a camada destilada nunca aparece.
    - dedup por CONTEÚDO: descarta trechos quase idênticos (mesma aula republicada em
      vários cursos), que antes ocupavam vários slots do top-k.
    - `hyde`: gera uma hipótese de resposta e busca por ela (melhora perguntas vagas).

    - `pools_separados`: o pool principal passa a trazer SÓ fontes primárias (must_not
      resumo/qa) e o pool sintético já existente traz os sintéticos; os dois rankings
      são juntados antes do rerank. Sem isso, o sintético que ganha a disputa direta
      no pool único (viés de tipo do BGE-M3) tira o primário do top-`cand`.
    - `penalidade_sintetico`: subtrai do score do rerank dos resumo/qa (0 = desligado).
    - `reranker_fn(pergunta, textos) -> list[float]`: troca o reranker (eval de outros
      modelos, ex. app Modal). None = default (Jina via get_reranker()).

    - `collections`: busca em várias coleções de uma vez (padrão: `COLLECTION_NAMES`). Cada
      coleção traz seu pool; os pools se juntam e o MESMO rerank ordena tudo, então o score é
      comparável entre coleções. O total de candidatos fica no tamanho de uma coleção só
      (dividido entre elas), para a latência do rerank não crescer. Os docs ganham
      `metadata['_colecao']`. Todas precisam usar o mesmo embedding.

    - `exact`: busca exata no Qdrant (`SearchParams(exact=True)`), sem o HNSW; o funil (R5) usa.
    - `trace`: dict que recebe o conjunto de cada etapa (denso, esparso, fusao, sintetico, pool,
      rerank, final) e a `config` efetiva; as buscas só-densa e só-esparsa rodam apenas com ele.

    `area`/`channel`/`source_type` aceitam str OU lista. Os docs voltam com
    `metadata['_score']` (híbrido) e `metadata['_rerank']` (se rerankeado)."""
    from rag.vectorstore import get_reranker
    cols = list(collections or COLLECTION_NAMES)
    reranker = None if reranker_fn else (get_reranker() if rerank else None)
    reranker = reranker_fn or reranker  # callable(pergunta, textos) OU objeto com .rerank
    cand = (fetch_k or max(k * 5, 30)) if reranker else k

    consulta = _hyde(question) if hyde else question

    separar = pools_separados and priorizar_sintese and not source_type and reranker
    flt = _build_filter(area, channel, source_type, module, sem_sintetico=bool(separar))
    # pool sintético dedicado: só quando o usuário NÃO fixou um source_type
    com_pool_sintetico = priorizar_sintese and not source_type and reranker
    flt_s = _build_filter(area, channel, list(SINTETICO), module) if com_pool_sintetico else None
    n = len(cols)
    cand_c = cand if n == 1 else max(-(-cand // n), k)  # piso k: cada coleção ainda pode dar um top-k inteiro
    cand_sint = fetch_k_sintetico or max(k * 2, 12)
    cand_s = cand_sint if n == 1 else max(-(-cand_sint // n), 6)
    sp = qmodels.SearchParams(exact=True) if exact else None
    if trace is not None:
        from langchain_qdrant import RetrievalMode
        for etapa in ("denso", "esparso", "fusao", "sintetico"):
            trace[etapa] = []
        trace["config"] = {"k": k, "cand": cand_c, "cand_sintetico": cand_s if com_pool_sintetico else 0,
                           "pools_separados": bool(separar), "penalidade_sintetico": penalidade_sintetico,
                           "rerank": bool(reranker), "exact": exact}

    def _pools_ou_erro(col):
        try:
            return _pools(col), None
        except Exception as e:  # noqa: BLE001 — uma coleção fora do ar não derruba as outras
            return [], f"{col}: {e}"

    def _pools(col):
        # uma coleção: pool principal (respeita o filtro pedido) + pool sintético
        vs = get_vectorstore(create=False) if n == 1 else get_vectorstore(create=False, collection=col)
        # busca_k > cand: a fusão RRF enxerga listas mais fundas e só os `cand` primeiros únicos vão ao rerank
        ps = vs.similarity_search_with_score(consulta, k=max(cand_c, busca_k or 0), filter=flt, search_params=sp)
        ps_s = (vs.similarity_search_with_score(consulta, k=cand_s, filter=flt_s, search_params=sp)
                if com_pool_sintetico else [])
        if dedup_pool:
            ps, ps_s = _unicos_pool(ps), _unicos_pool(ps_s)
        ps, ps_s = ps[:cand_c], ps_s[:cand_s]
        if trace is not None:  # o funil separa a primeira busca por modo, com o mesmo pool e filtro
            trace["fusao"] += [d for d, _ in ps]
            trace["sintetico"] += [d for d, _ in ps_s]
            for etapa, modo in (("denso", RetrievalMode.DENSE), ("esparso", RetrievalMode.SPARSE)):
                vm = get_vectorstore(create=False, collection=col, retrieval_mode=modo)
                pm = vm.similarity_search_with_score(consulta, k=max(cand_c, busca_k or 0), filter=flt, search_params=sp)
                trace[etapa] += [d for d, _ in (_unicos_pool(pm) if dedup_pool else pm)[:cand_c]]  # mesma regra do pool
        ps += ps_s
        if n > 1:
            for d, _ in ps:
                d.metadata["_colecao"] = col
        return ps

    if n == 1:
        pairs = _pools(cols[0])
    else:
        from rag.vectorstore import get_client, get_dense_embeddings, get_sparse_embeddings
        # os globais são lazy e sem trava: aquece antes das threads para o modelo carregar uma vez só
        get_dense_embeddings(), get_sparse_embeddings(), get_client()
        with ThreadPoolExecutor(max_workers=n) as ex:  # uma busca por coleção, em paralelo
            res = list(ex.map(_pools_ou_erro, cols))
        erros = [e for _, e in res if e]
        if len(erros) == n:
            raise RuntimeError("busca falhou em todas as coleções: " + "; ".join(erros))
        for e in erros:
            print(f"aviso: busca sem a coleção {e}", file=sys.stderr)
        pairs = [p for ps, _ in res for p in ps]

    # dedup por id de ponto (os dois pools podem se sobrepor)
    seen_id, uniq = set(), []
    for d, s in pairs:
        pid = id(d) if d.metadata.get("_pid") is None else d.metadata["_pid"]
        chave = (d.metadata.get("_colecao"), d.metadata.get("video_id"), d.metadata.get("chunk"))
        if chave in seen_id:
            continue
        seen_id.add(chave)
        d.metadata["_score"] = round(float(s), 4)
        uniq.append(d)
    docs = uniq

    for d in docs:  # score do rerank no metadata (None sem reranker): o eval e o limiar de R4 leem daqui
        d.metadata["relevance_score"] = None
    if reranker and docs:
        textos = [d.page_content for d in docs]
        scores = list(reranker_fn(question, textos) if reranker_fn
                      else reranker.rerank(question, textos, batch_size=16))  # batch 64 estourou os 8 GiB do CT com pool de 105
        if penalidade_sintetico:
            scores = [sc - penalidade_sintetico
                      if (d.metadata.get("source_type") or "").lower() in SINTETICO else sc
                      for d, sc in zip(docs, scores)]
        if PENALIDADE_TITULO:
            scores = [sc - PENALIDADE_TITULO if TITULOS_REBAIXADOS.search(d.metadata.get("title") or "") else sc
                      for d, sc in zip(docs, scores)]
        ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        out = []
        for d, sc in ranked:
            d.metadata["_rerank"] = round(float(sc), 4)
            d.metadata["relevance_score"] = d.metadata["_rerank"]
            out.append(d)
        ordenados = _dedup_conteudo(out)
    else:
        if n > 1:  # sem rerank o score híbrido é o único critério; sem ordenar, a 1ª coleção ocupa o top-k
            docs.sort(key=lambda d: d.metadata["_score"], reverse=True)
        ordenados = _dedup_conteudo(docs)
    if trace is not None:
        trace.update(pool=list(docs), rerank=ordenados, final=ordenados[:k])
    return ordenados[:k]


def _hyde(question: str) -> str:
    """HyDE: pede ao LLM um parágrafo hipotético que responderia à pergunta e usa
    ESSE texto como consulta (casa melhor com o vocabulário das aulas). Se não houver
    LLM configurado, devolve a pergunta original."""
    try:
        if LLM_PROVIDER == "claude" and ANTHROPIC_API_KEY:
            hip = gen_claude(f"Escreva 1 parágrafo curto, em português, respondendo como "
                             f"se fosse trecho de uma aula: {question}")
            return f"{question}\n{hip}"[:1500]
        if LLM_PROVIDER == "grok":
            hip = gen_grok(f"Escreva 1 parágrafo curto, em português, respondendo como "
                           f"se fosse trecho de uma aula ou material de estudo: {question}")
            return f"{question}\n{hip}"[:1500]
        if LLM_PROVIDER == "ollama":
            return f"{question}\n{gen_ollama(f'Responda em 1 parágrafo: {question}')}"[:1500]
    except Exception:  # noqa: BLE001 — HyDE é best-effort
        pass
    return question


def get_lesson_chunks(video_id: str, client=None, collection: str | None = None):
    """Devolve TODOS os chunks de uma aula/material (mesmo video_id), ordenados por
    posição (`chunk`). Base para ler a aula inteira ou expandir contexto."""
    from rag.vectorstore import get_client
    from config import COLLECTION_NAME
    client = client or get_client()
    flt = qmodels.Filter(must=[qmodels.FieldCondition(
        key="metadata.video_id", match=qmodels.MatchValue(value=video_id))])
    chunks, offset = [], None
    while True:
        pts, offset = client.scroll(
            collection_name=collection or COLLECTION_NAME, scroll_filter=flt,
            limit=512, offset=offset, with_payload=True, with_vectors=False)
        for p in pts:
            m = (p.payload or {}).get("metadata", {}) or {}
            chunks.append({"chunk": m.get("chunk", 0), "text": p.payload.get("page_content", ""),
                           "start": m.get("start", 0), "meta": m})
        if offset is None or not pts:
            break
    chunks.sort(key=lambda x: x["chunk"])
    return chunks


def expand_context(doc, client=None, window: int = 1) -> str:
    """Retorna o texto do chunk + vizinhos (chunk±window) da mesma aula, em ordem —
    para não cortar a frase no meio."""
    vid = doc.metadata.get("video_id")
    c = doc.metadata.get("chunk", 0)
    if not vid:
        return doc.page_content
    allc = get_lesson_chunks(vid, client=client, collection=doc.metadata.get("_colecao"))
    win = [x for x in allc if abs(x["chunk"] - c) <= window]
    if not win:
        return doc.page_content
    return " ".join(x["text"] for x in win)


def _prov_tag(source_type: str) -> str:
    """Rótulo de proveniência p/ o contexto: distingue fonte primária (citável) da
    camada sintética (resumo/qa gerados por IA — não citável)."""
    st = (source_type or "").lower()
    if st in SINTETICO:
        return "RESUMO IA — NÃO CITÁVEL"
    if st == "material":
        return "MATERIAL"
    return "TRANSCRIÇÃO"


def format_context(docs) -> str:
    blocks = []
    for d in docs:
        m = d.metadata
        tag = _prov_tag(m.get("source_type", ""))
        blocks.append(
            f"[{tag}] [{m.get('title','?')}] ({m.get('url_at', m.get('url',''))})\n{d.page_content}"
        )
    return "\n\n---\n\n".join(blocks)


def contexto_citavel(docs, sintetico: bool = False):
    """Trechos que entram no prompt de resposta. `resumo` e `qa` (camada sintética) ajudam a ACHAR o
    assunto na busca, mas são texto gerado por IA: por padrão ficam fora do contexto, para a resposta só
    poder citar material e transcrição. `sintetico=True` mantém todos. Sem nenhum primário, devolve tudo."""
    if sintetico:
        return list(docs)
    prim = [d for d in docs if (d.metadata.get("source_type") or "").lower() not in SINTETICO]
    return prim or list(docs)


def montar_prompt(question: str, docs, sintetico: bool = False):
    """(prompt, docs no contexto): ponto único para o `main` e o eval montarem o prompt de resposta."""
    ctx = contexto_citavel(docs, sintetico)
    return PROMPT.format(context=format_context(ctx), question=question), ctx


def gen_ollama(prompt: str) -> str:
    payload = json.dumps(
        {"model": OLLAMA_MODEL, "prompt": prompt, "stream": False}
    ).encode()
    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/generate", data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read())["response"].strip()


def gen_grok(prompt: str) -> str:
    """Gera a resposta pelo `grok` CLI (headless, 1 turno): o prompt vai por arquivo, sem passar pelo argv."""
    with tempfile.NamedTemporaryFile("w", suffix=".txt", encoding="utf-8") as f:
        f.write(prompt)
        f.flush()
        r = subprocess.run([os.getenv("GROK_BIN", "grok"), "--prompt-file", f.name, "--output-format", "plain", "--max-turns", "1"],
                           capture_output=True, text=True, timeout=300)
    if r.returncode != 0 or not r.stdout.strip():
        raise RuntimeError(f"grok saiu {r.returncode}: {r.stderr.strip()[:200]}")
    return r.stdout.strip()


def gen_claude(prompt: str) -> str:
    from langchain_anthropic import ChatAnthropic

    llm = ChatAnthropic(model=CLAUDE_MODEL, api_key=ANTHROPIC_API_KEY, temperature=0)
    return llm.invoke(prompt).content


def generator(provider: str):
    """Gerador do pipeline `answer` por provedor; `claude-cli` (padrão) é o Haiku por `claude -p`."""
    if provider == "claude-cli":
        return lambda p, timeout=RAG_DEADLINE_S: run_claude(p, RAG_GEN_MODEL, timeout)
    fns = {"grok": gen_grok, "ollama": gen_ollama, "claude": gen_claude}
    if provider not in fns:
        raise ValueError(f"LLM_PROVIDER desconhecido: {provider}")
    return fns[provider]


def verifier():
    """Verificador do pipeline `answer`: Sonnet por `claude -p`, para qualquer provedor de geração."""
    return lambda p, timeout=RAG_DEADLINE_S: run_claude(p, RAG_VERIFY_MODEL, timeout)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--k", type=int, default=15,
                    help="trechos recuperados (padrão 15: +0,03 de facetas no contexto sobre k=12 por ~1,7 mil chars a mais; "
                         "medir no próprio acervo)")
    ap.add_argument("--channel", default=None, help="filtra por channel_id")
    ap.add_argument("--module", default=None, help="filtra por módulo (nome exato; ver `listar_modulos`)")
    ap.add_argument("--no-llm", action="store_true", help="só mostra os trechos")
    ap.add_argument("--json", action="store_true", help="imprime o resultado completo de `answer` (inclusive frases cortadas e motivo)")
    ap.add_argument("--hyde", action="store_true",
                    help="expande a consulta com uma resposta hipotética (HyDE; desligado por padrão: com k=12 soma pouco e custa "
                         "uma chamada ao LLM, ~40 s no grok)")
    ap.add_argument("--no-hyde", action="store_true", help=argparse.SUPPRESS)  # legado: HyDE já é desligado por padrão
    ap.add_argument("--com-sintetico", action="store_true",
                    help="inclui resumo/Q&A (texto gerado por IA) no contexto da resposta; por padrão só material e transcrição")
    ap.add_argument("--decompose", action="store_true",
                    help="quebra a pergunta em sub-perguntas (uma busca por faceta); desligado por padrão: "
                         "desligado por padrão: medir no próprio acervo antes de ligar")
    args = ap.parse_args()

    # HyDE e decomposição chamam o LLM antes da busca: com --no-llm ou provider "none" ficam desligados
    usa_llm_na_busca = not (args.no_llm or LLM_PROVIDER == "none")
    usa_hyde = usa_llm_na_busca and args.hyde and not args.no_hyde
    if args.decompose and usa_llm_na_busca:
        from rag.decompose import retrieve_decomposto
        docs, subs = retrieve_decomposto(args.question, args.k, channel=args.channel)
        if len(subs) > 1:
            print("Sub-perguntas:", *[f"  - {s}" for s in subs], sep="\n", file=sys.stderr)
    else:
        docs = retrieve(args.question, args.k, args.channel, module=args.module, hyde=usa_hyde)
    if not docs:
        print("Nenhum trecho encontrado. A coleção está vazia?")
        return

    if args.no_llm or LLM_PROVIDER == "none":
        print("\n=== Trechos encontrados ===\n")
        for d in docs:
            m = d.metadata
            print(f"• {m.get('title','?')}  ({m.get('url_at','')})")
            print(f"  {d.page_content[:300]}...\n")
        return

    from rag.answer import answer, format_text
    res = answer(args.question, docs, generator(LLM_PROVIDER), verifier(), RAG_DEADLINE_S)
    print(json.dumps(res, ensure_ascii=False, indent=2) if args.json else format_text(res))


if __name__ == "__main__":
    main()

"""Eval runner — same retrieve pipeline as the MCP server, plus citation metrics.

Uso:
    python -m eval.run_eval                      # golden.jsonl, k=6, com resposta+juiz
    python -m eval.run_eval --k 8 --no-judge     # só retrieval + números (rápido)
    python -m eval.run_eval --golden eval/golden_legal.jsonl --limit 10
"""
import argparse
import hashlib
import json
import os
import statistics as st
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import ANTHROPIC_API_KEY, CLAUDE_MODEL, LLM_PROVIDER, OLLAMA_MODEL  # noqa: E402
from rag.query import (  # noqa: E402
    PROMPT, format_context, gen_claude, gen_ollama, retrieve,
)
from eval import metrics as M  # noqa: E402
from eval.judge import faithfulness  # noqa: E402
from eval.judge_cache import JudgeCache  # noqa: E402
from eval.llm_cli import gen_claude_cli  # noqa: E402

HERE = Path(__file__).resolve().parent


def _model_label() -> str:
    """Modelo que `_gen` de fato usa — mesma condição, pra `cfg["llm"]` no
    cabeçalho do relatório nunca divergir do que gerou a resposta (achado ao
    rodar a matriz R2: a versão anterior gravava OLLAMA_MODEL mesmo quando
    `_gen` caía no fallback do CLI, sem chave nem ollama)."""
    return OLLAMA_MODEL if LLM_PROVIDER == "ollama" else CLAUDE_MODEL


def _gen(prompt: str) -> str:
    if ANTHROPIC_API_KEY and LLM_PROVIDER != "ollama":
        return gen_claude(prompt)
    if LLM_PROVIDER == "ollama":
        return gen_ollama(prompt)
    return gen_claude_cli(prompt, model=CLAUDE_MODEL)


def load_golden(path: str, include_draft: bool) -> list[dict]:
    """Lê golden.jsonl no formato do gen_golden.py ({pergunta, area, expected_ids, fonte}).
    Aceita também o formato antigo ({question, video_id, curated}). Devolve linhas com
    `question`, `video_ids` (lista) e `id`. Sem `curated` = curada (gerada do Q&A);
    só `curated: false` explícito é rascunho."""
    rows = []
    for n, ln in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        ln = ln.strip()
        if not ln:
            continue
        r = json.loads(ln)
        if not (include_draft or r.get("curated", True)):
            continue
        r["question"] = r.get("question") or r["pergunta"]
        r["video_ids"] = r.get("expected_ids") or ([r["video_id"]] if r.get("video_id") else [])
        r.setdefault("id", f"q{n}")
        rows.append(r)
    return rows


# tipos que NÃO são camada sintética (ver rag.query.SINTETICO); lista fixa p/ --primary-only
PRIMARIOS = ["video", "material", "texto", "documento", "swipe", "pdf", "planilha", "referencia"]


def gold_texts(video_ids: list[str]) -> list[str]:
    """Textos dos chunks primários da(s) aula(s)-gold, direto do Qdrant (p/ content_recall)."""
    from qdrant_client.http import models as qm
    from config import COLLECTION_NAME
    from rag.vectorstore import get_client
    pts, _ = get_client().scroll(
        COLLECTION_NAME, limit=500, with_payload=["page_content"],
        scroll_filter=qm.Filter(must=[qm.FieldCondition(
            key="metadata.video_id", match=qm.MatchAny(any=video_ids))]))
    return [p.payload["page_content"] for p in pts]


def mean(xs):
    xs = [x for x in xs if x is not None]
    return round(st.mean(xs), 3) if xs else None


def _run_one(i: int, total: int, r: dict, args, cache: JudgeCache | None) -> dict:
    """Uma pergunta do golden. Nunca deixa exceção subir: `--workers` roda isto
    em paralelo via `ThreadPoolExecutor.map`, e uma exceção não tratada aqui
    derruba `list(ex.map(...))` por inteiro — perdendo os resultados de TODAS
    as outras perguntas já computadas por outras threads (achado da revisão
    do PR #6, antes de rodar as 532 gerações de R2/R3). Em erro, devolve um
    `rec` com `"erro"` e métricas em 0.0 (conservador — nunca conta como
    acerto) em vez de derrubar o batch inteiro."""
    try:
        q = r["question"]
        area = r.get("area") if args.use_area else None
        t0 = time.perf_counter()
        docs = retrieve(q, k=args.k, area=area,
                        source_type=PRIMARIOS if args.primary_only else None,
                        pools_separados=args.pools_separados,
                        penalidade_sintetico=args.penalidade_sintetico,
                        fetch_k=getattr(args, "fetch_k", None),
                        reranker_fn=getattr(args, "reranker_fn", None))
        latencia_ms = round((time.perf_counter() - t0) * 1000, 1)
        primary_ctx = "\n".join(d.page_content for d in docs if M.is_citavel(d))

        rec = {
            "id": r.get("id"), "tipo": r.get("tipo", "factual"), "area": r.get("area"),
            "latencia_ms": latencia_ms,
            # ponytail: várias expected_ids = melhor acerto entre elas, não união
            "primary_recall": max((M.primary_recall_at_k(docs, v) for v in r["video_ids"]), default=0.0),
            "context_precision": max((M.context_precision(docs, v) for v in r["video_ids"]), default=0.0),
            "lesson_recall": max((M.lesson_recall_at_k(docs, v) for v in r["video_ids"]), default=0.0),
            "content_recall": M.content_recall_at_k(docs, gold_texts(r["video_ids"])) if r["video_ids"] else 0.0,
            "synthetic_share": M.synthetic_share(docs),
        }
        if not args.no_answer:
            prompt_hash = hashlib.sha256(PROMPT.encode()).hexdigest()[:10]
            ans = _gen(PROMPT.format(context=format_context(docs), question=q))
            rec["numbers"] = M.numbers_hallucination(ans, primary_ctx)
            if r.get("tipo") == "armadilha":
                rec["abstained"] = M.abstained(ans)
            if not args.no_judge:
                rec["faithfulness"] = faithfulness(ans, primary_ctx, rec["id"], args.variant_tag,
                                                   prompt_hash, cache=cache)
            rec["answer"] = ans
            rec["primary_ctx"] = primary_ctx  # p/ eval.rejudge refazer o juiz sem refazer o retrieve
        print(f"  [{i}/{total}] {rec['id']} recall={rec['primary_recall']} "
              f"faith={rec.get('faithfulness',{}).get('score','-')} "
              f"disag={rec.get('faithfulness',{}).get('disagreement','-')} "
              f"lat_ms={latencia_ms}", file=sys.stderr)
        return rec
    except Exception as e:  # noqa: BLE001 — 1 pergunta não pode derrubar o batch inteiro
        print(f"  [{i}/{total}] {r.get('id')} ERRO: {e}", file=sys.stderr)
        return {"id": r.get("id"), "tipo": r.get("tipo", "factual"), "area": r.get("area"),
                "latencia_ms": 0.0, "primary_recall": 0.0, "context_precision": 0.0,
                "lesson_recall": 0.0, "content_recall": 0.0, "synthetic_share": 0.0,
                "erro": str(e)}


def _run_all(rows: list[dict], args, cache, passes: int = 3, pausa: float = 30.0) -> list[dict]:
    """Roda o golden inteiro e re-roda só as perguntas com `erro` (rede caiu,
    DNS, rate limit) até `passes` vezes. Sem isto, uma queda de rede no meio de
    um run de ~35 min vira dezenas de perguntas com recall 0.0 e o relatório
    parece válido (medido: base_k6 de 2026-09-25 saiu com 31/133 erros)."""
    total = len(rows)
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        per = list(ex.map(lambda ir: _run_one(ir[0], total, ir[1], args, cache),
                          enumerate(rows, 1)))
    for _ in range(passes):
        falhas = [i for i, x in enumerate(per) if "erro" in x]
        if not falhas:
            break
        print(f"  re-rodando {len(falhas)} pergunta(s) com erro em {pausa:.0f}s", file=sys.stderr)
        time.sleep(pausa)
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            novos = list(ex.map(lambda i: _run_one(i + 1, total, rows[i], args, cache), falhas))
        for i, n in zip(falhas, novos):
            per[i] = n
    return per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden", default=str(HERE / "golden.jsonl"))
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--include-draft", action="store_true",
                    help="inclui linhas com curated:false (senão só as curadas)")
    ap.add_argument("--no-answer", action="store_true", help="só retrieval (sem gerar resposta)")
    ap.add_argument("--no-judge", action="store_true", help="pula o juiz LLM de faithfulness")
    ap.add_argument("--use-area", action="store_true",
                    help="filtra retrieval pela `area` da pergunta (espelha 'estudar por área')")
    ap.add_argument("--primary-only", action="store_true",
                    help="busca só em fontes primárias (sem resumo/qa): o Q&A que gerou a "
                         "pergunta no gen_golden não pode mais 'gabaritar' a si mesmo")
    ap.add_argument("--pools-separados", action="store_true",
                    help="retrieve(pools_separados=True): pool principal só-primário + pool sintético")
    ap.add_argument("--penalidade-sintetico", type=float, default=0.0,
                    help="subtrai do score do rerank dos resumo/qa")
    ap.add_argument("--out-tag", default="", help="sufixo no nome do relatório (p/ achar depois)")
    ap.add_argument("--workers", type=int, default=6,
                    help="perguntas em paralelo (I/O-bound: subprocess+HTTP, não CPU)")
    ap.add_argument("--variant-tag", default="base",
                    help="nome curto da variante (chave do cache de julgamento)")
    ap.add_argument("--reranker", default="jina", choices=["jina", "bge-v2-m3", "qwen3-0.6b"],
                    help="jina = padrão do retrieve; os outros usam o app "
                         "Modal rag-rerank-eval (scripts/modal_rerank.py)")
    ap.add_argument("--fetch-k", type=int, default=None,
                    help="candidatos antes do rerank (padrão de retrieve: max(k*5, 30)); "
                         "R3 fixa 50 nos 3 rerankers pra compararem o MESMO top-50")
    args = ap.parse_args()
    args.reranker_fn = None
    if args.reranker != "jina":
        from eval.rerank_client import RemoteReranker
        args.reranker_fn = RemoteReranker(args.reranker).rerank

    rows = load_golden(args.golden, args.include_draft)
    if args.limit:
        rows = rows[: args.limit]
    if not rows:
        print("Golden set vazio (só curated:false? Use --include-draft p/ testar o rascunho).",
              file=sys.stderr)
        return

    model = _model_label()
    cfg = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "k": args.k,
        "n_perguntas": len(rows),
        "answer": not args.no_answer,
        "judge": (not args.no_judge) and (not args.no_answer),
        "llm": model,
        "variant": args.variant_tag,
        "reranker": args.reranker,
        "fetch_k": args.fetch_k,
        "prompt_hash": hashlib.sha256(PROMPT.encode()).hexdigest()[:10],
        "golden": args.golden,
        "primary_only": args.primary_only,
        "pools_separados": args.pools_separados,
        "penalidade_sintetico": args.penalidade_sintetico,
        "collection": os.getenv("COLLECTION_NAME"),
        "embed_model": os.getenv("EMBED_MODEL"),
    }
    print(f"config: {cfg}", file=sys.stderr)

    cache = JudgeCache(HERE / ".judge_cache.jsonl") if (not args.no_judge and not args.no_answer) else None
    per = _run_all(rows, args, cache)

    armad = [x for x in per if x["tipo"] == "armadilha"]
    latencias = sorted(x["latencia_ms"] for x in per)
    agg = {
        "primary_recall@k": mean([x["primary_recall"] for x in per]),
        "context_precision": mean([x["context_precision"] for x in per]),
        "lesson_recall@k": mean([x["lesson_recall"] for x in per]),
        "content_recall@k": mean([x["content_recall"] for x in per]),
        "synthetic_share": mean([x["synthetic_share"] for x in per]),
        "numbers_hallucination_rate": mean([x["numbers"]["rate"] for x in per if "numbers" in x]),
        "perguntas_com_numero_fora": sum(1 for x in per if x.get("numbers", {}).get("fora")),
        "faithfulness": mean([x["faithfulness"]["score"] for x in per if "faithfulness" in x]),
        "faithfulness_disagreement": mean(
            [x["faithfulness"]["disagreement"] for x in per
             if x.get("faithfulness", {}).get("disagreement") is not None]),
        "abstencao_correta_armadilhas": (
            mean([1.0 if x.get("abstained") else 0.0 for x in armad]) if armad else None),
        "judge_fallbacks": sum(1 for x in per if x.get("faithfulness", {}).get("fallback")),
        "n_erros": sum(1 for x in per if "erro" in x),
        "latencia_ms_mediana": st.median(latencias) if latencias else None,
        "latencia_ms_p95": (latencias[int(len(latencias) * 0.95)]
                            if len(latencias) > 1 else (latencias[0] if latencias else None)),
    }

    HERE.joinpath("reports").mkdir(exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M%S")
    out_json = HERE / "reports" / f"report_{ts}{args.out_tag}.json"
    out_md = HERE / "reports" / f"report_{ts}{args.out_tag}.md"
    out_json.write_text(json.dumps({"config": cfg, "agg": agg, "per": per},
                                    ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [f"# Eval rag-mcp — {cfg['timestamp']}", "",
             "## Config", "```json", json.dumps(cfg, ensure_ascii=False, indent=2), "```",
             "## Agregado", "| métrica | valor |", "|---|---|"]
    for k, v in agg.items():
        lines.append(f"| {k} | {v} |")
    lines += ["", "## Por pergunta",
              "| id | tipo | recall | lesson | content | ctx_prec | synth% | num_fora | faith | disag | lat_ms |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
    for x in per:
        lines.append(
            f"| {x['id']} | {x['tipo']} | {x['primary_recall']} | {x['lesson_recall']} "
            f"| {x['content_recall']} | {x['context_precision']} "
            f"| {x['synthetic_share']} | {x.get('numbers',{}).get('fora','-')} "
            f"| {x.get('faithfulness',{}).get('score','-')} "
            f"| {x.get('faithfulness',{}).get('disagreement','-')} "
            f"| {x['latencia_ms']} |")
    out_md.write_text("\n".join(lines), encoding="utf-8")

    print("\n=== AGREGADO ===")
    print(json.dumps(agg, ensure_ascii=False, indent=2))
    print(f"\nrelatório: {out_md}")
    if agg["n_erros"]:
        # relatório com erro NÃO é comparável (erro conta como recall 0.0): falha o run
        raise SystemExit(f"{agg['n_erros']} pergunta(s) seguem com erro — relatório inválido: {out_md}")


if __name__ == "__main__":
    main()

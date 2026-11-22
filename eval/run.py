"""Fase 0 — Régua de avaliação da recuperação.

Lê eval/golden.jsonl (uma pergunta-gabarito por linha) e mede a qualidade do
`retrieve()` em cada configuração. Sem isso, "melhorou" é achismo.

Formato de cada linha do golden.jsonl:
  {"pergunta": "...", "area": "constitucional",
   "expected_ids": ["<video_id da aula certa>"],   # opcional
   "expected_keywords": ["palavra", "outra"]}       # opcional (casa no texto)

Métricas:
  hit@k  — fração de perguntas em que a aula/keyword certa apareceu no top-k
  MRR    — média de 1/posição do primeiro acerto (0 se não achou)

Uso:
  python -m eval.run                      # config atual (rerank on)
  python -m eval.run --k 6
  python -m eval.run --compare            # rerank ON vs OFF, lado a lado
  python -m eval.run --source-type resumo # mede só contra a camada sintética
"""
import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

GOLDEN = ROOT / "eval" / "golden.jsonl"


def load_golden(path=None):
    g = Path(path) if path else GOLDEN
    if not g.exists():
        print(f"Sem gabarito em {g}. Gere com `python -m eval.gen_golden`.")
        sys.exit(1)
    items = []
    for line in g.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            items.append(json.loads(line))
    return items


def hit_rank(item, docs):
    """Retorna a posição (1-based) do primeiro doc que casa o gabarito, ou 0."""
    exp_ids = set(item.get("expected_ids") or [])
    kws = [w.lower() for w in (item.get("expected_keywords") or [])]
    for i, d in enumerate(docs, 1):
        m = d.metadata
        # casa por video_id de origem (aula original OU record sintético dela)
        vid = m.get("video_id", "")
        src_ref = m.get("source_ref", "")
        if exp_ids and (vid in exp_ids or src_ref in exp_ids
                        or any(vid.endswith(e) for e in exp_ids)):
            return i
        if kws and any(w in d.page_content.lower() for w in kws):
            return i
    return 0


def evaluate(items, k, rerank, source_type=None):
    os.environ["RAG_RERANK"] = "1" if rerank else "0"
    # importa DEPOIS de setar env (get_reranker lê no 1º uso)
    import importlib
    from rag import query as q
    importlib.reload(q)

    hits = 0
    rr = 0.0
    for it in items:
        docs = q.retrieve(it["pergunta"], k=k, area=it.get("area"),
                          source_type=source_type, rerank=rerank)
        r = hit_rank(it, docs)
        if r:
            hits += 1
            rr += 1.0 / r
    n = len(items)
    return {"hit@k": hits / n, "mrr": rr / n, "n": n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--source-type", default=None)
    ap.add_argument("--compare", action="store_true", help="rerank ON vs OFF")
    ap.add_argument("--golden", default=None, help="caminho de um golden alternativo")
    ap.add_argument("--by-area", action="store_true", help="breakdown por área")
    args = ap.parse_args()

    items = load_golden(args.golden)
    print(f"Gabarito: {len(items)} perguntas | k={args.k}"
          + (f" | source_type={args.source_type}" if args.source_type else ""))

    if args.compare:
        for label, rr in (("rerank OFF", False), ("rerank ON", True)):
            m = evaluate(items, args.k, rr, args.source_type)
            print(f"  {label:12}  hit@{args.k}={m['hit@k']:.2%}  MRR={m['mrr']:.3f}")
    else:
        m = evaluate(items, args.k, True, args.source_type)
        print(f"  GERAL  hit@{args.k}={m['hit@k']:.2%}  MRR={m['mrr']:.3f}")
        if args.by_area:
            from collections import defaultdict
            ba = defaultdict(list)
            for it in items:
                ba[it.get("area", "?")].append(it)
            for area, its in sorted(ba.items()):
                ma = evaluate(its, args.k, True, args.source_type)
                print(f"    {area:22} hit@{args.k}={ma['hit@k']:.2%}  MRR={ma['mrr']:.3f}  (n={ma['n']})")


if __name__ == "__main__":
    main()

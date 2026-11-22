"""Varredura de k e do pool de candidatos do rerank (`fetch_k`) sobre um golden.

Responde: pôr mais trechos no contexto (k maior) ou rerankear um pool maior (fetch_k) recupera as
facetas que faltam? Reporta facet_coverage, primary_recall, tamanho médio do contexto (custo no LLM)
e a diferença pareada contra a primeira configuração.

    python -m eval.run_k_sweep --golden eval/golden_legal.jsonl
HyDE reaproveita o cache de LLM de eval/variants.py (hipóteses já geradas custam 0).
"""
import argparse
import statistics as st
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval import metrics as M  # noqa: E402
from eval.bootstrap_ci import boot_diff_ci, boot_mean_ci  # noqa: E402
from eval.run_variants import carregar  # noqa: E402
from eval.variants import ContaChamadas  # noqa: E402

# (nome, k, fetch_k, hyde): fetch_k None = padrão do retrieve (max(5k, 30))
CONFIGS = [
    ("k6 (base)", 6, None, False),
    ("k6 pool100", 6, 100, False),
    ("k10", 10, None, False),
    ("k10 pool100", 10, 100, False),
    ("k15", 15, None, False),
    ("k20", 20, None, False),
    ("k6 + HyDE", 6, None, True),
    ("k10 + HyDE", 10, None, True),
    ("k15 + HyDE", 15, None, True),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden", default="eval/golden_legal.jsonl")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    from rag import query as Q
    gen = ContaChamadas(Q.gen_grok)
    Q.gen_grok = gen  # o _hyde passa pelo cache
    rows = carregar(args.golden)[: args.limit]
    res = {}
    for nome, k, fetch_k, hyde in CONFIGS:
        per = []
        for r in rows:
            t0 = time.perf_counter()
            docs = Q.retrieve(r["pergunta"], k, fetch_k=fetch_k, hyde=hyde)
            ms = (time.perf_counter() - t0) * 1000
            ids = r.get("expected_ids", [])
            per.append({
                "tipo": r.get("tipo"), "ms": ms,
                "facet": M.facet_coverage(docs, r.get("expected_keywords", [])) or 0.0,
                "prim": max((M.primary_recall_at_k(docs, i) for i in ids), default=0.0),
                "chars": sum(len(d.page_content) for d in docs),
            })
        res[nome] = per
        print(f"{nome:<14} facet={st.mean(p['facet'] for p in per):.3f} prim={st.mean(p['prim'] for p in per):.3f} "
              f"ctx={st.mean(p['chars'] for p in per):.0f} chars", flush=True)

    base = res[CONFIGS[0][0]]
    comp = [i for i, r in enumerate(rows) if r.get("tipo") == "composta"]
    print("\n| config | facet_coverage (IC95) | Δ vs base (IC95) | Δ compostas (n=%d) | primary_recall | contexto médio (chars) | busca mediana ms |" % len(comp))
    print("|---|---|---|---|---|---|---|")
    for nome, per in res.items():
        f = [p["facet"] for p in per]
        m = boot_mean_ci(f)
        d = boot_diff_ci(f, [p["facet"] for p in base]) if per is not base else (0, 0, 0)
        dc = boot_diff_ci([f[i] for i in comp], [base[i]["facet"] for i in comp]) if per is not base else (0, 0, 0)
        print(f"| {nome} | {m[0]} [{m[1]}, {m[2]}] | {d[0]:+.3f} [{d[1]:+.3f}, {d[2]:+.3f}] | {dc[0]:+.3f} [{dc[1]:+.3f}, {dc[2]:+.3f}] | "
              f"{st.mean(p['prim'] for p in per):.3f} | {st.mean(p['chars'] for p in per):.0f} | {st.median(p['ms'] for p in per):.0f} |")


if __name__ == "__main__":
    main()

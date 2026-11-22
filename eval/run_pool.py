"""Pool de candidatos do rerank (`fetch_k`) a k fixo: o contexto tem o mesmo tamanho (k=15), só muda
quantos candidatos o reranker enxerga. Responde: o trecho certo está no top-N do denso, mas fora do pool padrão?

    python -m eval.run_pool [--limit N]   # goldens: golden_legal.jsonl
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

K = 15
POOLS = [("pool 75 (padrão)", None), ("pool 150", 150), ("pool 300", 300), ("pool 600", 600)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    from rag import query as Q
    rows = carregar("eval/golden_legal.jsonl")[: args.limit]
    res = {}
    for nome, fetch_k in POOLS:
        per = []
        for r in rows:
            t0 = time.perf_counter()
            docs = Q.retrieve(r["pergunta"], K, fetch_k=fetch_k)
            ms = (time.perf_counter() - t0) * 1000
            per.append({"ms": ms, "facet": M.facet_coverage(docs, r.get("expected_keywords", [])) or 0.0,
                        "chars": sum(len(d.page_content) for d in docs), "tipo": r.get("tipo")})
        res[nome] = per
        print(f"{nome:<18} facet={st.mean(p['facet'] for p in per):.3f} ctx={st.mean(p['chars'] for p in per):.0f} "
              f"mediana={st.median(p['ms'] for p in per):.0f}ms", flush=True)
    base = res[POOLS[0][0]]
    comp = [i for i, r in enumerate(rows) if r.get("tipo") == "composta"]
    print(f"\n{len(rows)} perguntas ({len(comp)} compostas), k={K}")
    print("| pool | facet_ctx (IC95) | Δ vs pool 75 (IC95) | Δ compostas | contexto (chars) | busca mediana ms |")
    print("|---|---|---|---|---|---|")
    for nome, per in res.items():
        f = [p["facet"] for p in per]
        m = boot_mean_ci(f)
        d = boot_diff_ci(f, [p["facet"] for p in base]) if per is not base else (0, 0, 0)
        dc = boot_diff_ci([f[i] for i in comp], [base[i]["facet"] for i in comp]) if per is not base and comp else (0, 0, 0)
        print(f"| {nome} | {m[0]} [{m[1]}, {m[2]}] | {d[0]:+.3f} [{d[1]:+.3f}, {d[2]:+.3f}] | "
              f"{dc[0]:+.3f} [{dc[1]:+.3f}, {dc[2]:+.3f}] | {st.mean(p['chars'] for p in per):.0f} | {st.median(p['ms'] for p in per):.0f} |")


if __name__ == "__main__":
    main()

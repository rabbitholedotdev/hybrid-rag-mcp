"""Recuperação (sem LLM) das estratégias de expansão do contexto, em um ou mais goldens.

Mede, por estratégia: facet_ctx (todas as facetas), facet_ctx_txt (sem número solto), nº de chunks e chars do
contexto (contexto = só trechos primários), e a diferença pareada contra a primeira estratégia.

    python -m eval.run_expansao --golden eval/golden_legal.jsonl eval/golden_legal.jsonl
"""
import argparse
import re
import statistics as st
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval import metrics as M  # noqa: E402
from eval.bootstrap_ci import boot_diff_ci, boot_mean_ci  # noqa: E402
from eval.run_variants import carregar  # noqa: E402
from eval import expansao as X  # noqa: E402
from rag.query import contexto_citavel, retrieve  # noqa: E402


def so_texto(kws):
    return [k for k in kws if not re.fullmatch(r"[\d.,% ]+|.{1,3}", k.split("|")[0].strip())]


def estrategias(gen=None):
    prim = lambda docs: contexto_citavel(docs)
    return {
        "A k12 (base)": lambda q: prim(retrieve(q, 12)),
        "B k15": lambda q: prim(retrieve(q, 15)),
        "C vizinhos ±1 (6 sementes)": lambda q: X.vizinhos(retrieve(q, 12), 6, 1, 18),
        "D vizinhos ±1 (4 sementes)": lambda q: X.vizinhos(retrieve(q, 12), 4, 1, 12),
        "E por documento 2x6": lambda q: X.por_documento(q, 2, 6),
        "F por documento 3x5": lambda q: X.por_documento(q, 3, 5),
        "G por documento 3x4": lambda q: X.por_documento(q, 3, 4),
        "H por documento 1x12": lambda q: X.por_documento(q, 1, 12),
        "K sub-pergunta no documento (3 docs, m=3, teto 15)": lambda q: X.por_subperguntas_no_documento(q, 3, 3, 12, 15, gen=gen),
        "L sub-pergunta no documento (3 docs, m=4, teto 20)": lambda q: X.por_subperguntas_no_documento(q, 3, 4, 12, 20, gen=gen),
        "I k20": lambda q: prim(retrieve(q, 20)),
        "J k25": lambda q: prim(retrieve(q, 25)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden", nargs="+", default=["eval/golden_legal.jsonl", "eval/golden_legal.jsonl"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--only", default=None, help="nomes (prefixo, separados por vírgula) das estratégias a rodar; a primeira é a base")
    args = ap.parse_args()
    rows = [r for g in args.golden for r in carregar(g)][: args.limit]
    res = {}
    from rag import query as Q
    from eval.variants import ContaChamadas
    gen = ContaChamadas(Q.gen_grok)
    todas = estrategias(gen)
    if args.only:
        pref = [x.strip() for x in args.only.split(",")]
        todas = {n: f for p in pref for n, f in todas.items() if n.startswith(p)}
    if any(n.startswith(("K", "L")) for n in todas):  # aquece o cache da decomposição em paralelo (1 chamada por pergunta)
        from concurrent.futures import ThreadPoolExecutor
        from rag.decompose import decompose
        with ThreadPoolExecutor(5) as ex:
            list(ex.map(lambda r: decompose(r["pergunta"], gen=gen), rows))
    for nome, fn in todas.items():
        per = []
        for r in rows:
            t0 = time.perf_counter()
            ctx = fn(r["pergunta"])
            kws = r.get("expected_keywords", [])
            per.append({
                "facet": M.facet_coverage(ctx, kws) or 0.0, "facet_txt": M.facet_coverage(ctx, so_texto(kws)) or 0.0,
                "n": len(ctx), "chars": sum(len(d.page_content) for d in ctx),
                "prim": max((M.primary_recall_at_k(ctx, i) for i in r.get("expected_ids", [])), default=0.0),
                "ms": (time.perf_counter() - t0) * 1000,
            })
        res[nome] = per
        print(f"{nome:<28} facet={st.mean(p['facet'] for p in per):.3f} txt={st.mean(p['facet_txt'] for p in per):.3f} "
              f"chunks={st.mean(p['n'] for p in per):.1f} chars={st.mean(p['chars'] for p in per):.0f}", flush=True)
    base = next(iter(res.values()))
    print(f"\n{len(rows)} perguntas\n| estratégia | facet_ctx (IC95) | Δ vs base | Δ só texto | chunks | chars | facetas/1000 chars | busca mediana ms |")
    print("|---|---|---|---|---|---|---|---|")
    for nome, per in res.items():
        f, ft = [p["facet"] for p in per], [p["facet_txt"] for p in per]
        m = boot_mean_ci(f)
        d = boot_diff_ci(f, [p["facet"] for p in base]) if per is not base else (0, 0, 0)
        dt = boot_diff_ci(ft, [p["facet_txt"] for p in base]) if per is not base else (0, 0, 0)
        ch = st.mean(p["chars"] for p in per)
        print(f"| {nome} | {m[0]} [{m[1]}, {m[2]}] | {d[0]:+.3f} [{d[1]:+.3f}, {d[2]:+.3f}] | {dt[0]:+.3f} [{dt[1]:+.3f}, {dt[2]:+.3f}] | "
              f"{st.mean(p['n'] for p in per):.1f} | {ch:.0f} | {1000 * st.mean(f) / ch:.3f} | {st.median(p['ms'] for p in per):.0f} |")


if __name__ == "__main__":
    main()

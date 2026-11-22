"""Compara variantes de recuperação (A..E, ver eval/variants.py) sobre um golden.

Uso:
    python -m eval.run_variants --golden eval/golden_legal.jsonl --k 6 --variants A,B,C,D,E --out-tag legal
Grava um report por variante em eval/reports/ (formato `per` aceito por eval.bootstrap_ci) e
imprime a tabela com média, IC 95% e latência. `--ic-base A` calcula a diferença pareada de cada
variante contra a base.
"""
import argparse
import json
import statistics as st
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval import metrics as M  # noqa: E402
from eval.bootstrap_ci import boot_diff_ci, boot_mean_ci  # noqa: E402
from eval.variants import NOMES, VARIANTES, ContaChamadas  # noqa: E402

REPORTS = Path(__file__).resolve().parent / "reports"
METRICAS = ("facet_coverage", "primary_recall", "lesson_recall")


def carregar(path: str) -> list[dict]:
    rows = []
    for n, ln in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if ln.strip():
            r = json.loads(ln)
            r.setdefault("id", f"q{n}")
            rows.append(r)
    return rows


def rodar_variante(v: str, rows: list[dict], k: int, gen: ContaChamadas) -> list[dict]:
    fn, per = VARIANTES[v], []
    for r in rows:
        antes = gen.chamadas
        t0 = time.perf_counter()
        try:
            docs, meta = fn(r["pergunta"], k, gen)
            erro = None
        except Exception as e:  # noqa: BLE001 — uma pergunta ruim não derruba a comparação
            docs, meta, erro = [], {}, f"{type(e).__name__}: {str(e)[:160]}"
        ms = round((time.perf_counter() - t0) * 1000, 1)
        ids = r.get("expected_ids", [])
        per.append({
            "id": r["id"], "tipo": r.get("tipo", "simples"), "disciplina": r.get("disciplina"),
            "facet_coverage": M.facet_coverage(docs, r.get("expected_keywords", [])) or 0.0,
            "primary_recall": max((M.primary_recall_at_k(docs, i) for i in ids), default=0.0),
            "lesson_recall": max((M.lesson_recall_at_k(docs, i) for i in ids), default=0.0),
            "latencia_ms": ms, "llm_calls": gen.chamadas - antes,
            "titulos": [d.metadata.get("title", "")[:40] for d in docs], "erro": erro, **{k_: v_ for k_, v_ in meta.items() if k_ == "sub"},
        })
        print(f"  [{v}] {r['id']:<8} facet={per[-1]['facet_coverage']:<5} prim={per[-1]['primary_recall']} {ms:>7.0f} ms", flush=True)
    return per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden", default="eval/golden_legal.jsonl")
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--variants", default="A,B,C,D,E")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out-tag", default="variants")
    ap.add_argument("--ic-base", default="A")
    args = ap.parse_args()

    from rag.query import gen_grok
    gen = ContaChamadas(gen_grok)
    rows = carregar(args.golden)[: args.limit]
    REPORTS.mkdir(exist_ok=True)
    resultados = {}
    for v in args.variants.split(","):
        print(f"== variante {v} ({NOMES[v]}) · {len(rows)} perguntas", flush=True)
        per = rodar_variante(v, rows, args.k, gen)
        resultados[v] = per
        (REPORTS / f"report_{args.out_tag}_{v}_{NOMES[v]}.json").write_text(
            json.dumps({"cfg": {"variant": v, "nome": NOMES[v], "k": args.k, "golden": args.golden}, "per": per},
                       ensure_ascii=False, indent=1), encoding="utf-8")

    print("\n| var | nome | facet_coverage (IC95) | primary_recall | lesson_recall | lat. mediana ms | lat. p95 ms | chamadas LLM/perg |")
    print("|---|---|---|---|---|---|---|---|")
    for v, per in resultados.items():
        fc = boot_mean_ci([p["facet_coverage"] for p in per])
        lat = sorted(p["latencia_ms"] for p in per)
        p95 = lat[min(len(lat) - 1, int(len(lat) * 0.95))]
        print(f"| {v} | {NOMES[v]} | {fc[0]} [{fc[1]}, {fc[2]}] | {st.mean(p['primary_recall'] for p in per):.3f} | "
              f"{st.mean(p['lesson_recall'] for p in per):.3f} | {st.median(lat):.0f} | {p95:.0f} | "
              f"{st.mean(p['llm_calls'] for p in per):.1f} |")
    base = resultados.get(args.ic_base)
    if base:
        print(f"\nDiferença pareada contra {args.ic_base} (IC 95%; se não cruza 0, é real):")
        for v, per in resultados.items():
            if v == args.ic_base:
                continue
            for m in METRICAS:
                d = boot_diff_ci([p[m] for p in per], [p[m] for p in base])
                print(f"  {v}-{args.ic_base} {m}: {d[0]:+.3f} [{d[1]:+.3f}, {d[2]:+.3f}]")
        comp = [t for t in ("compostas", ) if any(p["tipo"] == "composta" for p in base)]
        if comp:
            print("\nSó perguntas compostas:")
            idx = [i for i, p in enumerate(base) if p["tipo"] == "composta"]
            for v, per in resultados.items():
                if v == args.ic_base:
                    continue
                d = boot_diff_ci([per[i]["facet_coverage"] for i in idx], [base[i]["facet_coverage"] for i in idx])
                print(f"  {v}-{args.ic_base} facet_coverage (compostas, n={len(idx)}): {d[0]:+.3f} [{d[1]:+.3f}, {d[2]:+.3f}]")


if __name__ == "__main__":
    main()

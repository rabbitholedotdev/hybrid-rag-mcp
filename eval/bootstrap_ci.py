"""IC via bootstrap simples sobre relatórios do run_eval.py — pra saber se uma
diferença entre dois runs (e5-large vs BGE-M3, k=6 vs k=10, etc.) é real ou ruído.

Uso:
    python -m eval.bootstrap_ci report_a.json report_b.json --metric primary_recall
    python -m eval.bootstrap_ci report_a.json report_b.json --metric primary_recall lesson_recall content_recall
"""
import argparse
import json
import random
import statistics as st

N_BOOT = 5000


def load_per(path: str) -> list[dict]:
    return json.loads(open(path, encoding="utf-8").read())["per"]


def metric_value(x: dict, m: str) -> float:
    """Métrica plana (primary_recall...) ou aninhada por pergunta: faithfulness
    vem como {"score": ..}, numbers_hallucination_rate como numbers.rate."""
    if m == "faithfulness":
        return x["faithfulness"]["score"]
    if m == "numbers_hallucination_rate":
        return x["numbers"]["rate"]
    return x[m]


def boot_mean_ci(xs: list[float], n_boot: int = N_BOOT, alpha: float = 0.05) -> tuple[float, float, float]:
    """(média, lo, hi) do IC 95% por resampling com reposição."""
    n = len(xs)
    means = []
    for _ in range(n_boot):
        sample = [xs[random.randrange(n)] for _ in range(n)]
        means.append(st.mean(sample))
    means.sort()
    lo = means[int(n_boot * alpha / 2)]
    hi = means[int(n_boot * (1 - alpha / 2))]
    return round(st.mean(xs), 3), round(lo, 3), round(hi, 3)


def boot_diff_ci(a: list[float], b: list[float], n_boot: int = N_BOOT, alpha: float = 0.05) -> tuple[float, float, float]:
    """IC da diferença pareada (a - b) por pergunta (mesmo id, mesma ordem).
    Se o IC não cruza 0, a diferença é real (não ruído de amostragem)."""
    n = len(a)
    diffs_obs = [x - y for x, y in zip(a, b)]
    boots = []
    for _ in range(n_boot):
        idx = [random.randrange(n) for _ in range(n)]
        boots.append(st.mean([diffs_obs[i] for i in idx]))
    boots.sort()
    lo = boots[int(n_boot * alpha / 2)]
    hi = boots[int(n_boot * (1 - alpha / 2))]
    return round(st.mean(diffs_obs), 3), round(lo, 3), round(hi, 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("report_a")
    ap.add_argument("report_b")
    ap.add_argument("--metric", nargs="+", default=["primary_recall", "lesson_recall", "content_recall"])
    ap.add_argument("--label-a", default="A")
    ap.add_argument("--label-b", default="B")
    args = ap.parse_args()

    per_a = load_per(args.report_a)
    per_b = load_per(args.report_b)
    ids_a = [x["id"] for x in per_a]
    ids_b = [x["id"] for x in per_b]
    if ids_a != ids_b:
        raise SystemExit(f"reports com perguntas/ordem diferentes — não são pareáveis "
                          f"({len(ids_a)} vs {len(ids_b)} ids, ou ordem diferente)")

    print(f"n_perguntas={len(ids_a)}\n")
    for m in args.metric:
        xs_a = [metric_value(x, m) for x in per_a]
        xs_b = [metric_value(x, m) for x in per_b]
        mean_a, lo_a, hi_a = boot_mean_ci(xs_a)
        mean_b, lo_b, hi_b = boot_mean_ci(xs_b)
        d, dlo, dhi = boot_diff_ci(xs_a, xs_b)
        sig = "REAL (IC não cruza 0)" if (dlo > 0 or dhi < 0) else "ruído (IC cruza 0)"
        print(f"{m}:")
        print(f"  {args.label_a}: {mean_a}  IC95 [{lo_a}, {hi_a}]")
        print(f"  {args.label_b}: {mean_b}  IC95 [{lo_b}, {hi_b}]")
        print(f"  diff ({args.label_a}-{args.label_b}): {d}  IC95 [{dlo}, {dhi}]  -> {sig}\n")


if __name__ == "__main__":
    main()

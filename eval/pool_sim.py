"""Lê `eval/reports/pool_<colecao>.jsonl` (de `eval/pool_oracle.py`) e conta, por estratégia de pool, em quantas
perguntas há evidência (cobertura de um chunk primário >= L) entre os N candidatos que iriam ao rerank.

  python -m eval.pool_sim <colecao>"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.funil import L_PADRAO  # noqa: E402

REPORTS = Path(__file__).resolve().parent / "reports"


def _chave(e):
    return (e[0], e[1])


def carregar(col: str) -> tuple[list[dict], dict]:
    with (REPORTS / f"pool_{col}.jsonl").open(encoding="utf-8") as f:
        recs = [json.loads(ln) for ln in f if ln.strip()]
    cob = {}
    for r in recs:
        m = {}
        for nome in ("D", "S", "F30", "F75", "F150"):
            for e in r[nome]:
                m[_chave(e)] = max(m.get(_chave(e), 0), e[3] if e[2] else 0)
        cob[r["id"]] = m
    return recs, cob


def rrf(r: dict, P: int, N: int, wd: float = 1.0, ws: float = 1.0, c: float = 2.0) -> list:
    """Os N primeiros da fusão RRF local das listas densa e esparsa, cada uma até a profundidade P."""
    sc: dict = {}
    for w, nome in ((wd, "D"), (ws, "S")):
        for i, e in enumerate(r[nome][:P]):
            sc[_chave(e)] = sc.get(_chave(e), 0) + w / (c + i + 1)
    return [k for k, _ in sorted(sc.items(), key=lambda x: -x[1])[:N]]


def main(col: str, L: float = L_PADRAO) -> None:
    recs, cob = carregar(col)

    def com_evidencia(r, chaves) -> bool:
        return any(cob[r["id"]].get(k, 0) >= L for k in chaves)

    def lista(r, nome, n=None):
        return [_chave(e) for e in r[nome][:n]]

    print(col, "n", len(recs), "L", L)
    for nome in ("F30", "F75", "F150"):
        print(nome, sum(com_evidencia(r, lista(r, nome)) for r in recs))
    print("F150[:30]", sum(com_evidencia(r, lista(r, "F150", 30)) for r in recs),
          "F150[:45]", sum(com_evidencia(r, lista(r, "F150", 45)) for r in recs),
          "F150[:60]", sum(com_evidencia(r, lista(r, "F150", 60)) for r in recs))
    print("D[:30]", sum(com_evidencia(r, lista(r, "D", 30)) for r in recs),
          "S[:30]", sum(com_evidencia(r, lista(r, "S", 30)) for r in recs),
          "D[:75]", sum(com_evidencia(r, lista(r, "D", 75)) for r in recs),
          "S[:75]", sum(com_evidencia(r, lista(r, "S", 75)) for r in recs))
    print("--- grade: P (profundidade das listas), N (pool), peso denso, peso esparso, c=2")
    for P, N in ((30, 30), (50, 30), (75, 30), (150, 30), (75, 45), (150, 45)):
        for wd, ws in ((1, 1), (2, 1), (1, 2), (3, 1), (1, 3), (1, 0), (0, 1)):
            print(f"P={P} N={N} wd={wd} ws={ws}:", sum(com_evidencia(r, rrf(r, P, N, wd, ws)) for r in recs))


if __name__ == "__main__":
    main(sys.argv[1])

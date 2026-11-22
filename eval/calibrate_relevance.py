"""Calibra o limiar de recusa por relevância (R4) só com a partição de calibração do portão.

  python -m eval.calibrate_relevance <tag-calibracao> [--margem 0.05]

limiar = menor top-1 `relevance_score` entre as respondíveis que o pipeline respondeu, menos a margem.
Conta as armadilhas que o limiar pegaria E que o contrato (prova por frase) deixou passar (status ok):
só essas justificam ligar o limiar."""
import argparse
import json
import sys
from pathlib import Path

REPORTS = Path(__file__).resolve().parent / "reports"


def calibrar(items: list[dict], margem: float) -> dict:
    top = lambda it: (it.get("relevance_score") or [None])[0]  # noqa: E731
    resp = [top(i) for i in items if i["respondivel"] and i.get("status") == "ok" and top(i) is not None]
    if not resp:
        return {"limiar": None, "armadilhas_a_mais": 0, "n_respondiveis": 0}
    limiar = round(min(resp) - margem, 4)
    vazadas = [i for i in items if not i["respondivel"] and i.get("status") == "ok"]
    pegas = [i["id"] for i in vazadas if top(i) is not None and top(i) < limiar]
    return {"limiar": limiar, "n_respondiveis": len(resp), "armadilhas_vazadas": len(vazadas),
            "armadilhas_a_mais": len(pegas), "ids": pegas}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("tag")
    ap.add_argument("--margem", type=float, default=0.05)
    a = ap.parse_args(argv)
    gate = json.loads((REPORTS / f"gate_{a.tag}.json").read_text(encoding="utf-8"))
    if gate["split"] != "calibracao":
        print("só a partição de calibração entra aqui", file=sys.stderr)
        return 2
    print(json.dumps(calibrar(gate["items"], a.margem), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())

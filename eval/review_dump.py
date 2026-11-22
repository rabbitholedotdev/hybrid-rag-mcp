"""Ajuda da revisão humana: para cada linha pendente de review_<tag>.jsonl, mostra a frase e a janela do
trecho com maior sobreposição de palavras (a revisão decide `confirmado`/`armadilha_ok` à mão)."""
import json
import re
import sys
from pathlib import Path

REPORTS = Path(__file__).resolve().parent / "reports"


def janela(frase: str, trechos: list[str], n: int = 380) -> str:
    pal = {w for w in re.findall(r"\w{4,}", frase.lower())}
    best, bs = "", -1
    for t in trechos:
        for i in range(0, max(1, len(t) - n // 2), n // 2):
            w = t[i:i + n]
            sc = len(pal & set(re.findall(r"\w{4,}", w.lower())))
            if sc > bs:
                best, bs = w, sc
    return re.sub(r"\s+", " ", best)


if __name__ == "__main__":
    tag, ini, fim = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    gate = json.loads((REPORTS / f"gate_{tag}.json").read_text())
    por = {i["id"]: i for i in gate["items"]}
    rev = [json.loads(ln) for ln in (REPORTS / f"review_{tag}.jsonl").read_text().splitlines() if ln.strip()]
    for n, r in enumerate(rev):
        if not ini <= n < fim or r["kind"] == "armadilha":
            continue
        print(f"#{n} [{r['kind']}/{r['veredito']}] {r['id']} :: {r['frase']}\n   -> {janela(r['frase'], por[r['id']]['trechos'])}\n")

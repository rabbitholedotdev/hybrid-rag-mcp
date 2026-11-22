"""Mede o cold start do `claude -p` (R6, Tarefa 2): quente (em sequência) e frio (1ª chamada depois de
`--idle-s` parado, intercalando Haiku e Sonnet) com prompt do tamanho real (k=15, ~15 mil chars), e a busca
`retrieve(k=15)` por coleção. Grava eval/reports/latency_<fase>.json.

  python -m eval.measure_latency warm --n 10
  python -m eval.measure_latency cold --n 5 --idle-s 600
  COLLECTION_NAME=legal_rag python -m eval.measure_latency search --n 10"""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
REPORTS = Path(__file__).resolve().parent / "reports"
MODELOS = {"haiku": "claude-haiku-5-5", "sonnet": "claude-sonnet-5-5"}
PERGUNTAS = {
    "legal_rag": [
        "Quais os cinco princípios da administração pública no art. 37?",
        "Qual a antecedência da comunicação de greve na regra geral?",
        "Qual o prazo decadencial do CDC para vício em produto durável?",
        "A casa é asilo inviolável em quais hipóteses?",
        "O que o art. 421 do Código Civil diz sobre a liberdade de contratar?",
        "O racismo é crime imprescritível?",
        "Qual o prazo de validade do concurso público no art. 37?",
        "O que o CDC considera consumidor?",
        "Transporte coletivo é serviço essencial na Lei de Greve?",
        "A boa-fé do art. 422 vale na conclusão e na execução do contrato?",
    ],
}


def _stats(vals: list[float]) -> dict:
    v = sorted(vals)
    pos = (len(v) - 1) * 0.95
    lo = int(pos)
    p95 = v[lo] + (v[min(lo + 1, len(v) - 1)] - v[lo]) * (pos - lo)
    return {"n": len(v), "p50": round(statistics.median(v), 2), "p95": round(p95, 2), "min": round(v[0], 2), "max": round(v[-1], 2)}


def _prompt_real() -> str:
    from rag.query import montar_prompt, retrieve
    docs = retrieve(PERGUNTAS["legal_rag"][0], 15)
    return montar_prompt(PERGUNTAS["legal_rag"][0], docs)[0]


def _chamar(modelo: str, prompt: str) -> float:
    from rag.claude_cli import run_claude
    t = time.time()
    run_claude(prompt, MODELOS[modelo], 180)
    return round(time.time() - t, 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("fase", choices=("warm", "cold", "search"))
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--idle-s", type=int, default=600)
    a = ap.parse_args()
    REPORTS.mkdir(exist_ok=True)
    out: dict = {"fase": a.fase, "n_pedido": a.n}
    if a.fase == "search":
        import os
        col = os.environ["COLLECTION_NAME"]
        from rag.query import retrieve
        retrieve("aquecimento", 5)  # carrega modelos (embedding, reranker) fora da medida
        ts = []
        for q in PERGUNTAS.get(col, PERGUNTAS["legal_rag"])[:a.n]:
            t = time.time()
            retrieve(q, 15)
            ts.append(round(time.time() - t, 2))
        out.update(colecao=col, tempos=ts, **_stats(ts))
        nome = f"latency_search_{col}.json"
    else:
        prompt = _prompt_real()
        out["prompt_chars"] = len(prompt)
        if a.fase == "warm":
            for m in ("haiku", "sonnet"):
                _chamar(m, prompt)  # descarta a 1ª (aquece o cache de prompt)
                ts = [_chamar(m, prompt) for _ in range(a.n)]
                out[m] = {"tempos": ts, **_stats(ts)}
        else:
            ts = {"haiku": [], "sonnet": []}
            for i in range(a.n * 2):
                m = ("haiku", "sonnet")[i % 2]
                print(f"cold {i}: parado {a.idle_s}s antes de {m}", flush=True)
                time.sleep(a.idle_s)
                ts[m].append(_chamar(m, prompt))
                print(f"cold {i} {m} {ts[m][-1]}s", flush=True)
            for m, v in ts.items():
                out[m] = {"tempos": v, **_stats(v)}
        nome = f"latency_{a.fase}.json"
    (REPORTS / nome).write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "tempos"}, ensure_ascii=False))


if __name__ == "__main__":
    main()

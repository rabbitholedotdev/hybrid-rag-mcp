"""Latência A/B intercalada do `retrieve` (R5): duas configurações na MESMA passada, ordem alternada
por pergunta, HNSW (o da busca servida), sem trace e sem geração. Rodadas separadas misturam a variação de
carga do Mac com o efeito da mudança (rag-mcp#51); aqui as duas variantes veem a mesma carga.

  python -m eval.run_latency_ab --colecao legal_rag --tag T --a '{"cli": {...}, "mcp": {...}}' --b '{...}' [--split calibracao]

Cada variante é {caminho: kwargs do retrieve}, como `run_funnel.CAMINHOS`. Grava
eval/reports/funil_latencia_ab_<tag>.json com p50 e p95 por variante e caminho e a razão b/a, o que
`run_funnel compare --latencia-ab` lê no critério de latência. Sem ingestão nem outra consulta concorrente."""
import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.funil_dados import load_rows  # noqa: E402
from eval.run_funnel import _commit  # noqa: E402

REPORTS = Path(__file__).resolve().parent / "reports"


def _pct(vals: list[float], p: float) -> float:
    from eval.run_gate import _pct as pct  # a mesma definição do `_latencia` do funil
    return round(pct(vals, p), 3)


def medir(rows: list[dict], a: dict, b: dict, retrieve=None) -> dict:
    """{caminho: {a: {p50,p95,n}, b: {...}, razao: {p50,p95}}}; ordem a/b alternada por pergunta."""
    if retrieve is None:
        from rag.query import retrieve
    out = {}
    for caminho in a:
        cfgs = {"a": a[caminho], "b": b[caminho]}
        for cfg in cfgs.values():  # aquecimento descartado: modelos e conexão
            retrieve(rows[0]["pergunta"], **cfg)
        tempos = {"a": [], "b": []}
        for i, r in enumerate(rows):
            for v in (("a", "b") if i % 2 == 0 else ("b", "a")):
                t0 = time.perf_counter()
                retrieve(r["pergunta"], **cfgs[v])
                tempos[v].append(time.perf_counter() - t0)
        res = {v: {"p50": _pct(t, 0.5), "p95": _pct(t, 0.95), "n": len(t)} for v, t in tempos.items()}
        res["razao"] = {q: round(res["b"][q] / res["a"][q], 3) if res["a"][q] else None for q in ("p50", "p95")}
        res["tempos"] = {v: [round(x, 3) for x in t] for v, t in tempos.items()}
        out[caminho] = res
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--colecao", required=True)
    ap.add_argument("--split", default="calibracao", choices=("calibracao", "teste"))
    ap.add_argument("--tag", required=True)
    ap.add_argument("--a", required=True, help="variante base: JSON {caminho: kwargs do retrieve}")
    ap.add_argument("--b", required=True, help="variante nova")
    ap.add_argument("--limit", type=int, default=0, help="só as N primeiras perguntas (teste rápido)")
    args = ap.parse_args(argv)
    if os.environ.get("COLLECTION_NAME") != args.colecao:
        sys.exit(f"COLLECTION_NAME={os.environ.get('COLLECTION_NAME')!r} != {args.colecao!r}: uma rodada por coleção")
    rows = load_rows(args.split, args.colecao)
    if args.limit:
        rows = rows[:args.limit]
    a, b = json.loads(args.a), json.loads(args.b)
    if set(a) != set(b):
        sys.exit(f"--a e --b pedem os mesmos caminhos: {sorted(a)} contra {sorted(b)}")
    from config import RAG_RERANK, RERANK_MODEL
    res = {"colecao": args.colecao, "split": args.split, "host": platform.node(), "n": len(rows),
           "commit": _commit(), "env": {"RAG_RERANK": RAG_RERANK, "RERANK_MODEL": RERANK_MODEL},
           "a": a, "b": b, "resultado": medir(rows, a, b)}
    REPORTS.mkdir(parents=True, exist_ok=True)
    out = REPORTS / f"funil_latencia_ab_{args.tag}.json"
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(res["resultado"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())

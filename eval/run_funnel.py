"""Funil da recuperação (R5): em que degrau o trecho certo morre, por pergunta
e caminho (CLI k=15 e MCP k=6), sobre o `retrieve` real com `exact=True`.

  python -m eval.run_funnel run --colecao legal_rag --split calibracao --tag T
  python -m eval.run_funnel calibrar --tag T
  python -m eval.run_funnel compare --visado legal_rag --novo legal_rag=T1 --base legal_rag=B1

`run` grava eval/reports/funil_<tag>.jsonl e funil_<tag>_resumo.json.
Uma rodada por coleção (COLLECTION_NAME no ambiente)."""
import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import RAG_PENALIDADE_SINTETICO, RAG_POOLS_SEPARADOS  # noqa: E402
from eval.funil import L_PADRAO, classificar, diagnostico_aula, marcas, melhor_cobertura, resumo  # noqa: E402
from eval.funil_dados import load_rows  # noqa: E402
from eval.metrics import is_citavel  # noqa: E402

REPORTS = Path(__file__).resolve().parent / "reports"
MIN_FACETAS = 3
ETAPAS = ("denso", "esparso", "fusao", "pool", "final")

# Os dois pontos de chamada reais: `rag.query.main` (--k 15, sem pools) e `buscar_transcricoes` (k=10 desde o R5, pools do env).
CAMINHOS: dict[str, dict] = {
    "cli": {"k": 15},
    "mcp": {"k": 10, "pools_separados": RAG_POOLS_SEPARADOS, "penalidade_sintetico": RAG_PENALIDADE_SINTETICO},
}


def _docs_aula(chunks: list[dict]) -> list:
    return [SimpleNamespace(page_content=c["text"], metadata={**c["meta"], "chunk": c["chunk"]}) for c in chunks]


def _ids(docs) -> list:
    return [[d.metadata.get("video_id"), d.metadata.get("chunk")] for d in docs]


def _aula(row: dict, L: float) -> tuple[dict, int | None]:
    """Diagnóstico da aula-ouro de maior `uniao`; `n_chunks_ouro` None sem `expected_ids`."""
    from rag import query
    kw = row["expected_keywords"]
    if not row["expected_ids"]:
        return diagnostico_aula([], kw, L), None
    diags = [diagnostico_aula(_docs_aula(query.get_lesson_chunks(v)), kw, L) for v in row["expected_ids"]]
    melhor = max(diags, key=lambda d: (d["uniao"], d["n_chunks"]))
    return melhor, melhor["n_chunks"]


def _classifica(it: dict, L: float, cli_final_vivo: bool | None) -> None:
    """Preenche `vivo`, `degrau` e `marcas` a partir das coberturas gravadas (mesma regra em `reclassificar`)."""
    cob, aula = it["cob"], it["aula"]
    vivo = {"aula": aula["uniao"] >= L, "chunk": aula["melhor_chunk"] >= L,
            "pool": cob["pool"] >= L, "final": cob["final"] >= L}
    trace_vivo = {e: cob[e] >= L for e in ("denso", "esparso", "fusao")}
    cobre = None if it.get("cob_resposta") is None else it["cob_resposta"] >= L
    it["vivo"] = vivo
    it["degrau"] = classificar(vivo, it["tem_ouro"], it["caminho"], cli_final_vivo, cobre)
    it["marcas"] = marcas(it["degrau"], vivo, it["tem_ouro"], trace_vivo, it["n_sintetico_fusao"], it["n_chunks_ouro"])


def medir_pergunta(row: dict, L: float, geracao: bool) -> list[dict]:
    from rag import query
    from eval.metrics import answer_facet_coverage
    base = {"id": row["id"], "colecao": row["colecao"], "golden": row["golden"]}
    kw = row["expected_keywords"]
    if len(kw) < MIN_FACETAS:
        return [{**base, "caminho": c, "degrau": None, "marcas": ["sem_facetas"]} for c in CAMINHOS]
    aula, n_chunks = _aula(row, L)
    tem_ouro = bool(row["expected_ids"]) and bool(n_chunks)
    itens, cli_final_vivo = [], None
    for caminho, cfg in CAMINHOS.items():
        trace: dict = {}
        docs = query.retrieve(row["pergunta"], exact=True, trace=trace, **cfg)
        cob = {e: melhor_cobertura(trace[e], kw) for e in ETAPAS}
        it = {**base, "caminho": caminho, "kwargs": cfg, "tem_ouro": tem_ouro, "n_chunks_ouro": n_chunks, "aula": aula,
              "cob": cob, "cob_pool": cob["pool"], "cob_final": cob["final"],
              "n_sintetico_fusao": sum(1 for d in trace["fusao"] if not is_citavel(d)),
              "ids_final": _ids(docs), "config": trace["config"], "cob_resposta": None}
        if geracao:
            from rag.answer import answer, format_text
            from rag.config import LLM_PROVIDER, RAG_DEADLINE_S
            res = answer(row["pergunta"], docs, query.generator(LLM_PROVIDER), query.verifier(), RAG_DEADLINE_S)
            it["cob_resposta"] = (answer_facet_coverage(format_text(res), kw) or 0.0) if res["status"] == "ok" else 0.0
        _classifica(it, L, cli_final_vivo)
        if caminho == "cli":
            cli_final_vivo = it["vivo"]["final"]
        itens.append(it)
    return itens


def reclassificar(itens: list[dict], L: float) -> list[dict]:
    """Os mesmos itens com outro L, sem buscar de novo (calibração do L)."""
    out, cli_vivo = [], {}
    for it in itens:
        if it["degrau"] is None:
            out.append(it)
            continue
        novo = dict(it)
        _classifica(novo, L, cli_vivo.get(it["id"]) if it["caminho"] == "mcp" else None)
        if it["caminho"] == "cli":
            cli_vivo[it["id"]] = novo["vivo"]["final"]
        out.append(novo)
    return out


def _pct(vals, p):
    from eval.run_gate import _pct as pct
    return pct(vals, p)


def _latencia(rows: list[dict], itens: list[dict]) -> tuple[dict, dict]:
    """Segundo passe com HNSW e sem trace: só o `retrieve`, em sequência, com 1 rodada de aquecimento."""
    from rag import query
    if rows:
        for cfg in CAMINHOS.values():
            query.retrieve(rows[0]["pergunta"], **cfg)
    exato = {(i["id"], i["caminho"]): i.get("ids_final") for i in itens}
    lat, difere = {}, {}
    for caminho, cfg in CAMINHOS.items():
        tempos, n_dif = [], 0
        for r in rows:
            t0 = time.perf_counter()
            docs = query.retrieve(r["pergunta"], **cfg)
            tempos.append(time.perf_counter() - t0)
            ids = exato.get((r["id"], caminho))
            n_dif += int(ids is not None and {tuple(x) for x in ids} != {tuple(x) for x in _ids(docs)})  # conjunto, não ordem
        lat[caminho] = {"p50": _pct(tempos, 0.5), "p95": _pct(tempos, 0.95), "n": len(tempos)}
        difere[caminho] = n_dif
    return lat, difere


def _commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                              cwd=Path(__file__).resolve().parent).stdout.strip() or "?"
    except OSError:
        return "?"


def _ler(p: Path) -> list[dict]:
    return [json.loads(ln) for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()] if p.exists() else []


def run(colecao: str, split: str, tag: str, L: float = L_PADRAO, geracao: bool = False, latencia: bool = False,
        reports: Path = REPORTS) -> Path:
    if os.environ.get("COLLECTION_NAME") != colecao:
        sys.exit(f"COLLECTION_NAME={os.environ.get('COLLECTION_NAME')!r} != {colecao!r}: uma rodada por coleção")
    from config import RAG_RERANK, RERANK_MODEL
    reports.mkdir(parents=True, exist_ok=True)
    jl = reports / f"funil_{tag}.jsonl"
    rows = load_rows(split, colecao)
    gravados = _ler(jl)
    for i in gravados:  # retomar com outros caminhos misturaria perguntas medidas com configurações diferentes
        if i.get("kwargs") is not None and i["kwargs"] != CAMINHOS.get(i["caminho"]):
            sys.exit(f"{jl.name}: gravado com {i['kwargs']} no caminho {i['caminho']}, agora {CAMINHOS.get(i['caminho'])}; use outra --tag")
    feitos = {i["id"] for i in gravados}
    for r in rows:
        if r["id"] in feitos:
            continue
        itens = medir_pergunta(r, L, geracao)
        with jl.open("a", encoding="utf-8") as f:  # os dois caminhos num só append: o mcp depende do cli
            f.write("".join(json.dumps(i, ensure_ascii=False) + "\n" for i in itens))
        print(r["id"], " ".join(f'{i["caminho"]}={i["degrau"]}' for i in itens), flush=True)
    ids = {r["id"] for r in rows}
    itens = reclassificar([i for i in _ler(jl) if i["id"] in ids], L)  # a retomada pode vir com outro L
    cab = {"tag": tag, "commit": _commit(), "colecao": colecao, "split": split, "L": L, "host": platform.node(),
           "geracao": geracao,
           "env": {"RAG_RERANK": RAG_RERANK, "RERANK_MODEL": RERANK_MODEL, "RAG_POOLS_SEPARADOS": RAG_POOLS_SEPARADOS,
                   "RAG_PENALIDADE_SINTETICO": RAG_PENALIDADE_SINTETICO},
           "caminhos": CAMINHOS,
           "config": {c: next((i["config"] for i in itens if i["caminho"] == c and i.get("config")), None)
                      for c in CAMINHOS},
           "resumo": resumo(itens)}
    if latencia:
        cab["latencia"], cab["hnsw_difere"] = _latencia(rows, itens)
    out = reports / f"funil_{tag}_resumo.json"
    out.write_text(json.dumps(cab, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: cab[k] for k in ("resumo", "latencia", "hnsw_difere") if k in cab}, ensure_ascii=False))
    return out


def calibrar(tag: str, Ls: list[float], reports: Path = REPORTS) -> dict:
    """Fração de perguntas com ouro que morrem no degrau 0, por L (a aula-ouro conta como evidência?)."""
    itens = _ler(reports / f"funil_{tag}.jsonl")
    out = {}
    for L in Ls:
        com_ouro = [i for i in reclassificar(itens, L) if i["degrau"] is not None and i["tem_ouro"]]
        out[L] = {"n": len(com_ouro), "degrau0": sum(1 for i in com_ouro if i["degrau"] == "0"),
                  "frac": round(sum(1 for i in com_ouro if i["degrau"] == "0") / len(com_ouro), 4) if com_ouro else None}
    return out


def compare(visado: str, tags_novo: dict, tags_base: dict, gate_novo: str | None = None,
            gate_base: str | None = None, reports: Path = REPORTS, lat_ab: dict | None = None) -> dict:
    from eval.funil import aceite

    def carrega(tags):
        out = {}
        for col, tag in tags.items():
            res = json.loads((reports / f"funil_{tag}_resumo.json").read_text(encoding="utf-8"))
            out[col] = (reclassificar(_ler(reports / f"funil_{tag}.jsonl"), res["L"]), res)
        return out

    gate = lambda t: json.loads((reports / f"gate_{t}.json").read_text(encoding="utf-8")) if t else None  # noqa: E731
    ab = {col: json.loads((reports / f"funil_latencia_ab_{t}.json").read_text(encoding="utf-8"))
          for col, t in (lat_ab or {}).items()}
    for col, d in ab.items():  # o A/B tem de ser dos dois caminhos e da configuração comparada
        novo = json.loads((reports / f"funil_{tags_novo[col]}_resumo.json").read_text(encoding="utf-8"))
        if set(d["resultado"]) != {"cli", "mcp"} or d["b"] != {c: novo["caminhos"][c] for c in d["b"]}:
            raise SystemExit(f"latência A/B de {col} não cobre os dois caminhos ou não é da configuração nova ({tags_novo[col]})")
    return aceite(carrega(tags_base), carrega(tags_novo), visado, gate(gate_base), gate(gate_novo), ab or None)


def _col_tags(items: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for it in items:
        if "=" not in it:
            raise SystemExit(f"esperado colecao=tag, veio {it!r}")
        col, tag = it.split("=", 1)
        out[col] = tag
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--colecao", required=True)
    r.add_argument("--split", required=True, choices=("calibracao", "teste"))
    r.add_argument("--tag", required=True)
    r.add_argument("--L", type=float, default=L_PADRAO)
    r.add_argument("--geracao", action="store_true")
    r.add_argument("--latencia", action="store_true")
    r.add_argument("--caminhos", help='JSON {"cli": {...}, "mcp": {...}}: kwargs do retrieve por caminho (substitui CAMINHOS)')
    c = sub.add_parser("calibrar")
    c.add_argument("--tag", required=True)
    c.add_argument("--Ls", default="0.4,0.5,0.6,0.7,0.8")
    m = sub.add_parser("compare")
    m.add_argument("--visado", required=True)
    m.add_argument("--novo", action="append", required=True, help="colecao=tag")
    m.add_argument("--base", action="append", required=True, help="colecao=tag")
    m.add_argument("--gate-novo")
    m.add_argument("--gate-base")
    m.add_argument("--latencia-ab", action="append", help="colecao=tag do run_latency_ab")
    a = ap.parse_args(argv)
    if a.cmd == "compare":
        res = compare(a.visado, _col_tags(a.novo), _col_tags(a.base), a.gate_novo, a.gate_base, REPORTS,
                      _col_tags(a.latencia_ab or []))
        for cr in res["criterios"]:
            print(("OK   " if cr["ok"] else "FALHA"), cr["nome"], "-", json.dumps(cr["detalhe"], ensure_ascii=False))
        print("ACEITE:", "passa" if res["ok"] else "NÃO passa", "(pede OK do dono: latência)" if res["pede_dono"] else "")
        return 0 if res["ok"] else 1
    if a.cmd == "run":
        if a.caminhos:
            novos = json.loads(a.caminhos)
            if set(novos) != {"cli", "mcp"}:
                sys.exit('--caminhos pede os dois caminhos: {"cli": {...}, "mcp": {...}}')
            CAMINHOS.clear()
            CAMINHOS.update(novos)
        run(a.colecao, a.split, a.tag, a.L, a.geracao, a.latencia)
        return 0
    print(json.dumps(calibrar(a.tag, [float(x) for x in a.Ls.split(",")]), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())

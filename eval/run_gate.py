"""Portão de R9 (spec 2026-10-17-rag-assertive): linha de base e pipeline assertivo medidos no
mesmo conjunto fixo (eval/split.json), com juiz de outra família por frase e revisão humana.

  python -m eval.run_gate run --mode baseline|assertive --colecao legal_rag --split teste --tag base-legal_rag-teste
  python -m eval.run_gate check <tag> --baseline <tag-da-base>      # sai 0 só se o portão passa

`run` grava eval/reports/gate_<tag>.json e eval/reports/review_<tag>.jsonl (o que pede olho humano:
o revisor preenche `confirmado` ou `armadilha_ok`). Uma rodada por coleção (COLLECTION_NAME no ambiente)."""
import argparse
import json
import os
import random
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

REPORTS = Path(__file__).resolve().parent / "reports"
MAX_QUEDA_RESPOSTA = 0.05
MAX_QUEDA_FACET = 0.05
P50_MAX_S, P95_MAX_S = 45.0, 75.0  # meta revista na Tarefa 2 (spec, «Rapidez»): duas chamadas em série
N_AUDITORIA = 20
MOTIVOS = ("citacao_ausente", "numero_sem_prova", "nao_sustenta", "ressalva_omitida")


# ---------- agregação e portão (puro, testável) ----------

def _pct(vals: list[float], p: float) -> float | None:
    if not vals:
        return None
    v = sorted(vals)
    pos = (len(v) - 1) * p
    lo = int(pos)
    hi = min(lo + 1, len(v) - 1)
    return round(v[lo] + (v[hi] - v[lo]) * (pos - lo), 3)


def aggregate(items: list[dict]) -> dict:
    resp = [i for i in items if i["respondivel"]]
    armad = [i for i in items if not i["respondivel"]]
    facets = [i["facet_resp"] for i in resp if i.get("facet_resp") is not None]
    segs = [i["segundos"] for i in items if i.get("segundos") is not None]
    cortes = {m: 0 for m in MOTIVOS}
    for i in items:
        for c in i.get("cortadas", []):
            cortes[c.get("motivo", "?")] = cortes.get(c.get("motivo", "?"), 0) + 1
    return {
        "n": len(items), "n_respondiveis": len(resp), "n_armadilhas": len(armad),
        "taxa_resposta": round(sum(1 for i in resp if i["respondeu"]) / len(resp), 4) if resp else None,
        "facet_resp": round(sum(facets) / len(facets), 4) if facets else None,
        "p50_s": _pct(segs, 0.5), "p95_s": _pct(segs, 0.95), "n_latencia": len(segs),
        "cortes_por_motivo": cortes,
        "falha_fechada": round(sum(1 for i in items if i.get("status") == "falha_verificacao") / len(items), 4) if items else None,
    }


def build_review(items: list[dict], tag: str) -> list[dict]:
    """Tudo que pede olho humano. `confirmado` (frase/auditoria) = a frase realmente não tem base;
    `armadilha_ok` = a resposta recusou ou corrigiu a premissa, sem afirmar nada inventado."""
    rev: list[dict] = []
    aprovadas: list[dict] = []
    for it in items:
        for f in it.get("frases", []):
            linha = {"kind": "frase", "id": it["id"], "frase": f["texto"], "veredito": f["veredito"], "confirmado": None}
            if f["veredito"] in ("nao_sustenta", "indeterminado"):
                rev.append(linha)
            else:
                aprovadas.append({**linha, "kind": "auditoria"})
    rng = random.Random(tag)
    rev += rng.sample(aprovadas, min(N_AUDITORIA, len(aprovadas)))
    for it in items:
        if not it["respondivel"]:
            rev.append({"kind": "armadilha", "id": it["id"], "tipo": it["tipo"], "pergunta": it.get("pergunta", ""),
                        "resposta": it.get("resposta", ""), "frases": [f["texto"] for f in it.get("frases", [])],
                        "armadilha_ok": None})
    return rev


def _load(tag: str, reports: Path) -> tuple[dict, list[dict]]:
    gate = json.loads((reports / f"gate_{tag}.json").read_text(encoding="utf-8"))
    rp = reports / f"review_{tag}.jsonl"
    rev = [json.loads(ln) for ln in rp.read_text(encoding="utf-8").splitlines() if ln.strip()] if rp.exists() else []
    return gate, rev


def check(tag: str, baseline: str, reports: Path = REPORTS) -> dict:
    gate, rev = _load(tag, reports)
    base, _ = _load(baseline, reports)
    ag, ab = aggregate(gate["items"]), aggregate(base["items"])
    crit: list[dict] = []

    def add(nome, ok, detalhe):
        crit.append({"nome": nome, "ok": bool(ok), "detalhe": detalhe})

    # revisão completa: toda frase sinalizada e toda armadilha precisa de linha preenchida
    incompletos = []
    por = {(r["kind"], r["id"], r.get("frase")) for r in rev}
    for it in gate["items"]:
        if not it["respondivel"] and ("armadilha", it["id"], None) not in por:
            incompletos.append(f"armadilha {it['id']} sem revisão")
        for f in it.get("frases", []):
            if f["veredito"] in ("nao_sustenta", "indeterminado") and ("frase", it["id"], f["texto"]) not in por:
                incompletos.append(f"frase {it['id']} sem revisão")
    for r in rev:
        campo = "armadilha_ok" if r["kind"] == "armadilha" else "confirmado"
        if not isinstance(r.get(campo), bool):
            incompletos.append(f"{r['kind']} {r['id']} sem {campo}")
    erros = [i["id"] for i in gate["items"] if i.get("status") == "erro"]
    add("sem_erro_de_infra", not erros, f"{len(erros)} pergunta(s) com erro (rodar de novo): {erros[:5]}")
    add("revisao_completa", not incompletos, incompletos[:5] or "ok")

    sem_base = [r for r in rev if r["kind"] in ("frase", "auditoria") and r.get("confirmado") is True]
    add("sem_base_confirmada", not sem_base, f"{len(sem_base)} frase(s) sem base confirmadas")

    armad = [r for r in rev if r["kind"] == "armadilha"]
    ruins = [r["id"] for r in armad if r.get("armadilha_ok") is False]
    add("armadilhas_100pct", not ruins and len(armad) == ag["n_armadilhas"],
        f"{len(armad) - len(ruins)}/{ag['n_armadilhas']} ok; falham: {ruins}")

    if ag["taxa_resposta"] is not None and ab["taxa_resposta"] is not None:
        queda = round(ab["taxa_resposta"] - ag["taxa_resposta"], 4)
        add("taxa_resposta", queda <= MAX_QUEDA_RESPOSTA,
            f"{ag['taxa_resposta']} vs base {ab['taxa_resposta']} (queda {queda}, máx {MAX_QUEDA_RESPOSTA})")
    if ag["facet_resp"] is not None and ab["facet_resp"] is not None:
        queda = round(ab["facet_resp"] - ag["facet_resp"], 4)
        # reportado, não barra: responder só o que o trecho prova custa cobertura de facetas (dono, 2026-10-18)
        add("facet_resp", True, f"{ag['facet_resp']} vs base {ab['facet_resp']} (queda {queda}; informativo, antes máx {MAX_QUEDA_FACET})")
    if gate["colecao"] == "legal_rag":
        add("latencia", ag["p50_s"] is not None and ag["p50_s"] <= P50_MAX_S and ag["p95_s"] <= P95_MAX_S,
            f"p50 {ag['p50_s']} (máx {P50_MAX_S}), p95 {ag['p95_s']} (máx {P95_MAX_S}), n={ag['n_latencia']}")
    return {"ok": all(c["ok"] for c in crit), "criterios": crit, "agregado": ag, "base": ab}


# ---------- rodada ----------

_FIM = re.compile(r"\n\s*(⚠️|=== Fontes)")


_RECUSA = re.compile(r"^\W*(n[ãa]o\s+(encontr|localiz|consta|h[áa]\s+(essa|esse|informa|dado|trecho|registro|refer|men)|aparece|foi\s+poss[ií]vel|achei|identific)|sem\s+base)", re.I)


def recusou(resposta: str) -> bool:
    """A resposta ABRE recusando. `metrics.abstained` casa 'não ... transcri' em qualquer ponto do texto
    e marcou 12 de 15 respostas boas como recusa na primeira rodada (2026-10-17)."""
    return bool(_RECUSA.search(resposta or ""))


_LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_URL = re.compile(r"https?://\S+")
_FIM_FRASE = re.compile(r"(?<=[.!?])\s+(?=[A-ZÀ-Ú\"“(\[])")  # fim de frase: pontuação + espaço + maiúscula (530.000 e "art. 7º" não quebram)
_MARCA = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+")


def baseline_frases(resposta: str) -> list[str]:
    """Frases da resposta de hoje (texto livre): sem aviso do numbers_guard, sem lista de fontes, markdown e link
    tirados (o ponto de uma URL ou de '4.' de lista não é fim de frase)."""
    corpo = _FIM.split("\n" + resposta)[0]
    corpo = _URL.sub("", _LINK.sub(r"\1", corpo)).replace("**", "").replace("__", "")
    frases: list[str] = []
    for linha in corpo.splitlines():
        linha = _MARCA.sub("", linha).strip().lstrip("#").strip()
        if re.match(r"fontes?\b", linha, re.I):  # linha de citação de fonte, não afirmação
            continue
        frases += [x.strip() for x in _FIM_FRASE.split(linha) if len(x.split()) >= 4]
    return frases


def _run_one(row: dict, mode: str, k: int, cache, lock: threading.Lock, judge=None) -> dict:
    from eval import metrics as M
    from eval.judge import judge_phrases
    from rag.query import retrieve, contexto_citavel
    judge = judge or judge_phrases
    t0 = time.time()
    with lock:  # o reranker não é thread-safe; geração e juiz seguem em paralelo
        docs = retrieve(row["pergunta"], k)
    out = {"id": row["id"], "tipo": row["tipo"], "colecao": row["colecao"], "pergunta": row["pergunta"],
           "respondivel": not row["tipo"].startswith("armadilha"),
           "relevance_score": [d.metadata.get("relevance_score") for d in docs[:5]],
           "trechos": [d.page_content for d in contexto_citavel(docs)],
           "frases": [], "cortadas": [], "erro": None}
    chunks = [d.page_content for d in contexto_citavel(docs)]
    resposta = ""
    try:
        if mode == "baseline":
            from rag.query import gen_grok, montar_prompt  # caminho antigo, só para refazer a linha de base
            prompt, ctx = montar_prompt(row["pergunta"], docs)
            resposta = gen_grok(prompt)  # shortcut: sem o aviso de número do guard apagado; a linha de base já medida não muda
            out["segundos"] = round(time.time() - t0, 2)
            out["resposta"] = resposta
            out["respondeu"] = not recusou(resposta)
            out["status"] = "ok" if out["respondeu"] else "nao_encontrado"
            fs = baseline_frases(resposta) if out["respondeu"] else []
            out["frases"] = [{"texto": f, "veredito": v["veredito"], "model": v["model"]}
                             for f, v in zip(fs, judge(row["id"], row["pergunta"], [(f, chunks) for f in fs], cache))]
        else:
            from rag.answer import answer, format_text
            from rag.query import generator, verifier
            from rag.config import LLM_PROVIDER, RAG_DEADLINE_S
            res = answer(row["pergunta"], docs, generator(LLM_PROVIDER), verifier(), RAG_DEADLINE_S)
            out["segundos"] = round(time.time() - t0, 2)
            resposta = format_text(res)
            out.update(status=res["status"], resposta=resposta, respondeu=res["status"] == "ok",
                       cortadas=res.get("cortadas", []))
            itens = [(fr["texto"], [res["trechos"][p["trecho"] - 1]["texto"] for p in fr["provas"]]) for fr in res["frases"]]
            out["frases"] = [{"texto": f, "veredito": v["veredito"], "model": v["model"]}
                             for (f, _), v in zip(itens, judge(row["id"], row["pergunta"], itens, cache))]
    except Exception as e:  # noqa: BLE001 — uma pergunta ruim não derruba a rodada
        out.update(erro=repr(e)[:300], respondeu=False, status="erro", resposta=resposta,
                   segundos=round(time.time() - t0, 2))
    kw = row.get("expected_keywords")
    if kw and out["respondivel"]:
        out["facet_resp"] = M.answer_facet_coverage(resposta, kw) if out["respondeu"] else 0.0
    else:
        out["facet_resp"] = None
    return out


def run(mode: str, colecao: str, split: str, tag: str, k: int = 15, workers: int = 3, reports: Path = REPORTS) -> Path:
    from eval.judge import judge_phrases
    from eval.judge_cache import JudgeCache
    from eval.split import load_split
    if os.environ.get("COLLECTION_NAME") != colecao:
        sys.exit(f"COLLECTION_NAME={os.environ.get('COLLECTION_NAME')!r} != {colecao!r}: uma rodada por coleção")
    rows = [r for r in load_split()[split] if r["colecao"] == colecao]
    reports.mkdir(parents=True, exist_ok=True)
    parcial = reports / f"gate_{tag}.partial.jsonl"
    feitos = ({json.loads(ln)["id"]: json.loads(ln) for ln in parcial.read_text(encoding="utf-8").splitlines() if ln.strip()}
              if parcial.exists() else {})
    cache = JudgeCache(reports / "judge_phrase_cache.jsonl")
    lock, wlock = threading.Lock(), threading.Lock()

    def tarefa(r):
        if r["id"] in feitos:
            it = feitos[r["id"]]
            if mode == "baseline" and it.get("trechos") and it.get("respondeu"):  # refaz frases e juiz sobre a resposta já gerada (cache evita repetir)
                fs = baseline_frases(it["resposta"])
                vs = judge_phrases(r["id"], r["pergunta"], [(f, it["trechos"]) for f in fs], cache)
                it["frases"] = [{"texto": f, "veredito": v["veredito"], "model": v["model"]} for f, v in zip(fs, vs)]
            return it
        it = _run_one(r, mode, k, cache, lock)
        if it["status"] != "erro":  # erro é re-tentado na retomada
            with wlock, parcial.open("a", encoding="utf-8") as f:
                f.write(json.dumps(it, ensure_ascii=False) + "\n")
        print(f"{r['id']} {it['status']} {it['segundos']}s frases={len(it['frases'])}", flush=True)
        return it

    with ThreadPoolExecutor(max_workers=workers) as ex:
        items = list(ex.map(tarefa, rows))
    ag = aggregate(items)
    gate = {"tag": tag, "mode": mode, "colecao": colecao, "split": split, "k": k, "agregado": ag, "items": items}
    caminho = reports / f"gate_{tag}.json"
    caminho.write_text(json.dumps(gate, ensure_ascii=False, indent=1), encoding="utf-8")
    rp = reports / f"review_{tag}.jsonl"
    if not rp.exists():  # nunca sobrescreve revisão já preenchida
        rp.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in build_review(items, tag)) + "\n", encoding="utf-8")
    print(json.dumps(ag, ensure_ascii=False))
    return caminho


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--mode", choices=("baseline", "assertive"), required=True)
    r.add_argument("--colecao", required=True)
    r.add_argument("--split", required=True, choices=("calibracao", "teste"))
    r.add_argument("--tag", required=True)
    r.add_argument("--k", type=int, default=15)
    r.add_argument("--workers", type=int, default=3)
    c = sub.add_parser("check")
    c.add_argument("tag")
    c.add_argument("--baseline", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "run":
        run(a.mode, a.colecao, a.split, a.tag, a.k, a.workers)
        return 0
    res = check(a.tag, a.baseline)
    for cr in res["criterios"]:
        print(("OK   " if cr["ok"] else "FALHA"), cr["nome"], "-", cr["detalhe"])
    print("PORTÃO:", "passa" if res["ok"] else "NÃO passa")
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())

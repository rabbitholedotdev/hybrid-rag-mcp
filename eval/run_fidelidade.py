"""Fidelidade da RESPOSTA gerada (não só da recuperação) para k e sintético no contexto.

Para cada pergunta do golden e cada config (k, com/sem resumo+Q&A no contexto): busca (sem HyDE), monta o
prompt como o `rag query` (`rag.query.montar_prompt`), gera com o grok (cache em disco) e mede:
  facet_resp     facetas esperadas presentes na resposta (o que o aluno lê)
  facet_ctx      facetas presentes nos trechos primários recuperados
  sem_base       fração de palavras da resposta sem base nos primários (proxy lexical, ver eval.metrics)
  vaza_sint      nº de palavras da resposta que só existem em resumo/Q&A recuperado (vazamento da camada IA)

    python -m eval.run_fidelidade --golden eval/golden_legal.jsonl --workers 4
"""
import argparse
import json
import re
import statistics as st
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval import metrics as M  # noqa: E402
from eval.bootstrap_ci import boot_diff_ci, boot_mean_ci  # noqa: E402
from eval.run_variants import carregar  # noqa: E402
from eval.variants import ContaChamadas  # noqa: E402

REPORTS = Path(__file__).resolve().parent / "reports"
CONFIGS = [(6, True), (6, False), (12, True), (12, False), (15, True), (15, False)]


def so_texto(kws):
    """Palavras-chave que não são só número/sigla curta: número solto ("345", "8%") casa em qualquer trecho."""
    return [k for k in kws if not re.fullmatch(r"[\d.,% ]+|.{1,3}", k.split("|")[0].strip())]


def nome(k, sint):
    return f"k{k} {'com' if sint else 'sem'} sint"


def avalia(r, k, sint, docs_k, gen):
    from rag.query import montar_prompt
    prompt, ctx = montar_prompt(r["pergunta"], docs_k, sintetico=sint)
    resposta = gen(prompt)
    kws = r.get("expected_keywords", [])
    return {
        "id": r["id"], "tipo": r.get("tipo"), "k": k, "sintetico": sint,
        "facet_resp": M.answer_facet_coverage(resposta, kws) or 0.0,
        "facet_ctx": M.facet_coverage(ctx, kws) or 0.0,
        "facet_resp_txt": M.answer_facet_coverage(resposta, so_texto(kws)) or 0.0,
        "facet_ctx_txt": M.facet_coverage(ctx, so_texto(kws)) or 0.0,
        "sem_base": M.unsupported_rate(resposta, docs_k, r["pergunta"]),
        "vaza_sint": len(M.termos_so_sintetico(resposta, docs_k, r["pergunta"])),
        "vaza_termos": M.termos_so_sintetico(resposta, docs_k, r["pergunta"]),
        "ctx_chars": sum(len(d.page_content) for d in ctx),
        "resposta": resposta,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden", default="eval/golden_legal.jsonl")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out-tag", default="fidelidade")
    ap.add_argument("--configs", default=None,
                    help='lista "k:s" separada por vírgula (s=1 com resumo/Q&A no contexto, s=0 sem), ex.: "12:0,15:0"; '
                         "a primeira é a base das diferenças pareadas. Padrão: as 6 combinações de k em 6, 12, 15")
    args = ap.parse_args()
    configs = ([(int(a), b == "1") for a, b in (c.split(":") for c in args.configs.split(","))] if args.configs else CONFIGS)

    from rag import query as Q
    gen = ContaChamadas(Q.gen_grok)
    rows = carregar(args.golden)[: args.limit]
    # busca uma vez por (pergunta, k): as duas variantes de contexto partem dos mesmos trechos
    docs = {(r["id"], k): Q.retrieve(r["pergunta"], k, hyde=False) for r in rows for k in sorted({c[0] for c in configs})}
    tarefas = [(r, k, s) for r in rows for (k, s) in configs]
    with ThreadPoolExecutor(args.workers) as ex:
        saida = list(ex.map(lambda t: avalia(t[0], t[1], t[2], docs[(t[0]["id"], t[1])], gen), tarefas))
    res = {nome(k, s): [x for x in saida if x["k"] == k and x["sintetico"] == s] for k, s in configs}
    REPORTS.mkdir(exist_ok=True)
    for n, per in res.items():
        (REPORTS / f"report_{args.out_tag}_{n.replace(' ', '_')}.json").write_text(
            json.dumps({"cfg": {"config": n, "golden": args.golden}, "per": per}, ensure_ascii=False, indent=1), encoding="utf-8")

    base_n = nome(*configs[0])
    base = res[base_n]
    comp = [i for i, r in enumerate(rows) if r.get("tipo") == "composta"]
    print(f"\n| config | facet_resp (IC95) | Δ vs {base_n} | Δ compostas (n={len(comp)}) | facet_ctx | sem_base | vaza_sint (média) | ctx chars |")
    print("|---|---|---|---|---|---|---|---|")
    for n, per in res.items():
        f = [p["facet_resp"] for p in per]
        m = boot_mean_ci(f)
        d = boot_diff_ci(f, [p["facet_resp"] for p in base]) if per is not base else (0, 0, 0)
        dc = boot_diff_ci([f[i] for i in comp], [base[i]["facet_resp"] for i in comp]) if per is not base else (0, 0, 0)
        print(f"| {n} | {m[0]} [{m[1]}, {m[2]}] | {d[0]:+.3f} [{d[1]:+.3f}, {d[2]:+.3f}] | {dc[0]:+.3f} [{dc[1]:+.3f}, {dc[2]:+.3f}] | "
              f"{st.mean(p['facet_ctx'] for p in per):.3f} | {st.mean(p['sem_base'] for p in per):.3f} | "
              f"{st.mean(p['vaza_sint'] for p in per):.2f} | {st.mean(p['ctx_chars'] for p in per):.0f} |")
    for k in sorted({c[0] for c in configs}):
        if (k, True) in configs and (k, False) in configs:
            if k == sorted({c[0] for c in configs})[0]:
                print("\nDiferença pareada de sem_base e vaza_sint entre com e sem sintético (mesmo k; negativo = melhora sem sintético):")
            a, b = res[nome(k, False)], res[nome(k, True)]
            for met in ("sem_base", "vaza_sint"):
                d = boot_diff_ci([p[met] for p in a], [p[met] for p in b])
                print(f"  k{k} {met}: {d[0]:+.3f} [{d[1]:+.3f}, {d[2]:+.3f}]")
    if len(configs) >= 2:
        ult_n, ult = nome(*configs[-1]), res[nome(*configs[-1])]
        print(f"\nPareado {ult_n} − {base_n}:")
        for met in ("facet_resp", "facet_resp_txt", "facet_ctx", "facet_ctx_txt", "sem_base", "vaza_sint", "ctx_chars"):
            d = boot_diff_ci([p[met] for p in ult], [p[met] for p in base])
            print(f"  {met}: {d[0]:+.3f} [{d[1]:+.3f}, {d[2]:+.3f}]")
        mais = sum(1 for a, b in zip(ult, base) if a["facet_resp"] > b["facet_resp"])
        menos = sum(1 for a, b in zip(ult, base) if a["facet_resp"] < b["facet_resp"])
        print(f"  perguntas em que {ult_n} melhora: {mais}; piora: {menos}; empata: {len(base) - mais - menos}")
    print(f"\nchamadas lógicas ao LLM: {gen.chamadas}")


if __name__ == "__main__":
    main()

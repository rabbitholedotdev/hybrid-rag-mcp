"""Cobertura de facetas (k=15) numa coleção, por pergunta, salva em JSON para comparar embeddings.

    COLLECTION_NAME=legal_t_e5 python -m eval.run_embed_ab --out /tmp/ab_e5.json
    python -m eval.run_embed_ab --compara /tmp/ab_e5.json /tmp/ab_bgem3.json
O modelo de embedding vem de EMBED_MODEL (mesmo valor da ingestão da coleção)."""
import argparse
import json
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval import metrics as M  # noqa: E402
from eval.bootstrap_ci import boot_diff_ci, boot_mean_ci  # noqa: E402
from eval.run_variants import carregar  # noqa: E402
from ingest.ingest import sem_cabecalho  # noqa: E402

K = 15


def compara(a: dict, b: dict) -> str:
    ids = [i for i in a if i in b]
    fa, fb = [a[i] for i in ids], [b[i] for i in ids]
    d = boot_diff_ci(fb, fa)
    return (f"{len(ids)} perguntas | A {boot_mean_ci(fa)[0]} | B {boot_mean_ci(fb)[0]} | "
            f"Δ B−A {d[0]:+.3f} [{d[1]:+.3f}, {d[2]:+.3f}]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out")
    ap.add_argument("--compara", nargs=2)
    a = ap.parse_args()
    if a.compara:
        x, y = (json.loads(Path(p).read_text()) for p in a.compara)
        print(compara(x, y)); return
    from rag import query as Q
    rows = carregar("eval/golden_legal.jsonl")
    res = {}
    for r in rows:
        docs = Q.retrieve(r["pergunta"], K)
        for d in docs:  # cobertura só no conteúdo: o cabeçalho de contexto não pode inflar a métrica
            d.page_content = sem_cabecalho(d.page_content)
        res[r["id"]] = M.facet_coverage(docs, r.get("expected_keywords", [])) or 0.0
    Path(a.out).write_text(json.dumps(res))
    print(f"{len(res)} perguntas, facet_ctx médio {st.mean(res.values()):.3f} -> {a.out}")


if __name__ == "__main__":
    main()

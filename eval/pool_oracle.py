"""Oráculo do pool (R5, fase 2): para cada pergunta do funil, as listas só-densa, só-esparsa e híbrida (30, 75
e 150) com a cobertura de cada chunk primário, em busca exata. `eval/pool_sim.py` lê o resultado e diz em
quantas perguntas a evidência (cobertura >= L) entra num pool de N candidatos, para qualquer ordem de fusão,
sem rerank (o rerank quase não mata: degrau 3 tem 0 a 2 mortes).

  COLLECTION_NAME=<c> python -m eval.pool_oracle <c>      # grava eval/reports/pool_<c>.jsonl, retoma pelos ids"""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.funil import cobertura_chunk  # noqa: E402
from eval.funil_dados import load_rows  # noqa: E402
from eval.metrics import is_citavel  # noqa: E402
from eval.run_funnel import MIN_FACETAS  # noqa: E402

REPORTS = Path(__file__).resolve().parent / "reports"
PROFUNDIDADES = {"F30": 30, "F75": 75, "F150": 150}


def main(col: str) -> None:
    if os.environ.get("COLLECTION_NAME") != col:
        sys.exit(f"COLLECTION_NAME={os.environ.get('COLLECTION_NAME')!r} != {col!r}: uma rodada por coleção")
    from langchain_qdrant import RetrievalMode
    from qdrant_client.http import models as qm
    from rag.vectorstore import get_vectorstore

    rows = [r for r in load_rows("calibracao", col) if len(r["expected_keywords"]) >= MIN_FACETAS]
    sp = qm.SearchParams(exact=True)
    out = REPORTS / f"pool_{col}.jsonl"
    feitos = set()
    if out.exists():
        with out.open(encoding="utf-8") as f:
            feitos = {json.loads(ln)["id"] for ln in f if ln.strip()}
    vs = {m: get_vectorstore(create=False, collection=col, retrieval_mode=m)
          for m in (RetrievalMode.DENSE, RetrievalMode.SPARSE, RetrievalMode.HYBRID)}

    def lista(modo, q, k, kw):
        ps = vs[modo].similarity_search_with_score(q, k=k, search_params=sp)
        return [[d.metadata.get("video_id"), d.metadata.get("chunk"), int(is_citavel(d)),
                 round(cobertura_chunk(d, kw), 3) if is_citavel(d) else 0.0] for d, _ in ps]

    for r in rows:
        if r["id"] in feitos:
            continue
        q, kw = r["pergunta"], r["expected_keywords"]
        rec = {"id": r["id"], "golden": r["golden"],
               "D": lista(RetrievalMode.DENSE, q, 150, kw), "S": lista(RetrievalMode.SPARSE, q, 150, kw),
               **{nome: lista(RetrievalMode.HYBRID, q, k, kw) for nome, k in PROFUNDIDADES.items()}}
        with out.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        print(r["id"], flush=True)


if __name__ == "__main__":
    main(sys.argv[1])

"""Refaz só o juiz das perguntas de um relatório que caíram em fallback lexical
(juiz opencode instável), sem refazer retrieve/geração (~40 min/run).
Precisa de `primary_ctx` no relatório (gravado por run_eval a partir de 2026-09-26).

Uso: venv/bin/python -m eval.rejudge eval/reports/report_..._r2_pools_k6.json
Reescreve o JSON (agregados faithfulness/disagreement/judge_fallbacks) e sai != 0
se ainda sobrar fallback."""
import hashlib
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from eval.judge import faithfulness
from eval.judge_cache import JudgeCache
from eval.run_eval import HERE, mean
from rag.query import PROMPT


def main(path: str) -> int:
    rep = json.loads(Path(path).read_text(encoding="utf-8"))
    cfg, per = rep["config"], rep["per"]
    cache = JudgeCache(HERE / ".judge_cache.jsonl")
    ph = hashlib.sha256(PROMPT.encode()).hexdigest()[:10]
    alvo = [x for x in per if x.get("faithfulness", {}).get("fallback")]
    print(f"{len(alvo)} pergunta(s) com fallback em {path}", file=sys.stderr)

    def refaz(x):
        x["faithfulness"] = faithfulness(x["answer"], x["primary_ctx"], x["id"], cfg["variant"], ph, cache=cache)
    with ThreadPoolExecutor(max_workers=4) as ex:
        list(ex.map(refaz, alvo))

    agg = rep["agg"]
    agg["faithfulness"] = mean([x["faithfulness"]["score"] for x in per if "faithfulness" in x])
    agg["faithfulness_disagreement"] = mean(
        [x["faithfulness"]["disagreement"] for x in per
         if x.get("faithfulness", {}).get("disagreement") is not None])
    agg["judge_fallbacks"] = sum(1 for x in per if x.get("faithfulness", {}).get("fallback"))
    Path(path).write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"judge_fallbacks restantes: {agg['judge_fallbacks']}", file=sys.stderr)
    return 1 if agg["judge_fallbacks"] else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))

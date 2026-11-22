"""Fixed eval split: calibration vs held-out test.

Prompt, threshold, or normalization changes may look only at `calibracao`.
The gate measures `teste`. `eval/split.json` is written once (ids); the
stability test fails if it is regenerated accidentally.
"""
import json
import random
from collections import defaultdict
from pathlib import Path

EVAL = Path(__file__).resolve().parent
SEED = 20261017


def _jsonl(nome: str) -> list[dict]:
    return [json.loads(ln) for ln in (EVAL / nome).read_text(encoding="utf-8").splitlines() if ln.strip()]


def _todas(seed: int = SEED) -> list[dict]:
    rows = []
    for r in _jsonl("golden_legal.jsonl"):
        rows.append({
            "id": r["id"],
            "colecao": "legal_rag",
            "tipo": r["tipo"],
            "pergunta": r["pergunta"],
            "expected_keywords": r.get("expected_keywords", []),
        })
    for r in _jsonl("traps_legal.jsonl"):
        rows.append({
            "id": r["id"],
            "colecao": r["colecao"],
            "tipo": r["tipo"],
            "pergunta": r["pergunta"],
            "nota": r.get("nota", ""),
        })
    return rows


def build_split(seed: int = SEED) -> dict[str, list[str]]:
    """ids per side. Half of each (collection, type) stratum goes to each side; remainder to `teste`."""
    estratos: dict[tuple, list[str]] = defaultdict(list)
    for r in _todas(seed):
        estratos[(r["colecao"], r["tipo"])].append(r["id"])
    rng = random.Random(seed)
    cal: list[str] = []
    tst: list[str] = []
    for chave in sorted(estratos):
        ids = sorted(estratos[chave])
        rng.shuffle(ids)
        meio = len(ids) // 2
        cal += ids[:meio]
        tst += ids[meio:]
    return {"calibracao": sorted(cal), "teste": sorted(tst)}


def load_split() -> dict[str, list[dict]]:
    ids = json.loads((EVAL / "split.json").read_text(encoding="utf-8"))
    por_id = {r["id"]: r for r in _todas()}
    return {lado: [por_id[i] for i in ids[lado]] for lado in ("calibracao", "teste")}


if __name__ == "__main__":
    (EVAL / "split.json").write_text(
        json.dumps(build_split(), ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )

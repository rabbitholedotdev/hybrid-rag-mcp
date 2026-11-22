"""Funnel questions (R5) and their split.

Covers every non-trap golden in `eval/golden_legal.jsonl` on collection `legal_rag`.
Ids already drawn in `split.json` keep their side; any leftovers go to
`split_funil.json` by hash.
"""
import hashlib
import json

from eval.split import EVAL, _jsonl

SPLIT_FUNIL = EVAL / "split_funil.json"
FACETAS = EVAL / "facetas.jsonl"


def _facetas() -> dict[str, list[str]]:
    if not FACETAS.exists():
        return {}
    return {r["id"]: r["expected_keywords"] for r in _jsonl(FACETAS.name)}


def todas_funil() -> list[dict]:
    rows = []
    fac = _facetas()
    for r in _jsonl("golden_legal.jsonl"):
        rid = r["id"]
        rows.append({
            "id": rid,
            "colecao": "legal_rag",
            "golden": "legal",
            "pergunta": r["pergunta"],
            "expected_ids": r.get("expected_ids", []),
            "expected_keywords": r.get("expected_keywords") or fac.get(rid, []),
        })
    return rows


def _split_r9() -> dict[str, list[str]]:
    return json.loads((EVAL / "split.json").read_text(encoding="utf-8"))


def build_split_funil() -> dict[str, list[str]]:
    ja = {i for lado in _split_r9().values() for i in lado}
    novos = [r["id"] for r in todas_funil() if r["id"] not in ja]
    cal = [i for i in novos if int(hashlib.sha1(i.encode()).hexdigest(), 16) % 2 == 0]
    return {"calibracao": sorted(cal), "teste": sorted(set(novos) - set(cal))}


def load_rows(split: str, colecao: str) -> list[dict]:
    ids = set(_split_r9()[split]) | set(json.loads(SPLIT_FUNIL.read_text(encoding="utf-8"))[split])
    return [r for r in todas_funil() if r["id"] in ids and r["colecao"] == colecao]


if __name__ == "__main__":
    SPLIT_FUNIL.write_text(
        json.dumps(build_split_funil(), ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )

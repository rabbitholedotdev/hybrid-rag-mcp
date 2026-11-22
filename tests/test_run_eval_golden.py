"""load_golden() lê o formato real produzido por eval/gen_golden.py."""
import json
from pathlib import Path

from eval.run_eval import load_golden

GOLDEN = Path(__file__).resolve().parent.parent / "eval" / "golden.jsonl"


def test_golden_real_carrega_sem_adaptacao():
    rows = load_golden(str(GOLDEN), include_draft=False)
    n = sum(1 for ln in GOLDEN.read_text(encoding="utf-8").splitlines() if ln.strip())
    assert len(rows) == n > 0
    for r in rows:
        assert r["question"] and r["video_ids"] and r["id"]


def test_curated_false_so_com_include_draft(tmp_path):
    p = tmp_path / "g.jsonl"
    p.write_text("\n".join(json.dumps(x) for x in [
        {"pergunta": "a?", "expected_ids": ["v1"]},
        {"question": "b?", "video_id": "v2", "curated": False},
    ]), encoding="utf-8")
    assert [r["question"] for r in load_golden(str(p), False)] == ["a?"]
    rows = load_golden(str(p), True)
    assert [r["video_ids"] for r in rows] == [["v1"], ["v2"]]

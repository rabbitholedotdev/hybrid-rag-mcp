"""Gera eval/golden.jsonl automaticamente a partir da camada de Q&A sintética.

Cada record `syn-qa-<vid>` tem perguntas (linhas "P:") cujo gabarito é a aula de
origem `<vid>`. Isso vira um golden set grande e fiel, cobrindo todas as áreas —
sem escrever pergunta na mão.

Roda LOCAL na box (conecta no Qdrant em localhost). Amostra N perguntas por área.

Uso:  python -m eval.gen_golden --por-area 12 --out eval/golden.jsonl
"""
import argparse
import json
import os
import re
import sys
import urllib.request
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
QURL = os.getenv("QDRANT_URL", "http://127.0.0.1:6333")
COLL = os.getenv("COLLECTION_NAME", "legal_rag")
_P = re.compile(r"P:\s*(.+?)(?:\n|$)")


def scroll_qa():
    off = None
    while True:
        body = {"limit": 4000, "with_payload": ["page_content", "metadata.video_id",
                "metadata.area", "metadata.title"], "with_vector": False,
                "filter": {"must": [{"key": "metadata.source_type",
                                     "match": {"value": "qa"}}]}}
        if off is not None:
            body["offset"] = off
        r = urllib.request.Request(QURL + f"/collections/{COLL}/points/scroll",
                                   data=json.dumps(body).encode(),
                                   headers={"Content-Type": "application/json"})
        res = json.load(urllib.request.urlopen(r, timeout=120))["result"]
        for p in res["points"]:
            yield p.get("payload") or {}
        off = res.get("next_page_offset")
        if off is None:
            break


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--por-area", type=int, default=12)
    ap.add_argument("--out", default=str(ROOT / "eval" / "golden.jsonl"))
    a = ap.parse_args()

    # 1 pergunta por aula (video_id), agrupado por área
    by_area = defaultdict(list)
    seen_vid = set()
    for pl in scroll_qa():
        m = pl.get("metadata", {}) or {}
        vid = m.get("video_id", "")           # syn-qa-<orig>
        orig = vid[len("syn-qa-"):] if vid.startswith("syn-qa-") else vid
        if not orig or orig in seen_vid:
            continue
        mt = _P.search(pl.get("page_content", ""))
        if not mt:
            continue
        q = mt.group(1).strip()
        if len(q) < 15:
            continue
        seen_vid.add(orig)
        by_area[m.get("area", "?")].append(
            {"pergunta": q, "area": m.get("area"), "expected_ids": [orig],
             "fonte": (m.get("title") or "")[:60]})

    # amostra determinística (sem random — pega os primeiros N por área)
    out = []
    for area, items in sorted(by_area.items()):
        out.extend(items[:a.por_area])
    with open(a.out, "w", encoding="utf-8") as fh:
        for it in out:
            fh.write(json.dumps(it, ensure_ascii=False) + "\n")
    print(f"golden gerado: {len(out)} perguntas | {len(by_area)} áreas -> {a.out}")
    for area, items in sorted(by_area.items()):
        print(f"  {area}: {min(len(items), a.por_area)}")


if __name__ == "__main__":
    main()

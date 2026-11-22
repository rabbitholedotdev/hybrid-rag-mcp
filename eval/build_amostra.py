"""Amostra para comparar embeddings sem tocar a coleção servida: os documentos do gabarito dos goldens
mais N distratores sorteados (seed fixa), com a camada sintética (resumo/qa) de cada um.

    python -m eval.build_amostra --data ./data --out legal-test-sample --distratores 150
Depois: COLLECTION_NAME=legal_t_<modelo> EMBED_MODEL=<modelo> python -m ingest.ingest --channel legal-test-sample --no-replace
"""
import argparse
import json
import random
import shutil
from pathlib import Path


def ids_do_gabarito(goldens: list[str]) -> set[str]:
    ids: set[str] = set()
    for g in goldens:
        for ln in Path(g).read_text(encoding="utf-8").splitlines():
            if ln.strip():
                ids.update(json.loads(ln).get("expected_ids", []))
    return ids


def monta(primarios: list[str], gold: set[str], n: int, seed: int = 0) -> list[str]:
    """Gabarito primeiro, depois n distratores (ordem estável)."""
    resto = sorted(p for p in primarios if p not in gold)
    random.Random(seed).shuffle(resto)
    return sorted(gold & set(primarios)) + resto[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--src", default="legal")
    ap.add_argument("--out", default="legal-test-sample")
    ap.add_argument("--distratores", type=int, default=150)
    ap.add_argument("--goldens", nargs="+", default=["eval/golden_legal.jsonl"])
    a = ap.parse_args()
    data = Path(a.data)
    if a.out in (a.src, f"{a.src}-syn"):
        raise SystemExit("recuso sobrescrever o canal de origem")
    src, syn, dst = data / a.src, data / f"{a.src}-syn", data / a.out
    primarios = [p.stem for p in src.glob("*.json") if not p.name.startswith("._")]
    escolhidos = monta(primarios, ids_do_gabarito(a.goldens), a.distratores)
    dst.mkdir(exist_ok=True)
    n_syn = 0
    for vid in escolhidos:
        shutil.copy2(src / f"{vid}.json", dst / f"{vid}.json")
        for s in syn.glob(f"syn-*-{vid}.json"):
            shutil.copy2(s, dst / s.name); n_syn += 1
        for s in syn.glob(f"syn-*-{vid}-p*.json"):
            shutil.copy2(s, dst / s.name); n_syn += 1
    print(f"{len(escolhidos)} documentos + {n_syn} registros sintéticos em {dst}")


if __name__ == "__main__":
    main()

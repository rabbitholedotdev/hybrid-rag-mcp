import json
from collections import Counter
from pathlib import Path

from eval.split import build_split, load_split

EVAL = Path(__file__).resolve().parent.parent / "eval"


def n_traps(colecao: str | None = None) -> int:
    n = 0
    for ln in (EVAL / "traps_legal.jsonl").read_text(encoding="utf-8").splitlines():
        if ln.strip() and (colecao is None or json.loads(ln)["colecao"] == colecao):
            n += 1
    return n


def n_gold() -> int:
    return sum(1 for ln in (EVAL / "golden_legal.jsonl").read_text(encoding="utf-8").splitlines() if ln.strip())


def test_split_cobre_tudo_sem_sobrepor():
    s = load_split()
    cal, tst = {r["id"] for r in s["calibracao"]}, {r["id"] for r in s["teste"]}
    assert not cal & tst
    assert len(cal | tst) == n_gold() + n_traps()


def test_armadilhas_por_colecao():
    assert n_traps("legal_rag") >= 12


def test_armadilhas_sutis_por_colecao():
    s = load_split()
    sutis = [r for r in s["calibracao"] + s["teste"]
             if r["colecao"] == "legal_rag" and r["tipo"] == "armadilha-sutil"]
    assert len(sutis) >= 6


def test_ids_das_armadilhas_do_smoke():
    ids = {r["id"] for r in load_split()["calibracao"] + load_split()["teste"]}
    assert {"t1", "t2", "t3", "t4", "t5", "t6"} <= ids


def test_split_estratifica_por_colecao_e_tipo():
    s = load_split()
    lados = {k: Counter((r["colecao"], r["tipo"]) for r in s[k]) for k in ("calibracao", "teste")}
    total = lados["calibracao"] + lados["teste"]
    for estrato, n in total.items():
        if n >= 2:
            assert lados["calibracao"][estrato] >= 1 and lados["teste"][estrato] >= 1, estrato


def test_split_json_e_estavel():
    assert build_split() == json.loads((EVAL / "split.json").read_text())


def test_linhas_tem_campos_do_contrato():
    for r in load_split()["calibracao"] + load_split()["teste"]:
        assert {"id", "colecao", "tipo", "pergunta"} <= set(r)
        assert r["colecao"] == "legal_rag"

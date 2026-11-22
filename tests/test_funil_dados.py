"""Perguntas e split do funil (R5): split estável e disjunto do gate, sem armadilhas."""
import json

from eval.build_facetas import filtra_facetas
from eval.funil_dados import SPLIT_FUNIL, build_split_funil, load_rows, todas_funil
from eval.split import EVAL


def _r9():
    return json.loads((EVAL / "split.json").read_text(encoding="utf-8"))


def test_split_funil_estavel_e_disjunto():
    s = build_split_funil()
    assert s == json.loads(SPLIT_FUNIL.read_text(encoding="utf-8"))
    r9 = {i for lado in _r9().values() for i in lado}
    novos = set(s["calibracao"]) | set(s["teste"])
    assert not novos & r9 and not set(s["calibracao"]) & set(s["teste"])
    ids_funil = {r["id"] for r in todas_funil()}
    assert len(novos) + len(r9 & ids_funil) == len(todas_funil()) == len(ids_funil)


def test_load_rows_sem_armadilha_e_por_colecao():
    rows = load_rows("calibracao", "legal_rag")
    assert rows and all(r["colecao"] == "legal_rag" for r in rows)
    assert not any(r["id"].startswith("t") and r["id"][1:].isdigit() for r in rows)
    assert all("armadilha" not in r.get("tipo", "") for r in rows)


def test_filtro_faceta_literal():
    assert filtra_facetas(["pessoalidade", "inventada"], "A pessoalidade é...") == ["pessoalidade"]
    assert filtra_facetas(["subordinacao"], "requisito da subordinação jurídica") == ["subordinacao"]

"""`eval/pool_sim.py` (R5): fusão RRF local sobre as listas do oráculo do pool."""
from eval import pool_sim as PS


def _e(vid, chunk, cob=0.0):
    return [vid, chunk, 1, cob]


def test_rrf_mistura_densa_e_esparsa_e_respeita_o_peso():
    r = {"D": [_e("a", 0), _e("b", 0), _e("c", 0)], "S": [_e("b", 0), _e("c", 0), _e("a", 0)]}
    assert PS.rrf(r, 3, 3)[0] == ("b", 0)  # bem colocado nas duas listas
    assert PS.rrf(r, 3, 1, wd=5, ws=1) == [("a", 0)]  # peso no denso: o 1º do denso
    assert PS.rrf(r, 3, 1, wd=1, ws=5) == [("b", 0)]  # peso no esparso: o 1º do esparso


def test_rrf_profundidade_corta_as_listas():
    r = {"D": [_e("a", 0), _e("b", 0)], "S": [_e("x", 0), _e("b", 0)]}
    assert set(PS.rrf(r, 1, 10)) == {("a", 0), ("x", 0)}

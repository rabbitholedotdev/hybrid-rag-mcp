"""`eval/run_latency_ab.py` (R5): as duas variantes rodam na mesma passada, em ordem alternada por pergunta."""
import json
from unittest.mock import patch

from eval import run_latency_ab as AB


def test_medir_alterna_ordem_e_calcula_razao():
    chamadas = []
    relogio = iter(range(1000))

    def retrieve(pergunta, **cfg):
        chamadas.append((pergunta, cfg["id"]))

    rows = [{"pergunta": f"p{i}"} for i in range(4)]
    with patch.object(AB.time, "perf_counter", side_effect=lambda: next(relogio)):
        res = AB.medir(rows, {"cli": {"id": "a"}}, {"cli": {"id": "b"}}, retrieve=retrieve)
    medidas = chamadas[2:]  # as duas primeiras são o aquecimento
    assert [v for _, v in medidas] == ["a", "b", "b", "a", "a", "b", "b", "a"]
    assert res["cli"]["a"]["n"] == 4 and res["cli"]["b"]["n"] == 4
    assert res["cli"]["razao"] == {"p50": 1.0, "p95": 1.0}  # relógio fake: 1 s por chamada


def test_medir_razao_reflete_a_variante_mais_lenta():
    tempo = {"v": 0.0}

    def retrieve(pergunta, **cfg):
        tempo["v"] += cfg["custo"]

    rows = [{"pergunta": f"p{i}"} for i in range(5)]
    with patch.object(AB.time, "perf_counter", side_effect=lambda: tempo["v"]):
        res = AB.medir(rows, {"mcp": {"custo": 1.0}}, {"mcp": {"custo": 1.5}}, retrieve=retrieve)
    assert res["mcp"]["razao"] == {"p50": 1.5, "p95": 1.5}
    json.dumps(res)


def test_medir_sem_tempo_na_base_nao_divide_por_zero():
    with patch.object(AB.time, "perf_counter", return_value=0.0):
        res = AB.medir([{"pergunta": "p"}], {"cli": {}}, {"cli": {}}, retrieve=lambda *a, **k: None)
    assert res["cli"]["razao"] == {"p50": None, "p95": None}

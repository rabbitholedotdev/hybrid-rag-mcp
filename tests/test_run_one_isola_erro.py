"""Achado da revisão do PR #6: exceção em `_gen`/`retrieve` dentro de
`_run_one` propagava e derrubava `list(ex.map(...))` inteiro, perdendo os
resultados de todas as outras perguntas. Confirma que agora não."""
from types import SimpleNamespace
from unittest.mock import patch

from eval.run_eval import _run_one


def _args(**over):
    base = dict(k=6, use_area=False, primary_only=False, pools_separados=False,
                penalidade_sintetico=0.0, no_answer=True, no_judge=True, variant_tag="base")
    base.update(over)
    return SimpleNamespace(**base)


def test_erro_no_retrieve_nao_propaga_e_devolve_rec_com_erro():
    r = {"id": "q1", "tipo": "factual", "area": "x", "question": "pergunta?", "video_ids": []}
    with patch("eval.run_eval.retrieve", side_effect=RuntimeError("qdrant fora")):
        rec = _run_one(1, 1, r, _args(), cache=None)
    assert rec["id"] == "q1"
    assert rec["erro"] == "qdrant fora"
    assert rec["primary_recall"] == 0.0


def test_erro_no_gen_nao_propaga_e_devolve_rec_com_erro():
    r = {"id": "q1", "tipo": "factual", "area": "x", "question": "pergunta?", "video_ids": []}
    with patch("eval.run_eval.retrieve", return_value=[]), \
         patch("eval.run_eval._gen", side_effect=RuntimeError("claude -p falhou")):
        rec = _run_one(1, 1, r, _args(no_answer=False), cache=None)
    assert rec["erro"] == "claude -p falhou"


def test_sem_erro_devolve_rec_sem_chave_erro():
    r = {"id": "q1", "tipo": "factual", "area": "x", "question": "pergunta?", "video_ids": []}
    with patch("eval.run_eval.retrieve", return_value=[]):
        rec = _run_one(1, 1, r, _args(), cache=None)
    assert "erro" not in rec


def test_run_all_reroda_so_as_perguntas_com_erro():
    from eval.run_eval import _run_all
    rows = [{"id": f"q{i}", "tipo": "factual", "area": "x", "question": "?", "video_ids": []}
            for i in range(3)]
    chamadas = {"q1": 0}

    def retrieve_falha_uma_vez(q, **kw):
        return []

    def fake_run_one(i, total, r, args, cache):
        if r["id"] == "q1":
            chamadas["q1"] += 1
            if chamadas["q1"] == 1:
                return {"id": "q1", "tipo": "factual", "erro": "No route to host"}
        return {"id": r["id"], "tipo": "factual"}

    with patch("eval.run_eval._run_one", side_effect=fake_run_one):
        per = _run_all(rows, SimpleNamespace(workers=2), None, pausa=0)
    assert all("erro" not in x for x in per)
    assert chamadas["q1"] == 2

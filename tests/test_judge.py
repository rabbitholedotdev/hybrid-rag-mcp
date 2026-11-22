from unittest.mock import patch

import pytest

from eval.judge import DEFAULT_JUDGES, _esfriando, faithfulness
from eval.judge_cache import JudgeCache

CTX = "O curso dura 8 semanas e usa o framework AIDA."


@pytest.fixture(autouse=True)
def _sem_cooldown_entre_testes():
    """_esfriando é estado global do módulo — sem isto, um teste que provoca
    falha esfria o modelo pros testes seguintes (achado 2026-09-27)."""
    _esfriando.clear()
    yield
    _esfriando.clear()


def test_dois_juizes_concordam_media_e_discordancia_zero():
    with patch("eval.judge.judge_opencode") as m:
        m.side_effect = ['{"score": 0.9}', '{"score": 0.9}']
        r = faithfulness("O curso dura 8 semanas.", CTX, "q1", "base", "hash1")
    assert r["score"] == 0.9
    assert r["disagreement"] == 0.0


def test_juizes_discordam_media_e_diff_absoluta():
    with patch("eval.judge.judge_opencode") as m:
        m.side_effect = ['{"score": 1.0}', '{"score": 0.4}']
        r = faithfulness("...", CTX, "q1", "base", "hash1")
    assert r["score"] == 0.7
    assert r["disagreement"] == 0.6


def test_juiz_que_falha_cai_no_fallback_lexical():
    dois = DEFAULT_JUDGES[:2]
    with patch("eval.judge.judge_opencode") as m:
        m.side_effect = [RuntimeError("timeout"), RuntimeError("timeout"), '{"score": 0.8}']
        r = faithfulness("O curso dura 8 semanas.", CTX, "q1", "base", "hash1", judges=dois, n_juizes=2)
    modelo_falho = dois[0][1]
    assert 0.0 <= r["per_judge"][modelo_falho] <= 1.0
    assert r["fallback"] == [modelo_falho]


def test_pool_pula_modelo_esgotado_e_completa_com_o_proximo():
    """Achado 2026-09-27: cota esgota de forma imprevisível — o painel deve
    conseguir 2 julgamentos reais andando pelo pool, não travar em 2 fixos."""
    from eval.judge import _esfriando
    dois_falham_terceiro_ok = DEFAULT_JUDGES[:3]
    try:
        with patch("eval.judge.judge_opencode") as m:
            m.side_effect = [
                RuntimeError("provider.quota: Go usage limit exceeded (429)"),
                RuntimeError("provider.quota: Go usage limit exceeded (429)"),  # retry do 1º
                RuntimeError("provider.quota: Go usage limit exceeded (429)"),
                RuntimeError("provider.quota: Go usage limit exceeded (429)"),  # retry do 2º
                '{"score": 0.9}',
            ]
            r = faithfulness("...", CTX, "q1", "base", "hash1", judges=dois_falham_terceiro_ok, n_juizes=2)
        assert r["per_judge"][dois_falham_terceiro_ok[2][1]] == 0.9
        assert len(r["per_judge"]) == 2  # painel continua com só 2, não 3
        assert r["fallback"] == [dois_falham_terceiro_ok[0][1]]  # só a vaga que faltou
        assert m.call_count == 5
    finally:  # isola o cooldown desta chamada (dict de módulo é global entre testes)
        for _, model in dois_falham_terceiro_ok:
            _esfriando.pop(model, None)


def test_modelo_em_cooldown_e_pulado_sem_chamar():
    from eval.judge import _esfriar, _esfriando
    m0, m1 = DEFAULT_JUDGES[0][1], DEFAULT_JUDGES[1][1]
    _esfriando.pop(m1, None)  # isola de cooldown que outro teste possa ter deixado
    _esfriar(m0)
    try:
        with patch("eval.judge.judge_opencode") as m:
            m.return_value = '{"score": 0.7}'
            r = faithfulness("...", CTX, "q1", "base", "hash1", judges=DEFAULT_JUDGES[:2], n_juizes=1)
        assert m0 not in r["per_judge"]
        assert r["per_judge"][m1] == 0.7
    finally:
        _esfriando.pop(m0, None)


def test_fallback_lexical_nao_e_cacheado(tmp_path):
    cache = JudgeCache(tmp_path / "cache.jsonl")
    with patch("eval.judge.judge_opencode", side_effect=RuntimeError("timeout")):
        faithfulness("O curso dura 8 semanas.", CTX, "q1", "base", "hash1", cache=cache)
    _esfriando.clear()  # a falha acima esfriou os 2 primeiros; este teste não é sobre cooldown
    with patch("eval.judge.judge_opencode", return_value='{"score": 0.9}') as m:
        r = faithfulness("O curso dura 8 semanas.", CTX, "q1", "base", "hash1", cache=cache)
    assert m.call_count == 2 and r["fallback"] == []


def test_resposta_diferente_nao_reaproveita_nota_do_cache(tmp_path):
    cache = JudgeCache(tmp_path / "cache.jsonl")
    with patch("eval.judge.judge_opencode", return_value='{"score": 0.9}'):
        faithfulness("resposta A", CTX, "q1", "base", "hash1", cache=cache)
    with patch("eval.judge.judge_opencode", return_value='{"score": 0.2}') as m:
        r = faithfulness("resposta B", CTX, "q1", "base", "hash1", cache=cache)
    assert m.call_count == 2 and r["score"] == 0.2


def test_cache_evita_segunda_chamada(tmp_path):
    cache = JudgeCache(tmp_path / "cache.jsonl")
    from eval.judge import _cache_id
    cid = _cache_id("hash1", "...", CTX)
    for provider, model in DEFAULT_JUDGES:
        cache.set(cache.key("q1", "base", cid, model), 0.5)
    with patch("eval.judge.judge_opencode") as m:
        r = faithfulness("...", CTX, "q1", "base", "hash1", cache=cache)
    m.assert_not_called()
    assert r["score"] == 0.5


def test_sanitiza_angulos_do_contexto_e_resposta_antes_do_opencode():
    from eval.judge import _sanitize_para_opencode
    assert _sanitize_para_opencode("x < 5 e y > 2") == "x ( 5 e y ) 2"


def test_faithfulness_no_hang_com_angulos_na_resposta():
    with patch("eval.judge.judge_opencode") as m:
        m.return_value = '{"score": 0.9}'
        faithfulness("resposta com < e > dentro", "contexto com < e > também", "q1", "base", "hash1")
    # o prompt de fato enviado não deve carregar < ou > vindos do conteúdo
    prompt_enviado = m.call_args_list[0].args[0]
    assert "resposta com ( e )" in prompt_enviado


# --- judge_phrase (portão de R9): juiz por frase contra o chunk inteiro ---
from eval.judge import judge_phrase, judge_phrases  # noqa: E402

J = [("codex", "codex-gpt")]


def test_judge_phrase_veredito_e_cache(tmp_path):
    cache = JudgeCache(tmp_path / "c.jsonl")
    with patch("eval.judge.judge_codex", return_value='{"veredictos": [{"i": 1, "sustenta": true}]}') as m:
        r1 = judge_phrase("q1", "q?", "O prazo é 5 anos.", ["O prazo é de 5 anos."], cache, J)
        r2 = judge_phrase("q1", "q?", "O prazo é 5 anos.", ["O prazo é de 5 anos."], cache, J)
    assert r1["veredito"] == r2["veredito"] == "sustenta" and r1["model"]
    assert m.call_count == 1  # segunda leu do cache


def test_judge_phrase_nao_sustenta_e_chunk_alterado_invalida_cache(tmp_path):
    cache = JudgeCache(tmp_path / "c.jsonl")
    with patch("eval.judge.judge_codex", side_effect=['{"veredictos": [{"i": 1, "sustenta": false}]}',
                                                      '{"veredictos": [{"i": 1, "sustenta": true}]}']) as m:
        a = judge_phrase("q1", "q?", "F.", ["chunk a"], cache, J)
        b = judge_phrase("q1", "q?", "F.", ["chunk b"], cache, J)
    assert a["veredito"] == "nao_sustenta" and b["veredito"] == "sustenta"
    assert m.call_count == 2


def test_judge_phrases_uma_chamada_para_varias_frases_e_chunk_inteiro_no_prompt(tmp_path):
    cache = JudgeCache(tmp_path / "c.jsonl")
    with patch("eval.judge.judge_codex", return_value='{"veredictos": [{"i": 1, "sustenta": true}, {"i": 2, "sustenta": false}]}') as m:
        r = judge_phrases("q1", "q?", [("A.", ["chunk inteiro salvo exceto X"]), ("B.", ["outro"])], cache, J)
    assert [x["veredito"] for x in r] == ["sustenta", "nao_sustenta"]
    assert m.call_count == 1 and "salvo exceto X" in m.call_args[0][0]


def test_judge_phrase_pool_falhando_e_indeterminado_sem_cache(tmp_path):
    cache = JudgeCache(tmp_path / "c.jsonl")
    with patch("eval.judge.judge_codex", side_effect=RuntimeError("429")):
        r = judge_phrase("q1", "q?", "F.", ["c"], cache, J)
    assert r["veredito"] == "indeterminado"
    _esfriando.clear()
    with patch("eval.judge.judge_codex", return_value='{"veredictos": [{"i": 1, "sustenta": true}]}'):
        r2 = judge_phrase("q1", "q?", "F.", ["c"], cache, J)
    assert r2["veredito"] == "sustenta"  # indeterminado nunca foi cacheado


def test_judge_phrase_json_sem_veredito_por_frase_e_indeterminado(tmp_path):
    with patch("eval.judge.judge_codex", return_value='{"veredictos": []}'):
        assert judge_phrase("q1", "q?", "F.", ["c"], None, J)["veredito"] == "indeterminado"

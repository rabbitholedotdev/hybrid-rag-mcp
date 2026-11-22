from eval.judge_cache import JudgeCache


def test_get_sem_entrada_devolve_none(tmp_path):
    c = JudgeCache(tmp_path / "cache.jsonl")
    assert c.get("k1") is None


def test_set_depois_get_devolve_valor(tmp_path):
    c = JudgeCache(tmp_path / "cache.jsonl")
    c.set("k1", 0.75)
    assert c.get("k1") == 0.75


def test_persiste_entre_instancias(tmp_path):
    p = tmp_path / "cache.jsonl"
    JudgeCache(p).set("k1", 0.5)
    assert JudgeCache(p).get("k1") == 0.5


def test_key_e_deterministica_e_sensivel_a_cada_campo():
    k1 = JudgeCache.key("q1", "base", "abc123", "kimi-k3")
    k2 = JudgeCache.key("q1", "base", "abc123", "kimi-k3")
    k3 = JudgeCache.key("q1", "pools", "abc123", "kimi-k3")
    assert k1 == k2 != k3

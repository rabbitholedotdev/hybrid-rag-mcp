import json
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from rag import query
from rag.answer import format_text

DOC = SimpleNamespace(page_content="O prazo é de 5 anos.", metadata={"title": "Aula 1", "source_type": "material", "url": "http://x/1"})
GEN = json.dumps({"frases": [
    {"texto": "O prazo é de 5 anos.", "provas": [{"trecho": 1, "citacao": "prazo é de 5 anos"}]},
    {"texto": "Vale 9 anos.", "provas": [{"trecho": 1, "citacao": "vale 9 anos"}]}]})
VER = json.dumps({"itens": [{"sustenta": True, "ressalva_omitida": False}]})


def _main(monkeypatch, capsys, *args, provider="claude-cli"):
    monkeypatch.setattr(sys, "argv", ["rag.query", "pergunta", *args])
    chamadas = {}
    with patch("rag.query.retrieve", return_value=[DOC]), patch("rag.query.LLM_PROVIDER", provider), \
         patch("rag.query.generator", side_effect=lambda p: chamadas.setdefault("gen", p) and (lambda _: GEN)), \
         patch("rag.query.verifier", return_value=lambda _: VER):
        query.main()
    return capsys.readouterr().out, chamadas


def test_sem_json_imprime_format_text(monkeypatch, capsys):
    out, _ = _main(monkeypatch, capsys)
    assert "O prazo é de 5 anos. [1]" in out and "Vale 9 anos" not in out and "Provas" in out


def test_com_json_imprime_dict_com_motivo_do_corte(monkeypatch, capsys):
    out, _ = _main(monkeypatch, capsys, "--json")
    d = json.loads(out)
    assert d["status"] == "ok" and d["cortadas"][0]["motivo"] == "citacao_ausente"


def test_grok_passa_pelo_mesmo_answer(monkeypatch, capsys):
    out, ch = _main(monkeypatch, capsys, provider="grok")
    assert ch["gen"] == "grok" and "Provas" in out


def test_numbers_guard_nao_existe_mais():
    assert not hasattr(query, "numbers_guard")


def test_no_llm_mostra_so_trechos(monkeypatch, capsys):
    out, ch = _main(monkeypatch, capsys, "--no-llm")
    assert "Trechos encontrados" in out and "gen" not in ch


def test_generator_escolhe_por_provedor():
    with patch("rag.query.run_claude", return_value="x") as rc:
        assert query.generator("claude-cli")("p") == "x"
        assert rc.call_args[0][1] == "claude-haiku-5-5"
        assert query.verifier()("p") == "x" and rc.call_args[0][1] == "claude-sonnet-5-5"
    assert query.generator("grok") is query.gen_grok or callable(query.generator("grok"))
    with pytest.raises(ValueError):
        query.generator("nada")


def test_channel_e_module_chegam_ao_retrieve(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["rag.query", "p", "--channel", "c1", "--module", "Intro", "--no-llm"])
    with patch("rag.query.retrieve", return_value=[DOC]) as r:
        query.main()
    assert r.call_args.args[2] == "c1" and r.call_args.kwargs["module"] == "Intro"


def test_indice_de_course_na_lista_de_payload_indexes():
    from unittest.mock import MagicMock
    from rag.vectorstore import ensure_payload_indexes
    c = MagicMock()
    ensure_payload_indexes(c, "x")
    assert "metadata.course" in [k.kwargs["field_name"] for k in c.create_payload_index.call_args_list]

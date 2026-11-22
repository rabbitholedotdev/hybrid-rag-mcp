from unittest.mock import patch, MagicMock

import pytest

from rag import query


def _run(rc=0, out="resposta", err=""):
    return MagicMock(returncode=rc, stdout=out, stderr=err)


def test_gen_grok_usa_arquivo_e_devolve_stdout():
    with patch("rag.query.subprocess.run", return_value=_run(out="  ok  \n")) as m:
        assert query.gen_grok("pergunta") == "ok"
    argv = m.call_args[0][0]
    assert argv[0] == "grok" and "--prompt-file" in argv and "pergunta" not in argv


def test_gen_grok_falha_levanta_runtimeerror():
    with patch("rag.query.subprocess.run", return_value=_run(rc=2, out="", err="boom")):
        with pytest.raises(RuntimeError):
            query.gen_grok("p")


def test_hyde_com_grok_concatena_pergunta_e_hipotese():
    with patch("rag.query.LLM_PROVIDER", "grok"), patch("rag.query.gen_grok", return_value="hipótese de resposta"):
        assert query._hyde("pergunta?") == "pergunta?\nhipótese de resposta"


def test_hyde_com_grok_falha_volta_para_a_pergunta():
    with patch("rag.query.LLM_PROVIDER", "grok"), patch("rag.query.gen_grok", side_effect=RuntimeError("boom")):
        assert query._hyde("pergunta?") == "pergunta?"

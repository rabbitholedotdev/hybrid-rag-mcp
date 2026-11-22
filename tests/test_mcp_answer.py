import json
import threading
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from rag import mcp_server
from rag.answer import format_text
from rag.claude_cli import LLMError

DOC = SimpleNamespace(page_content="O prazo é de 5 anos, salvo fraude.",
                      metadata={"title": "Aula 1", "source_type": "material", "url": "http://x/1"})
FR = {"texto": "O prazo é de 5 anos.", "provas": [{"trecho": 1, "citacao": "prazo é de 5 anos"}]}
CUT = {"texto": "Vale 9 anos.", "provas": [{"trecho": 1, "citacao": "vale 9 anos"}]}
GEN = lambda p: json.dumps({"frases": [FR, CUT]})  # noqa: E731
VER_OK = lambda p: json.dumps({"itens": [{"sustenta": True, "ressalva_omitida": False}]})  # noqa: E731


def _call(question="q", ver=VER_OK, **kw):
    with patch.object(mcp_server, "retrieve", return_value=[DOC]), \
         patch.object(mcp_server, "generator", return_value=GEN), patch.object(mcp_server, "verifier", return_value=ver):
        return mcp_server.answer(question, **kw)


def test_texto_final_igual_a_format_text_e_sem_frase_cortada():
    r = _call()
    assert r["texto_final"].startswith(format_text({**r, "cortadas": [{}]}).split("\n")[0])
    assert "Vale 9 anos" not in r["texto_final"] and r["cortadas"] == [{"motivo": "citacao_ausente"}]
    assert "Vale 9 anos" not in json.dumps(r)


def test_cota_estourada_falha_fechada_sem_texto_do_gerador():
    def ver(p):
        raise LLMError("quota")
    r = _call(ver=ver)
    assert r["status"] == "falha_verificacao" and "O prazo é de 5 anos." not in r["texto_final"]


def test_chamadas_concorrentes_nao_se_contaminam():
    def ver(p):
        if "COTA" in p:
            raise LLMError("quota")
        return VER_OK(p)
    saidas = {}
    def run(nome, q):
        saidas[nome] = _call(q, ver=ver)["status"]
    ts = [threading.Thread(target=run, args=("a", "COTA?")), threading.Thread(target=run, args=("b", "normal?"))]
    [t.start() for t in ts]; [t.join() for t in ts]
    assert saidas == {"a": "falha_verificacao", "b": "ok"}


def test_erro_nao_vaza_token(monkeypatch):
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "sk-ant-oat01-SEGREDO")
    def ver(p):
        raise LLMError("falhou com sk-ant-oat01-SEGREDO")
    assert "SEGREDO" not in json.dumps(_call(ver=ver))


def test_colecao_diferente_da_instancia_da_erro_claro():
    r = mcp_server.answer("q", collection="outra_rag")
    assert "erro" in r and "outra_rag" in r["erro"]


def test_documentacao_lista_answer_e_o_risco():
    d = mcp_server.documentacao()
    assert "answer" in d["ferramentas"] and "risco" in json.dumps(d["ferramentas"]["answer"]).lower()

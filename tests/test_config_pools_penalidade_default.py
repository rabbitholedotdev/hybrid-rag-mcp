"""Without the env vars, pooled retrieval stays off (opt-in per instance)."""
import importlib

from rag import config


def test_pools_separados_desligado_por_padrao(monkeypatch):
    monkeypatch.delenv("RAG_POOLS_SEPARADOS", raising=False)
    monkeypatch.delenv("RAG_PENALIDADE_SINTETICO", raising=False)
    importlib.reload(config)
    assert config.RAG_POOLS_SEPARADOS is False
    assert config.RAG_PENALIDADE_SINTETICO == 0.0


def test_pools_separados_liga_por_env(monkeypatch):
    monkeypatch.setenv("RAG_POOLS_SEPARADOS", "1")
    monkeypatch.setenv("RAG_PENALIDADE_SINTETICO", "0.5")
    importlib.reload(config)
    assert config.RAG_POOLS_SEPARADOS is True
    assert config.RAG_PENALIDADE_SINTETICO == 0.5
    # não deixa o reload vazar pros próximos testes
    monkeypatch.delenv("RAG_POOLS_SEPARADOS", raising=False)
    monkeypatch.delenv("RAG_PENALIDADE_SINTETICO", raising=False)
    importlib.reload(config)

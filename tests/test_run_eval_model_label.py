"""Achado ao rodar a matriz R2 de verdade: com LLM_PROVIDER=claude e sem
ANTHROPIC_API_KEY, `_gen` cai no CLI (gen_claude_cli, modelo=CLAUDE_MODEL),
mas o `cfg["llm"]` do cabeçalho do relatório gravava OLLAMA_MODEL — a
condição do label não cobria o caminho de fallback do CLI, então o relatório
mentia sobre qual LLM gerou a resposta."""
import eval.run_eval as run_eval


def test_label_usa_claude_model_no_fallback_cli_sem_chave(monkeypatch):
    monkeypatch.setattr(run_eval, "ANTHROPIC_API_KEY", None)
    monkeypatch.setattr(run_eval, "LLM_PROVIDER", "claude")
    assert run_eval._model_label() == run_eval.CLAUDE_MODEL


def test_label_usa_claude_model_com_chave(monkeypatch):
    monkeypatch.setattr(run_eval, "ANTHROPIC_API_KEY", "sk-ant-x")
    monkeypatch.setattr(run_eval, "LLM_PROVIDER", "claude")
    assert run_eval._model_label() == run_eval.CLAUDE_MODEL


def test_label_usa_ollama_model_quando_provider_ollama(monkeypatch):
    monkeypatch.setattr(run_eval, "ANTHROPIC_API_KEY", None)
    monkeypatch.setattr(run_eval, "LLM_PROVIDER", "ollama")
    assert run_eval._model_label() == run_eval.OLLAMA_MODEL

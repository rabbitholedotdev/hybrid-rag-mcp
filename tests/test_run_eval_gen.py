from unittest.mock import patch

import eval.run_eval as run_eval


def test_gen_usa_cli_sem_chave_anthropic():
    with patch.object(run_eval, "ANTHROPIC_API_KEY", None), \
         patch.object(run_eval, "LLM_PROVIDER", "claude"), \
         patch("eval.run_eval.gen_claude_cli", return_value="resposta") as m:
        assert run_eval._gen("prompt") == "resposta"
        m.assert_called_once()


def test_gen_usa_api_key_quando_presente():
    with patch.object(run_eval, "ANTHROPIC_API_KEY", "sk-ant-x"), \
         patch.object(run_eval, "LLM_PROVIDER", "claude"), \
         patch("eval.run_eval.gen_claude", return_value="resposta") as m:
        assert run_eval._gen("prompt") == "resposta"
        m.assert_called_once()


def test_gen_usa_ollama_quando_provider_ollama():
    with patch.object(run_eval, "ANTHROPIC_API_KEY", None), \
         patch.object(run_eval, "LLM_PROVIDER", "ollama"), \
         patch("eval.run_eval.gen_ollama", return_value="resposta") as m:
        assert run_eval._gen("prompt") == "resposta"
        m.assert_called_once()

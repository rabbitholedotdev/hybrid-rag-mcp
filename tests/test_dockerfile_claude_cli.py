"""A tool `answer` roda `claude -p` no container: CLI com versão fixa, config vazia, nenhum token no arquivo."""
import re
from pathlib import Path

TXT = (Path(__file__).resolve().parent.parent / "Dockerfile").read_text(encoding="utf-8")


def test_claude_code_instalado_com_versao_fixa():
    assert re.search(r"npm install -g @anthropic-ai/claude-code@\d+\.\d+\.\d+", TXT)


def test_config_vazia_criada_no_build():
    assert "RAG_CLAUDE_HOME=" in TXT and "mkdir -p /home/rag-claude" in TXT


def test_nenhum_token_no_dockerfile():
    assert "sk-ant" not in TXT and not re.search(r"CLAUDE_CODE_OAUTH_TOKEN\s*=\s*\S", TXT)

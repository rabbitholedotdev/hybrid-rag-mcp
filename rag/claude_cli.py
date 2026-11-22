"""Gerador/verificador headless via `claude -p` (spec 2026-10-17-rag-assertive, R6).

Prompt por stdin, 1 turno, sem ferramentas e sem carregar CLAUDE.md, hooks nem plugins do Mac
O isolamento é `--setting-sources ""` + `--tools ""` (medido: ~3,7 mil tokens de cache contra ~26 mil sem). No servidor
(CLAUDE_CODE_OAUTH_TOKEN no ambiente) a pasta de config também é a vazia RAG_CLAUDE_HOME; no Mac o login mora na
pasta da conta (CLAUDE_CONFIG_DIR, escolhida pelo rabbit-claude), então ela fica como está. O erro nunca carrega o token."""
import json
import os
import shutil
import subprocess
from pathlib import Path


class LLMError(RuntimeError):
    pass


def _redact(texto: str) -> str:
    token = os.environ.get("CLAUDE_CODE_OAUTH_TOKEN")
    return texto.replace(token, "[token]") if token else texto


def run_claude(prompt: str, model: str, timeout: float) -> str:
    env = dict(os.environ)
    if env.get("CLAUDE_CODE_OAUTH_TOKEN"):  # servidor: login vem do token, config vazia
        home = Path(env.get("RAG_CLAUDE_HOME") or Path.home() / ".cache" / "rag-claude")
        home.mkdir(parents=True, exist_ok=True)
        env["CLAUDE_CONFIG_DIR"] = str(home)
    binario = env.get("RAG_CLAUDE_BIN") or shutil.which("rabbit-claude") or "claude"  # rabbit-claude escolhe a conta do pool
    argv = [binario, "-p", "--model", model, "--output-format", "json", "--strict-mcp-config",
            "--disable-slash-commands", "--no-session-persistence", "--tools", "", "--setting-sources", ""]
    try:
        r = subprocess.run(argv, input=prompt.replace("\x00", ""), capture_output=True, text=True,
                           timeout=timeout, env=env)
    except subprocess.TimeoutExpired as e:
        raise LLMError(f"claude -p estourou {timeout}s") from e
    try:
        d = json.loads(r.stdout)
    except json.JSONDecodeError as e:
        raise LLMError(_redact(f"claude -p devolveu stdout não-JSON (rc={r.returncode}): "
                               f"{r.stdout[:200]!r} stderr={r.stderr[:200]!r}")) from None
    if not isinstance(d, dict) or d.get("is_error") or not isinstance(d.get("result"), str):
        res = d.get("result") if isinstance(d, dict) else d
        raise LLMError(_redact(f"claude -p falhou: {str(res)[:300]}"))
    return d["result"].strip()

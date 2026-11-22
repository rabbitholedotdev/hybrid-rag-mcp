"""Providers de LLM por CLI pra fase 2 do eval (sem ANTHROPIC_API_KEY nem
ollama nesta máquina — ver spec-fase2.md D2). Gerador = `claude -p`, juiz =
`opencode run`. Isolado do runtime do servidor (rag/config.py não muda)."""
import json
import re
import os
import subprocess
import threading
from pathlib import Path


def strip_json_fence(text: str) -> str:
    """Remove ```json ... ``` (ou ``` ... ```) em volta do texto, se houver."""
    m = re.match(r"^```(?:json)?\s*\n(.*)\n```\s*$", text.strip(), re.S)
    return m.group(1).strip() if m else text.strip()


def parse_json_veredict(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except Exception:  # noqa: BLE001 — juiz é best-effort
        return {}


def _extract_opencode_text(stdout: str) -> str:
    """`opencode run --format json` imprime um evento JSON por linha; o texto
    final está em part.text de eventos type:"text". Concatena todos (na
    prática sempre 1) e ignora linha que não for JSON válido (log/ruído)."""
    partes = []
    for linha in stdout.splitlines():
        linha = linha.strip()
        if not linha:
            continue
        try:
            ev = json.loads(linha)
        except json.JSONDecodeError:
            continue
        if ev.get("type") == "text":
            txt = (ev.get("part") or {}).get("text")
            if txt:
                partes.append(txt)
    return "".join(partes).strip()


def gen_claude_cli(prompt: str, model: str = "claude-sonnet-5", timeout: int = 180) -> str:
    """Gerador do eval: delega a `rag.claude_cli.run_claude` (stdin, sem tools/CLAUDE.md/hooks; ver lá).
    Custo medido em 2026-09-25: ~US$0,024/chamada com `--tools "" --setting-sources ""` contra ~US$0,12-0,22 sem."""
    from rag.claude_cli import LLMError, run_claude
    try:
        return run_claude(prompt, model, timeout)
    except LLMError as e:
        raise RuntimeError(str(e)) from e


# `opencode run` free/go tier rate-limita em rajada (429 "Rate limit
# exceeded"/"Go usage limit exceeded") quando várias chamadas concorrentes
# batem no mesmo provider ao mesmo tempo — achado rodando a matriz R2 com
# --workers 6 (até 12 chamadas de juiz concorrentes). O semáforo aqui limita
# quantas chamadas a `opencode run` saem ao mesmo tempo NESTE processo,
# independente de --workers (que continua paralelo pro retrieve/geração).
# Ajustável por env pra quem tiver cota maior; 2 é conservador.
_OPENCODE_SEM = threading.Semaphore(int(os.getenv("OPENCODE_MAX_CONCURRENT", "2")))


def judge_opencode(prompt: str, model: str, timeout: int = 90) -> str:
    """Juiz headless via `opencode run -m <model>`. `model` é a referência
    completa que o `opencode models` lista (ex.: `opencode-go/kimi-k3` ou
    `opencode/mimo-v2.6-flash-free`) — NÃO prefixa mais sozinho: prefixar
    "opencode-go/" na frente de um modelo já qualificado (ou de um do plano
    free, sem esse prefixo) gera um ID que não existe e falha silencioso
    (returncode!=0, stderr vazio) — achado ao trocar de juiz na matriz R2
    quando a cota do plano Go esgotou (`provider.quota: Go usage limit
    exceeded`, 429) e os dois DEFAULT_JUDGES caíram 100% no fallback lexical
    sem erro visível no relatório (só via `judge_fallbacks` no agregado)."""
    prompt = prompt.replace("\x00", "")
    with _OPENCODE_SEM:
        r = subprocess.run(
            ["opencode", "run", "-m", model, "--format", "json", prompt],
            capture_output=True, text=True, timeout=timeout,
        )
    if r.returncode != 0:
        raise RuntimeError(f"opencode run falhou ({model}): {r.stderr[:300]}")
    return _extract_opencode_text(r.stdout)


def judge_codex(prompt: str, timeout: int = 420) -> str:
    """Juiz headless via `codex exec` (família GPT, assinatura do dono): fora das famílias do gerador (Claude)
    e do grok. Em 2026-10-17 o tier free do `opencode run` passou a dar 403 FreeTierError e o plano Go
    não tem rota, então o pool do opencode ficou sem juiz. Prompt por stdin; resposta pelo arquivo de
    última mensagem. Sem sandbox de escrita, sem persistir sessão."""
    import tempfile
    prompt = prompt.replace("\x00", "")
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "last.txt"
        r = subprocess.run(["codex", "exec", "--sandbox", "read-only", "--skip-git-repo-check", "--ephemeral",
                            "-C", d, "-o", str(out), "-"],
                           input=prompt, capture_output=True, text=True, timeout=timeout)
        if r.returncode != 0 or not out.exists():
            raise RuntimeError(f"codex exec falhou (rc={r.returncode}): {r.stderr[-300:]}")
        return out.read_text(encoding="utf-8").strip()

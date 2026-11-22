"""Cobre o que a revisão do PR #6 apontou como gap: nada testava o argv real
passado a `subprocess.run` (a flag `--tools ""`/`--setting-sources ""` que
corta o custo de ~US$0,15 pra ~US$0,024/chamada podia ser removida num
refactor sem nenhum teste quebrar) nem o tratamento de falha (stdout
não-JSON, returncode != 0) — antes, isso propagava exceção crua."""
from unittest.mock import MagicMock, patch

import pytest

from eval.llm_cli import gen_claude_cli


def _fake_run(stdout="", returncode=0, stderr=""):
    r = MagicMock()
    r.stdout = stdout
    r.returncode = returncode
    r.stderr = stderr
    return r


def test_gen_claude_cli_inclui_flags_de_custo_no_argv():
    with patch("rag.claude_cli.subprocess.run",
               return_value=_fake_run('{"is_error": false, "result": "ok"}')) as m:
        gen_claude_cli("pergunta")
    argv = m.call_args.args[0]
    assert "--tools" in argv and argv[argv.index("--tools") + 1] == ""
    assert "--setting-sources" in argv and argv[argv.index("--setting-sources") + 1] == ""


def test_gen_claude_cli_stdout_invalido_levanta_runtimeerror_nao_jsondecodeerror():
    with patch("rag.claude_cli.subprocess.run", return_value=_fake_run("não é json", returncode=1)):
        with pytest.raises(RuntimeError):
            gen_claude_cli("pergunta")


def test_gen_claude_cli_json_valido_com_is_error_levanta_runtimeerror():
    with patch("rag.claude_cli.subprocess.run",
               return_value=_fake_run('{"is_error": true, "result": "falhou"}', returncode=0)):
        with pytest.raises(RuntimeError, match="falhou"):
            gen_claude_cli("pergunta")


def test_nul_no_prompt_e_removido_do_argv():
    # achado na matriz R2: chunk com \x00 => "embedded null byte" ao montar o argv
    from eval.llm_cli import judge_opencode
    with patch("rag.claude_cli.subprocess.run",
               return_value=_fake_run('{"is_error": false, "result": "ok"}')) as m:
        gen_claude_cli("a\x00b")
    assert m.call_args.kwargs["input"] == "ab"
    with patch("eval.llm_cli.subprocess.run", return_value=_fake_run("")) as m:
        judge_opencode("a\x00b", "kimi-k3")
    assert "ab" in m.call_args.args[0]


def test_judge_opencode_limita_concorrencia_por_semaforo():
    import threading
    from eval.llm_cli import _OPENCODE_SEM, judge_opencode
    em_voo = {"n": 0, "pico": 0}
    lock = threading.Lock()

    def fake_run(*a, **kw):
        with lock:
            em_voo["n"] += 1
            em_voo["pico"] = max(em_voo["pico"], em_voo["n"])
        import time; time.sleep(0.05)
        with lock:
            em_voo["n"] -= 1
        return _fake_run('{"type":"text","part":{"text":"ok"}}')

    with patch("eval.llm_cli.subprocess.run", side_effect=fake_run):
        threads = [threading.Thread(target=judge_opencode, args=("p", "m")) for _ in range(6)]
        for t in threads: t.start()
        for t in threads: t.join()
    assert em_voo["pico"] <= _OPENCODE_SEM._value + 2  # nunca sobe sem limite (checa teto sensato)
    assert em_voo["pico"] <= 2

from eval.llm_cli import strip_json_fence, parse_json_veredict, _extract_opencode_text


def test_strip_json_fence_remove_cerca():
    assert strip_json_fence('```json\n{"ok": true}\n```') == '{"ok": true}'


def test_strip_json_fence_sem_cerca_passa_direto():
    assert strip_json_fence('{"ok": true}') == '{"ok": true}'


def test_parse_json_veredict_extrai_do_meio_do_texto():
    assert parse_json_veredict('blá {"score": 0.8} blá') == {"score": 0.8}


def test_parse_json_veredict_invalido_devolve_vazio():
    assert parse_json_veredict("não é json") == {}


def test_extract_opencode_text_pega_evento_text():
    # formato real observado em 2026-09-25 (opencode run --format json v2.0.12):
    # um evento JSON por linha, texto no campo part.text do evento type:"text".
    stdout = (
        '{"type":"step_start","part":{"type":"step-start"}}\n'
        '{"type":"text","part":{"type":"text","text":"{\\"score\\": 0.9}"}}\n'
    )
    assert _extract_opencode_text(stdout) == '{"score": 0.9}'


def test_extract_opencode_text_ignora_linha_nao_json():
    stdout = 'linha de log qualquer\n{"type":"text","part":{"type":"text","text":"ok"}}\n'
    assert _extract_opencode_text(stdout) == "ok"

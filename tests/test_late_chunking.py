from unittest.mock import MagicMock, patch
import numpy as np
from rag.late_chunking import (
    pool_spans, sentence_token_bounds, group_sentences, LateChunkEncoder,
)


def test_pool_spans_mean_and_normalizes():
    # 4 tokens, dim 2. Span (0,2) = média de [1,0] e [3,0] = [2,0] -> normaliza p/ [1,0].
    embeddings = np.array([[1.0, 0.0], [3.0, 0.0], [0.0, 5.0], [0.0, 1.0]])
    out = pool_spans(embeddings, [(0, 2), (2, 4)])
    assert len(out) == 2
    np.testing.assert_allclose(out[0], [1.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(out[1], [0.0, 1.0], atol=1e-6)
    for v in out:
        assert abs(np.linalg.norm(v) - 1.0) < 1e-6


def test_pool_spans_empty_list_returns_empty():
    embeddings = np.zeros((3, 4))
    assert pool_spans(embeddings, []) == []


def test_sentence_token_bounds_maps_char_start_to_token_index():
    # <s> em (0,0), depois 3 tokens de char, depois </s> em (0,0).
    # Sentença 2 começa no char 4 -> primeiro token que TERMINA depois do 4.
    offsets = [(0, 0), (0, 3), (4, 7), (8, 11), (0, 0)]
    assert sentence_token_bounds(offsets, [0, 4, 8]) == [1, 2, 3, 5]


def test_sentence_token_bounds_past_truncation_gets_sentinel():
    # documento truncado: offsets acabam no char 7, mas há sentença no 20.
    offsets = [(0, 0), (0, 3), (4, 7)]
    assert sentence_token_bounds(offsets, [0, 20]) == [1, 3, 3]


def test_group_sentences_packs_without_exceeding_max():
    # 3 sentenças de 100 tokens cada, max_tokens=256 -> as duas primeiras
    # cabem juntas (200), a terceira não cabe com elas (300 > 256).
    assert group_sentences([0, 100, 200, 300], max_tokens=256) == [(0, 2), (2, 3)]


def test_group_sentences_long_sentence_gets_its_own_group():
    # sentença de 400 tokens sozinha excede max_tokens mas não é cortada.
    assert group_sentences([0, 50, 450, 500], max_tokens=256) == [(0, 1), (1, 2), (2, 3)]


def test_group_sentences_empty_input():
    assert group_sentences([0], max_tokens=256) == []


def test_group_sentences_never_emits_empty_span():
    # span vazio viraria mean de array vazio = NaN, e pool_spans não pega
    # NaN (só testa norma zero). Sentença sem token nenhum não vira grupo.
    assert group_sentences([5, 5], max_tokens=256) == []


def _fake_encoder(offsets, hidden):
    fake_session = MagicMock()
    fake_session.get_inputs.return_value = []
    fake_session.run.return_value = [np.array([hidden])]
    fake_encoding = MagicMock()
    fake_encoding.ids = list(range(len(offsets)))
    fake_encoding.attention_mask = [1] * len(offsets)
    fake_encoding.offsets = offsets
    fake_tokenizer = MagicMock()
    fake_tokenizer.encode.return_value = fake_encoding
    return patch("rag.late_chunking.ort.InferenceSession", return_value=fake_session), \
        patch("rag.late_chunking.Tokenizer.from_file", return_value=fake_tokenizer), \
        patch("rag.late_chunking.hf_hub_download", return_value="/fake/path")


def test_late_chunk_encoder_token_embeddings_uses_last_hidden_state():
    p1, p2, p3 = _fake_encoder(
        [(0, 2), (2, 5), (5, 9)], np.arange(12, dtype=np.float32).reshape(3, 4)
    )
    with p1, p2, p3:
        enc = LateChunkEncoder()
        offsets, embeddings = enc.token_embeddings("abbccc")

    assert offsets == [(0, 2), (2, 5), (5, 9)]
    assert embeddings.shape == (3, 4)
    np.testing.assert_allclose(embeddings[0], [0, 1, 2, 3])


def test_encode_document_spans_come_from_the_same_encode():
    # "Ab. Cd." -> <s>, "Ab.", "Cd.", </s>. Com max_chunk_tokens=1 cada
    # sentença vira um chunk; os índices usados no pooling são os índices
    # reais da matriz, nunca somas de encodes separados.
    text = "Ab. Cd."
    offsets = [(0, 0), (0, 3), (4, 7), (0, 0)]
    hidden = np.array([[9.0, 9.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
    p1, p2, p3 = _fake_encoder(offsets, hidden)
    with p1, p2, p3:
        enc = LateChunkEncoder()
        chunks = enc.encode_document(text, max_chunk_tokens=1)

    assert [c[0] for c in chunks] == ["Ab.", "Cd."]
    assert [c[1] for c in chunks] == [0, 4]
    np.testing.assert_allclose(chunks[0][2], [1.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(chunks[1][2], [0.0, 1.0], atol=1e-6)
    for _, _, v in chunks:
        assert not np.isnan(v).any()


class _WordTokenizer:
    """Tokenizer de mentira que imita o real: <s> e </s> em (0,0), um token
    por palavra com o espaço/quebra de linha ANTES dela no offset (como o
    "▁palavra" do XLM-R). id do token = índice global (facilita os asserts)."""
    def encode(self, text):
        import re
        offs = [(0, 0)] + [(m.start(), m.end()) for m in re.finditer(r"\s*\S+", text)] + [(0, 0)]
        enc = MagicMock()
        enc.ids = list(range(len(offs)))
        enc.attention_mask = [1] * len(offs)
        enc.offsets = offs
        return enc


def _word_encoder(dim_marker=True):
    """Encoder com tokenizer de palavras. O forward devolve, pra cada token
    da janela, uma linha DERIVADA DO PRÓPRIO id do token (não uma linha
    constante repetida pra janela inteira) — assim um off-by-k na posição
    local usada pro pooling muda o vetor do chunk, em vez de sair igual por
    coincidência. Guarda o tamanho de cada janela rodada."""
    calls = []

    def run(_, feed):
        ids = feed["input_ids"][0]
        calls.append(len(ids))
        return [np.array([[[float(i), 1.0] for i in ids]])]

    session = MagicMock()
    session.get_inputs.return_value = []
    session.run.side_effect = run
    with patch("rag.late_chunking.ort.InferenceSession", return_value=session), \
            patch("rag.late_chunking.Tokenizer.from_file", return_value=_WordTokenizer()), \
            patch("rag.late_chunking.hf_hub_download", return_value="/fake/path"):
        return LateChunkEncoder(), calls


def _caption(n_words, per_line=6):
    words = [f"palavra{i}" for i in range(n_words)]
    return "\n".join(" ".join(words[i:i + per_line]) for i in range(0, n_words, per_line))


def test_encode_document_splits_unpunctuated_caption_at_newlines():
    # o vídeo inteiro é UMA "sentença" (sem pontuação): antes virava 1 chunk.
    text = _caption(200, per_line=5)
    enc, _ = _word_encoder()
    chunks = enc.encode_document(text, max_chunk_tokens=20)
    assert len(chunks) >= 10
    for chunk, pos, v in chunks:
        assert text[pos:pos + len(chunk)] == chunk  # substring exata
        assert len(chunk.split()) <= 20
        assert chunk.split()[0].startswith("palavra")  # nunca no meio de palavra
    assert chunks[0][1] == 0
    assert chunks[-1][0].endswith("palavra199")
    # cortes caem em quebra de linha (linhas de 5 palavras cabem em 20)
    assert all(text[pos - 1] == "\n" for _, pos, _ in chunks[1:])


def test_encode_document_long_line_without_newline_cuts_on_word_boundary():
    text = " ".join(f"w{i}" for i in range(100))
    enc, _ = _word_encoder()
    chunks = enc.encode_document(text, max_chunk_tokens=16)
    joined = " ".join(c for c, _, _ in chunks)
    assert joined == text  # nenhuma palavra partida, nada perdido
    assert all(len(c.split()) <= 16 for c, _, _ in chunks)


def test_encode_document_short_sentences_still_group():
    text = "Um dois. Tres quatro. Cinco seis."
    enc, _ = _word_encoder()
    chunks = enc.encode_document(text, max_chunk_tokens=256)
    assert [c[0] for c in chunks] == [text]


def test_encode_document_macro_windows_cover_tail_without_duplicates(monkeypatch):
    import rag.late_chunking as lc
    monkeypatch.setattr(lc, "MAX_TOKENS", 102)   # janela de conteúdo = 100
    monkeypatch.setattr(lc, "WINDOW_STRIDE", 60)  # overlap 40
    text = _caption(500, per_line=5)
    enc, calls = _word_encoder()
    chunks = enc.encode_document(text, max_chunk_tokens=10)

    assert len(calls) > 1 and max(calls) <= 102
    assert chunks[-1][0].endswith("palavra499")     # cauda preservada
    positions = [p for _, p, _ in chunks]
    assert positions == sorted(set(positions))      # sem duplicata, em ordem
    assert " ".join(c for c, _, _ in chunks).split() == text.split()  # cobertura total
    for chunk, pos, v in chunks:
        assert text[pos:pos + len(chunk)] == chunk
        assert v.shape == (2,) and not np.isnan(v).any()
        assert abs(np.linalg.norm(v) - 1.0) < 1e-6


def test_encode_document_span_alignment_matches_independent_pooling(monkeypatch):
    """Regressão-alvo: um off-by-k em `max(a, ws) - ws + pre` (a conversão
    de índice de token GLOBAL pra índice LOCAL da janela) produzia, na
    rodada anterior, vetores que passavam em todo teste existente porque
    nada comparava o vetor do chunk contra um pooling recomputado de forma
    independente. Aqui a linha do forward é derivada do id de cada token
    (ver _word_encoder), e o esperado é recomputado só a partir dos offsets
    do tokenizer — nunca do código sob teste. Roda em janela única e em
    macro-janela (força mais de uma janela) pra exercitar os dois caminhos."""
    import rag.late_chunking as lc
    text = _caption(600, per_line=5)
    tok = _WordTokenizer()
    enc = tok.encode(text)
    ids, offs = enc.ids, enc.offsets

    def expected(pos, length):
        ts = [t for t, (s, f) in enumerate(offs) if f > s and s < pos + length and f > pos]
        rows = np.array([[float(ids[t]), 1.0] for t in ts])
        m = rows.mean(axis=0)
        return m / np.linalg.norm(m)

    # overlap (width - stride) tem que ficar bem acima de max_chunk_tokens,
    # senão nem o algoritmo consegue garantir chunk inteiro numa janela só
    # (ver Global Constraints do plano) — (30, 20) tinha overlap=8 < chunk=10
    # e falhava por violar essa precondição, não por bug de alinhamento.
    for max_tokens, stride in ((8192, 6144), (102, 60), (40, 20)):
        monkeypatch.setattr(lc, "MAX_TOKENS", max_tokens)
        monkeypatch.setattr(lc, "WINDOW_STRIDE", stride)
        enc_obj, calls = _word_encoder()
        chunks = enc_obj.encode_document(text, max_chunk_tokens=10)
        errs = [float(np.abs(v - expected(pos, len(chunk))).max()) for chunk, pos, v in chunks]
        assert len(chunks) >= 5
        assert max(errs) < 1e-9, f"max_tokens={max_tokens}: maxerr={max(errs)}"


def test_pick_window_prefers_the_most_central_containing_window():
    from rag.late_chunking import pick_window
    windows = [(0, 100, 1), (60, 160, 1), (120, 220, 1)]
    assert pick_window(windows, 10, 20) == 0
    assert pick_window(windows, 70, 80) == 0      # em 0 e 1; centro 75: 0 (c=50) mais perto que 1 (c=110)
    assert pick_window(windows, 90, 100) == 1     # em 0 e 1; centro 95: 1 mais perto
    assert pick_window(windows, 150, 160) == 2    # em 1 e 2; centro 155: 2 (c=170) mais perto que 1 (c=110)
    assert pick_window(windows, 210, 220) == 2
    assert pick_window(windows, 55, 105) == 0     # nenhuma contém inteiro: usa a que contém o início


def test_plan_windows_single_window_when_it_fits():
    from rag.late_chunking import plan_windows
    assert plan_windows(1, 99, 100, 8192, 6144) == [(0, 100, 0)]


def test_plan_windows_overlap_and_tail():
    from rag.late_chunking import plan_windows
    w = plan_windows(1, 25001, 25002, 8192, 6144)
    assert w[0] == (1, 8191, 1) and w[1][0] == 6145
    assert w[-1][1] == 25001                        # última janela termina no fim
    assert all(w[i + 1][0] < w[i][1] for i in range(len(w) - 1))  # overlap


def test_encode_document_empty_text():
    p1, p2, p3 = _fake_encoder([(0, 0)], np.zeros((1, 2)))
    with p1, p2, p3:
        enc = LateChunkEncoder()
        assert enc.encode_document("   ") == []


def test_get_encoder_is_a_singleton():
    """Cada InferenceSession recarrega ~2.3GB de peso: na ingestão o lado
    do documento e o BgeM3Embeddings têm que dividir a mesma sessão."""
    import rag.late_chunking as lc

    p1, p2, p3 = _fake_encoder([(0, 0)], np.zeros((1, 2)))
    lc._encoder = None
    with p1, p2, p3:
        a, b = lc.get_encoder(), lc.get_encoder()
    lc._encoder = None
    assert a is b

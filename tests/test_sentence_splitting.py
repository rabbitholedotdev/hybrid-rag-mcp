from rag.sentence_splitting import split_sentences, split_sentences_with_offsets


def test_split_sentences_basic():
    text = "Primeira frase. Segunda frase! Terceira frase?"
    assert split_sentences(text) == [
        "Primeira frase.", "Segunda frase!", "Terceira frase?",
    ]


def test_split_sentences_no_terminal_punctuation_is_one_sentence():
    assert split_sentences("frase sem ponto final") == ["frase sem ponto final"]


def test_split_sentences_empty_string():
    assert split_sentences("") == []


def test_split_sentences_with_offsets_points_at_real_char_positions():
    # o offset tem que apontar pro início da sentença JÁ sem o espaço de
    # separação — é ele que vira índice de token em sentence_token_bounds.
    text = "Primeira frase. Segunda frase! Terceira frase?"
    out = split_sentences_with_offsets(text)
    assert [s for s, _ in out] == [
        "Primeira frase.", "Segunda frase!", "Terceira frase?",
    ]
    for sentence, start in out:
        assert text[start:start + len(sentence)] == sentence


def test_split_sentences_with_offsets_handles_repeated_sentences():
    # sentença repetida: offsets têm que avançar, não voltar pro 1º match
    # (o código antigo usava text.find e era vulnerável a isso).
    text = "Igual. Igual. Igual."
    starts = [c for _, c in split_sentences_with_offsets(text)]
    assert starts == [0, 7, 14]

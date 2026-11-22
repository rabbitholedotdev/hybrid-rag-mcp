"""Verificação com tokenizer e modelo BGE-M3 REAIS (cache local em
BGE_M3_LOCAL_DIR). Pulado por padrão; rode com `pytest -m slow -s`.

Existe porque os testes com mock codificam a mesma suposição do código e
já esconderam bugs reais (offsets, drift de <s>/</s>, NaN em span fora da
matriz). Aqui a suposição é o tokenizer de verdade.
"""
import random

import numpy as np
import pytest

pytestmark = pytest.mark.slow

_SYLLABLES = ["ca", "sa", "mo", "ne", "ri", "to", "la", "pe", "do", "ma", "vi", "na",
              "ção", "men", "que", "não", "com", "tra", "es", "por", "u", "a", "o"]


def make_caption(target_tokens: int, tokenizer, seed: int = 7) -> str:
    """Legenda automática sintética: sem pontuação, minúscula, quebra de
    linha a cada ~40 chars, até passar de target_tokens."""
    rng = random.Random(seed)
    lines = []
    while len(tokenizer.encode("\n".join(lines)).ids) - 2 < target_tokens:
        for _ in range(20):
            line = ""
            while len(line) < 40:
                line += ("" if not line else " ") + "".join(rng.choices(_SYLLABLES, k=rng.randint(1, 3)))
            lines.append(line)
    return "\n".join(lines)


@pytest.fixture(scope="module")
def encoder():
    from rag.late_chunking import LateChunkEncoder
    return LateChunkEncoder()


def _tokens(encoder, text):
    return len(encoder.tokenizer.encode(text).ids) - 2


def _assert_contiguous(text, chunks):
    prev_end = 0
    for chunk, pos, _ in chunks:
        assert text[pos:pos + len(chunk)] == chunk
        assert not text[prev_end:pos].strip(), "lacuna de texto entre chunks"
        # chunks vizinhos sem espaço nenhum entre eles (prev_end == pos) não
        # cairiam no check de lacuna acima — o corte de palavra só aparece
        # comparando com o char imediatamente fora do chunk.
        assert pos == 0 or text[pos - 1].isspace(), f"corte no meio de palavra no início, pos={pos}"
        end = pos + len(chunk)
        assert end == len(text) or text[end].isspace(), f"corte no meio de palavra no fim, pos={end}"
        prev_end = end
    assert not text[prev_end:].strip(), "cauda perdida"


def test_caption_5400_tokens_splits_into_many_bounded_chunks(encoder):
    text = make_caption(5400, encoder.tokenizer)
    n = _tokens(encoder, text)
    chunks = encoder.encode_document(text, max_chunk_tokens=256)
    sizes = [_tokens(encoder, c) for c, _, _ in chunks]
    print(f"\n[5400] tokens={n} chunks={len(chunks)} max={max(sizes)} min={min(sizes)}")
    assert n >= 5400 and len(chunks) >= 15
    assert max(sizes) <= 256 + 2  # +2: re-tokenizar o chunk sozinho põe o ▁ inicial
    _assert_contiguous(text, chunks)
    for _, _, v in chunks:
        assert v.shape == (1024,) and not np.isnan(v).any()
        assert abs(np.linalg.norm(v) - 1.0) < 1e-4


def test_caption_25k_tokens_keeps_the_tail(encoder, monkeypatch):
    # Janela de 4096 (stride 3072) em vez da nativa de 8192: o forward de 8192
    # tokens em CPU estoura a RAM desta máquina (swap, nunca termina). O
    # caminho de macro-janelas é o mesmo; o plano em 8192/6144 é coberto por
    # test_default_window_plan_on_25k_tokens (tokenizer real, forward falso).
    import rag.late_chunking as lc
    monkeypatch.setattr(lc, "MAX_TOKENS", 4096)
    monkeypatch.setattr(lc, "WINDOW_STRIDE", 3072)
    text = make_caption(25000, encoder.tokenizer)
    n = _tokens(encoder, text)
    chunks = encoder.encode_document(text, max_chunk_tokens=256)
    print(f"\n[25k] tokens={n} chunks={len(chunks)} tokens/256={n / 256:.1f}")
    assert n >= 25000
    assert chunks[-1][1] + len(chunks[-1][0]) == len(text.rstrip())
    _assert_contiguous(text, chunks)
    assert abs(len(chunks) - n / 256) / (n / 256) < 0.35
    positions = [p for _, p, _ in chunks]
    assert positions == sorted(set(positions))
    for _, _, v in chunks:
        assert v.shape == (1024,) and not np.isnan(v).any()
        assert abs(np.linalg.norm(v) - 1.0) < 1e-4


_TOPICS = {
    "culinaria": "hoje eu vou ensinar a fazer um bolo de chocolate bem fofinho primeiro misture a farinha o açúcar e o cacau depois adicione os ovos o leite e a manteiga derretida leve ao forno por quarenta minutos e cubra com calda de brigadeiro quente a receita rende doze fatias e fica ótima com café",
    "futebol": "o time entrou em campo com o técnico apostando na defesa o atacante recebeu o passe dentro da área driblou o zagueiro e chutou forte no canto do goleiro no segundo tempo o juiz marcou pênalti e o campeonato ficou empatado a torcida lotou o estádio e cantou até o apito final",
    "programacao": "para criar uma api eu escrevo uma função que recebe a requisição valida o token consulta o banco de dados e devolve o json depois eu rodo os testes automatizados faço o deploy no servidor e configuro o container docker com as variáveis de ambiente e o monitoramento de logs",
}


def _topic_caption(sentence: str, reps: int) -> str:
    words = (sentence + " ") * reps
    words = words.split()
    return "\n".join(" ".join(words[i:i + 7]) for i in range(0, len(words), 7))


def test_single_chunk_equals_whole_document_embedding(encoder):
    """Prova exata do caso de 1 janela / 1 chunk: com um chunk só cobrindo o
    documento inteiro, o vetor late-chunked tem que ser o embed_whole (mesmo
    forward, mesmo pooling, span = (head, tail) = documento inteiro). Não
    testa alinhamento fino de span (esse span já é quase o documento
    inteiro — um desvio de poucos tokens no início/fim mal move o cosseno
    aqui); isso é o que test_span_alignment_matches_independent_pooling
    cobre, com uma métrica sensível a desvio de token."""
    text = _topic_caption(_TOPICS["culinaria"], 9)
    chunks = encoder.encode_document(text, max_chunk_tokens=4000)
    assert len(chunks) == 1 and chunks[0][0] == text.strip()
    cos = float(chunks[0][2] @ encoder.embed_whole(text))
    print(f"\n[1 chunk] cosseno vs embed_whole = {cos:.6f}")
    assert cos > 0.999


def test_late_chunk_vector_vs_chunk_embedded_alone(encoder):
    """So MEDE e imprime: cosseno entre o vetor late-chunked (com contexto
    do documento) e o mesmo texto embedado sozinho. Mean pooling do BGE-M3 é
    anisotrópico (medido: os 3 parágrafos de assunto diferente abaixo,
    embedados INTEIROS e sozinhos, já dão cosine ~0.70 entre si), então não
    há threshold discriminativo aqui — só um piso frouxo pra pegar vetor
    lixo. A prova de correção de verdade é
    test_single_chunk_equals_whole_document_embedding (exata) e
    test_span_alignment_matches_independent_pooling (discriminativa)."""
    parts = [_topic_caption(t, 9) for t in _TOPICS.values()]
    for name, text in (("3 assuntos", "\n".join(parts)),
                       ("ruido 2000", make_caption(2000, encoder.tokenizer, seed=11))):
        chunks = encoder.encode_document(text, max_chunk_tokens=256)
        alone = [encoder.embed_whole(c) for c, _, _ in chunks]
        own = [float(v @ a) for (_, _, v), a in zip(chunks, alone)]
        print(f"\n[{name}] chunks={len(chunks)} cosine vs chunk sozinho: "
              f"min={min(own):.3f} mean={np.mean(own):.3f} max={max(own):.3f}")
        assert len(chunks) >= 6 and np.mean(own) > 0.5


def test_span_alignment_matches_independent_pooling(encoder, monkeypatch):
    """Regressão-alvo do bug da rodada anterior (token-index vs
    sentence-index confundidos, só achado por verificação manual contra o
    modelo real). O fake `_run` aqui devolve, pra cada token, uma linha
    DERIVADA DO PRÓPRIO id do token (não uma linha constante repetida pra
    janela/documento inteiro como nos outros testes reais) — assim um
    off-by-k na conversão índice-global -> índice-local-da-janela muda o
    vetor pooled do chunk, em vez de sair igual por coincidência.

    O esperado é recomputado de forma totalmente independente do código sob
    teste: só a partir dos offsets do tokenizer REAL (quais tokens
    sobrepõem o char-range do chunk) e da mesma fórmula de linha por token.
    Roda com max_tokens pequeno (2048, 512 — com max_chunk_tokens reduzido
    junto, pra manter a folga overlap >> chunk que o algoritmo exige) além
    do tamanho cheio do modelo (8192) pra forçar várias janelas e exercitar de
    verdade a matemática de fronteira de janela — não só o caminho de
    janela única."""
    import rag.late_chunking as lc
    tok = encoder.tokenizer

    def fake_run(ids):
        return np.array([[float(i), 1.0, float(i) ** 2 / 1e5] for i in ids])

    monkeypatch.setattr(encoder, "_run", fake_run)
    text = make_caption(25000, tok, seed=3)
    enc = tok.encode(text)
    ids, offs = enc.ids, enc.offsets

    def expected(pos, length):
        ts = [t for t, (s, f) in enumerate(offs) if f > s and s < pos + length and f > pos]
        rows = np.array([[float(ids[t]), 1.0, float(ids[t]) ** 2 / 1e5] for t in ts])
        m = rows.mean(axis=0)
        return m / np.linalg.norm(m)

    # overlap (width - stride, ~max_tokens/4) tem que ficar bem acima de
    # max_chunk_tokens -- é a precondição do algoritmo (todo chunk cabe
    # inteiro em alguma janela, ver Global Constraints do plano), não algo
    # que este teste deva violar. max_chunk_tokens cai junto com max_tokens
    # pra manter essa folga em todos os tamanhos testados.
    for max_tokens, max_chunk_tokens in ((8192, 256), (2048, 64), (512, 16)):
        monkeypatch.setattr(lc, "MAX_TOKENS", max_tokens)
        monkeypatch.setattr(lc, "WINDOW_STRIDE", max_tokens * 3 // 4)
        chunks = encoder.encode_document(text, max_chunk_tokens=max_chunk_tokens)
        errs = [float(np.abs(v - expected(pos, len(chunk))).max()) for chunk, pos, v in chunks]
        print(f"\n[align max_tokens={max_tokens}] chunks={len(chunks)} maxerr={max(errs):.2e}")
        assert len(chunks) > 15
        assert max(errs) < 1e-5, f"max_tokens={max_tokens}: maxerr={max(errs)}"
        _assert_contiguous(text, chunks)


def test_default_window_plan_on_25k_tokens(encoder, monkeypatch):
    """Tokenizer REAL, forward FALSO (vetor = [1, id do 1º token da janela / 1e5]):
    valida o plano de janelas no tamanho cheio do modelo (8192/6144) sem pagar o
    forward de 8192 tokens em CPU."""
    import rag.late_chunking as lc
    assert (lc.MAX_TOKENS, lc.WINDOW_STRIDE) == (8192, 6144)
    seen = []

    def fake_run(ids):
        seen.append(len(ids))
        assert len(ids) <= 8192
        return np.tile(np.array([[1.0, ids[1] / 1e5]]), (len(ids), 1))

    monkeypatch.setattr(encoder, "_run", fake_run)
    text = make_caption(25000, encoder.tokenizer)
    chunks = encoder.encode_document(text, max_chunk_tokens=256)
    print(f"\n[25k/8192] chunks={len(chunks)} janelas={len(seen)} tamanhos={seen}")
    assert len(seen) == 4
    _assert_contiguous(text, chunks)
    assert chunks[-1][1] + len(chunks[-1][0]) == len(text.rstrip())
    # chunks de janelas diferentes: vetores diferem por janela
    assert len({round(float(v[1]), 4) for _, _, v in chunks}) == 4

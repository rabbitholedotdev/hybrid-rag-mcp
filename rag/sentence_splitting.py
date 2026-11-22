"""Quebra texto em sentenças, guardando o offset de char de cada uma.

O offset é o que liga a sentença ao índice de token real do documento
(ver rag/late_chunking.sentence_token_bounds). NÃO contamos token por
sentença aqui: `tokenizer.encode(sentença)` adiciona <s>/</s> em CADA
chamada, e somar essas contagens não bate com o índice da matriz de
embeddings, que vem de UMA encode() do documento inteiro.
"""
import re

_SENTENCE_END = re.compile(r"[^.!?]*[.!?]+|[^.!?]+$")


def split_sentences_with_offsets(text: str) -> list[tuple[str, int]]:
    """(sentença, offset de char da sentença no texto original)."""
    out = []
    for m in _SENTENCE_END.finditer(text):
        raw = m.group()
        stripped = raw.strip()
        if stripped:
            out.append((stripped, m.start() + len(raw) - len(raw.lstrip())))
    return out


def split_sentences(text: str) -> list[str]:
    return [s for s, _ in split_sentences_with_offsets(text)]

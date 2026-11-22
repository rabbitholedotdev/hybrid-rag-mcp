"""Decomposição de consulta: pergunta composta -> sub-perguntas, uma busca por faceta.

Motivo: a busca única de uma pergunta com várias facetas ("diferença entre X e Y e os
requisitos de Z") puxa o material de conceito e perde o trecho da lista. Cada sub-pergunta
tem o vocabulário de uma faceta só.
"""
import json
import math
import re

DECOMPOSE_PROMPT = """\
Você prepara consultas para uma busca em material de estudo (aulas e apostilas de Direito).
Se a pergunta tiver mais de uma faceta (ex.: uma diferença E uma lista de requisitos), quebre-a em
sub-perguntas, uma por faceta, cada uma completa e com o assunto escrito por extenso (sem pronomes
como "isso" ou "eles"). Se a pergunta tiver uma faceta só, devolva a própria pergunta como único item.
No máximo {max_n} sub-perguntas. Responda SOMENTE com JSON: {{"sub": ["...", "..."]}}

Pergunta: {question}
"""


def _parse(raw: str, max_n: int) -> list[str]:
    m = re.search(r"\{.*\}", raw or "", re.S)
    if not m:
        return []
    try:
        subs = json.loads(m.group(0)).get("sub", [])
    except (json.JSONDecodeError, AttributeError):
        return []
    if not isinstance(subs, list):
        return []
    out, vistos = [], set()
    for s in subs:
        if not isinstance(s, str):
            continue
        s = s.strip()
        chave = re.sub(r"\s+", " ", s.lower())
        if s and chave not in vistos:
            vistos.add(chave)
            out.append(s)
    return out[:max_n]


def decompose(question: str, max_n: int = 4, gen=None) -> list[str]:
    """Sub-perguntas da pergunta. Best-effort: sem LLM, erro ou JSON inválido devolve [question].
    `gen(prompt) -> str` troca o gerador (testes, cache); padrão = gen_grok."""
    if gen is None:
        from rag.query import gen_grok as gen
    try:
        subs = _parse(gen(DECOMPOSE_PROMPT.format(question=question, max_n=max_n)), max_n)
    except Exception:  # noqa: BLE001 — decomposição é best-effort, como o HyDE
        subs = []
    return subs or [question]


def intercalar(listas: list[list], k: int) -> list:
    """Round-robin entre as listas (uma por sub-pergunta), sem repetir o mesmo trecho
    (video_id, chunk). Garante vaga para cada faceta; o top-k de cada lista já vem rerankeado."""
    out, vistos, i = [], set(), 0
    while len(out) < k and any(i < len(l) for l in listas):
        for l in listas:
            if i < len(l):
                d = l[i]
                chave = (d.metadata.get("video_id"), d.metadata.get("chunk"))
                if chave not in vistos:
                    vistos.add(chave)
                    out.append(d)
                    if len(out) == k:
                        break
        i += 1
    return out


def retrieve_decomposto(question: str, k: int, retrieve_fn=None, gen=None, max_n: int = 4, **kw):
    """Uma busca por sub-pergunta e junção round-robin. Devolve (docs, subs).
    Pergunta de 1 faceta cai na busca normal (sem custo além da chamada de decomposição)."""
    if retrieve_fn is None:
        from rag.query import retrieve as retrieve_fn
    subs = decompose(question, max_n=max_n, gen=gen)
    if len(subs) == 1:
        return retrieve_fn(subs[0], k, **kw), subs
    k_sub = max(2, math.ceil(k / len(subs)) + 1)
    listas = [retrieve_fn(s, k_sub, **kw) for s in subs]
    return intercalar(listas, k), subs

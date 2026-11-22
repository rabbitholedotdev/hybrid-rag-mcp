"""Métricas do eval — retrieval + segurança de citação. Puro-python, sem modelo.

`docs` é a lista devolvida por `rag.query.retrieve` (objetos com `.page_content` e
`.metadata`). Fonte PRIMÁRIA (citável) = source_type NÃO em SINTETICO.
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rag.query import SINTETICO  # noqa: E402


def is_citavel(doc) -> bool:
    return (doc.metadata.get("source_type") or "").lower() not in SINTETICO


def _vid(doc) -> str:
    return doc.metadata.get("video_id") or ""


def primary_recall_at_k(docs, gold_video_id: str) -> float:
    """1.0 se a aula-fonte (gold) apareceu entre os resultados PRIMÁRIOS; senão 0.0.
    (recall binário por aula — simples e robusto p/ o baseline.)"""
    if not gold_video_id:
        return 0.0
    return 1.0 if any(is_citavel(d) and _vid(d) == gold_video_id for d in docs) else 0.0


def context_precision(docs, gold_video_id: str) -> float:
    """Fração do top-k que é primária E da aula-fonte."""
    if not docs:
        return 0.0
    rel = sum(1 for d in docs if is_citavel(d) and _vid(d) == gold_video_id)
    return round(rel / len(docs), 3)


def synthetic_share(docs) -> float:
    """Fração do top-k que é camada sintética (resumo/qa). Alto = mais risco de citar IA."""
    if not docs:
        return 0.0
    return round(sum(1 for d in docs if not is_citavel(d)) / len(docs), 3)


from rag.numeros import numeros_nao_encontrados  # noqa: E402

_NUM = re.compile(r"\d[\d.,]*\s*%?")


def numbers_hallucination(answer: str, primary_context: str) -> dict:
    """Números/percentuais afirmados na resposta que NÃO estão na fonte primária.
    Métrica (rate + lista); o aviso do antigo numbers_guard saiu, a checagem de número agora é rag.proof."""
    ans = re.split(r"\n⚠️", answer)[0]  # ignora avisos anexados
    nums = [n.strip() for n in _NUM.findall(ans)]
    fora = numeros_nao_encontrados(ans, primary_context)
    return {
        "rate": round(len(fora) / len(nums), 3) if nums else 0.0,
        "n_total": len(nums),
        "fora": fora,
    }


_ABST = re.compile(
    r"não\s+(encontr|verific|const|localiz|h[áa]\s+(essa|informa))|"
    r"não\s+.*transcri|sem\s+.*na\s+transcri|não\s+é\s+possível\s+confirmar",
    re.I,
)


def abstained(answer: str) -> bool:
    """Heurística: a resposta se absteve / marcou 'não verificado'? (p/ perguntas-armadilha)"""
    return bool(_ABST.search(answer or ""))


_SYN_PREFIX = re.compile(r"^syn-(qa|resumo)-")


def lesson_recall_at_k(docs, gold_video_id: str) -> float:
    """1.0 se QUALQUER doc (primário ou sintético) é da aula-gold. Os registros
    sintéticos têm video_id `syn-qa-<vid>`/`syn-resumo-<vid>`; o prefixo é removido.
    Mede 'achou a aula', não 'achou a transcrição' (esse é o primary_recall)."""
    if not gold_video_id:
        return 0.0
    return 1.0 if any(_SYN_PREFIX.sub("", _vid(d)) == gold_video_id for d in docs) else 0.0


def _shingles(text: str, n: int = 6) -> set:
    w = re.findall(r"\w+", text.lower())
    return {tuple(w[i:i + n]) for i in range(len(w) - n + 1)}


def content_recall_at_k(docs, gold_texts: list[str], threshold: float = 0.5) -> float:
    """1.0 se algum doc PRIMÁRIO do top-k tem >= `threshold` dos seus 6-gramas de
    palavras dentro do texto da aula-gold. Casa por conteúdo, então cobre a mesma
    aula republicada sob outro video_id — o dedup por
    conteúdo do retrieve() pode ter deixado só essa cópia no top-k."""
    gold = set().union(*map(_shingles, gold_texts)) if gold_texts else set()
    for d in docs:
        if not is_citavel(d):
            continue
        s = _shingles(d.page_content)
        if s and len(s & gold) / len(s) >= threshold:
            return 1.0
    return 0.0


def _sem_acento(s: str) -> str:
    import unicodedata
    return "".join(c for c in unicodedata.normalize("NFKD", s.lower()) if not unicodedata.combining(c))


def facet_coverage(docs, keywords: list[str]) -> float | None:
    """Fração das facetas esperadas presentes no texto dos trechos PRIMÁRIOS recuperados.
    Cada item de `keywords` é uma faceta; "a|b" aceita qualquer das alternativas. Mede a falha
    da pergunta composta: um trecho da faceta certa chegou ao contexto? None sem facetas."""
    if not keywords:
        return None
    texto = _sem_acento(" ".join(d.page_content for d in docs if is_citavel(d)))
    achou = sum(1 for k in keywords if any(_sem_acento(alt.strip()) in texto for alt in k.split("|")))
    return round(achou / len(keywords), 3)


_TOKEN = re.compile(r"[a-z]{7,}")
# palavras de moldura da resposta (citação, aviso de verificação) que não são afirmação de conteúdo
_MOLDURA = set("material transcricao transcricoes trechos trecho verificado verificada verificados fontes timestamp "
               "resumos resposta pergunta campus produtos conforme segundo apenas somente mesmo ainda "
               "porque portanto entretanto inclusive ".split())


def _tokens(texto: str) -> set:
    return set(_TOKEN.findall(_sem_acento(texto)))


def _limpa_resposta(answer: str) -> str:
    """Tira avisos (⚠️) e os links (não são afirmação de conteúdo)."""
    answer = answer.split("⚠️")[0]
    return re.sub(r"\(?https?://\S+\)?", " ", answer)


def answer_facet_coverage(answer: str, keywords: list[str]) -> float | None:
    """Fração das facetas esperadas que aparecem na RESPOSTA gerada (a métrica que o aluno vê)."""
    if not keywords:
        return None
    texto = _sem_acento(_limpa_resposta(answer))
    achou = sum(1 for k in keywords if any(_sem_acento(alt.strip()) in texto for alt in k.split("|")))
    return round(achou / len(keywords), 3)


def termos_sem_base(answer: str, ctx_docs, question: str = "") -> list[str]:
    """Palavras (>= 7 letras) da resposta que não estão em nenhum trecho PRIMÁRIO do contexto, nem na pergunta,
    nem na moldura da resposta. Proxy lexical de afirmação sem base: inclui o que veio só de resumo de IA e
    o que o modelo trouxe da própria memória. Compara configurações entre si; o valor absoluto tem ruído de paráfrase."""
    prim = _tokens(" ".join(d.page_content for d in ctx_docs if is_citavel(d)))
    base = prim | _tokens(question) | _MOLDURA
    return sorted(_tokens(_limpa_resposta(answer)) - base)


def unsupported_rate(answer: str, ctx_docs, question: str = "") -> float:
    toks = _tokens(_limpa_resposta(answer)) - _MOLDURA
    return round(len(termos_sem_base(answer, ctx_docs, question)) / len(toks), 3) if toks else 0.0


def termos_so_sintetico(answer: str, ctx_docs, question: str = "") -> list[str]:
    """Subconjunto de `termos_sem_base` que existe no texto sintético (resumo/Q&A) do contexto: vazamento da
    camada gerada por IA para a resposta (ex.: um nome de teoria que só o resumo traz)."""
    sint = _tokens(" ".join(d.page_content for d in ctx_docs if not is_citavel(d)))
    return [t for t in termos_sem_base(answer, ctx_docs, question) if t in sint]

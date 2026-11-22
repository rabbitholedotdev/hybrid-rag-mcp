"""Facetas (`expected_keywords`) for the funnel (R5).

The LLM reads the question and the gold lesson and proposes 3–6 short facets;
only those that appear literally (unaccented) in the lesson text are kept.
Appends `eval/facetas.jsonl` and resumes by skipping ids already written.

    COLLECTION_NAME=legal_rag python -m eval.build_facetas [--limite N] [--workers 4]"""
import argparse
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

from eval.funil_dados import FACETAS, _facetas, todas_funil
from eval.llm_cli import gen_claude_cli, strip_json_fence
from eval.metrics import _sem_acento

MODELO = "claude-sonnet-5-5"
MAX_TEXTO = 30000

PROMPT = """Abaixo estão uma pergunta e a transcrição da aula que a responde.
Liste de 3 a 6 facetas da resposta: termos ou expressões curtas (1 a 4 palavras) que uma resposta
completa precisa conter. Copie cada faceta LITERALMENTE do texto da aula, com a mesma grafia.
Prefira termos específicos da resposta (nomes, números, conceitos) a palavras genéricas.
Responda só com um array JSON de strings.

Pergunta: {pergunta}

Aula:
{texto}"""


def filtra_facetas(facetas: list[str], texto: str) -> list[str]:
    alvo = _sem_acento(texto)
    out = []
    for f in facetas:
        f = (f or "").strip()
        if f and _sem_acento(f) in alvo and f not in out:
            out.append(f)
    return out


def texto_aula(expected_ids: list[str]) -> str:
    from rag.query import get_lesson_chunks
    for vid in expected_ids:
        chunks = get_lesson_chunks(vid)
        if chunks:
            return " ".join(c["text"] for c in chunks)
    return ""


def facetas_de(row: dict) -> list[str]:
    texto = texto_aula(row["expected_ids"])
    if not texto:
        return []
    resp = gen_claude_cli(PROMPT.format(pergunta=row["pergunta"], texto=texto[:MAX_TEXTO]), model=MODELO)
    ini = resp.find("[")
    if ini < 0:
        raise ValueError(f"sem array JSON na resposta: {resp[:100]!r}")
    propostas, _ = json.JSONDecoder().raw_decode(resp[ini:])  # ignora texto depois do array
    return filtra_facetas([str(p) for p in propostas], texto)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limite", type=int)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    if os.getenv("COLLECTION_NAME") != "legal_rag":
        sys.exit("COLLECTION_NAME precisa ser legal_rag")
    feitos = _facetas()
    rows = [r for r in todas_funil() if r["colecao"] == "legal_rag" and r["id"] not in feitos][:a.limite]
    trava, falhas = threading.Lock(), []

    def um(row):
        try:
            fac = facetas_de(row)
        except Exception as e:  # noqa: BLE001 — falha não grava: a próxima rodada tenta de novo
            falhas.append(row["id"])
            print(f"{row['id']}: falhou: {str(e)[:200]}", file=sys.stderr, flush=True)
            return
        with trava, FACETAS.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"id": row["id"], "expected_keywords": fac}, ensure_ascii=False) + "\n")
        print(f"{row['id']}: {len(fac)} facetas", flush=True)

    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        list(ex.map(um, rows))
    todos = _facetas()
    poucas = sum(1 for v in todos.values() if len(v) < 3)
    print(f"gravadas {len(todos)}; com menos de 3 facetas: {poucas}; falhas nesta rodada: {len(falhas)}")


if __name__ == "__main__":
    main()

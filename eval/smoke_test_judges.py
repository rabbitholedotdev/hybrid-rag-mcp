"""Smoke test de calibração dos juízes (spec-fase2.md R1): roda os modelos de
`DEFAULT_JUDGES` contra pares (contexto, resposta, esperado) com veredito
conhecido. Rode uma vez antes de usar os juízes em R2/R3; troque qualquer
modelo com acerto < 0.7.

Uso: cd rag-mcp && venv/bin/python -m eval.smoke_test_judges
"""
import time

from eval.judge import DEFAULT_JUDGES, _judge_one

CASOS = [
    # (contexto, resposta, esperado: True=sustentada/score alto, False=score baixo)
    ("O curso tem 8 módulos e dura 6 semanas.", "O curso tem 8 módulos.", True),
    ("O curso tem 8 módulos e dura 6 semanas.", "O curso tem 12 módulos e dura 1 ano.", False),
    ("O framework se chama AIDA: atenção, interesse, desejo, ação.", "O framework é o AIDA.", True),
    ("O framework se chama AIDA.", "O framework se chama PASTOR.", False),
    ("A conversão subiu de 2% para 5% depois da mudança.", "A conversão foi de 2% para 5%.", True),
    ("A conversão subiu de 2% para 5%.", "A conversão subiu para 40%.", False),
    ("O autor recomenda testar por 30 dias antes de decidir.", "Recomenda-se testar por 30 dias.", True),
    ("O autor recomenda testar por 30 dias.", "O autor recomenda decidir no primeiro dia.", False),
    ("A ferramenta citada é o ClickFunnels.", "A ferramenta usada é o ClickFunnels.", True),
    ("A ferramenta citada é o ClickFunnels.", "A ferramenta usada é o HubSpot.", False),
]


def main():
    for provider, model in DEFAULT_JUDGES:
        acertos = 0
        for ctx, resp, esperado_alto in CASOS:
            try:
                score = _judge_one(provider, model, resp, ctx)
            except Exception as e:  # noqa: BLE001
                print(f"  ERRO {model}: {e}")
                continue
            finally:
                time.sleep(1)
            ok = (score >= 0.6) == esperado_alto
            acertos += ok
            print(f"  {model} score={score:.2f} esperado_alto={esperado_alto} "
                  f"{'OK' if ok else 'ERRADO'}: {resp[:50]!r}")
        taxa = acertos / len(CASOS)
        print(f"{model}: {acertos}/{len(CASOS)} = {taxa:.0%}"
              f"{'  <<< TROCAR MODELO' if taxa < 0.7 else ''}")


if __name__ == "__main__":
    main()

"""Números da resposta que não aparecem na fonte (guarda anti-invenção), sem falso positivo de formato."""
import re

_NUM = re.compile(r"\d[\d.,]*\s*%?")


def _sem_sep(s: str) -> str:
    return re.sub(r"\s+", "", s).replace(".", "").replace(",", "")


def numeros_nao_encontrados(answer: str, fonte: str) -> list[str]:
    """Números da resposta ausentes em `fonte`. Não conta como ausente:
    - pontuação final colada ("art. 312,", "19.12.2003.");
    - separador de milhar ou de data diferente ("9.868" contra "9868");
    - ano de 4 dígitos que a fonte traz abreviado ("1999" contra "9.868/99")."""
    alvo = _sem_sep(fonte)
    bruto = re.sub(r"\s+", "", fonte)
    achados: dict[str, None] = {}
    for n in _NUM.findall(answer):
        tok = re.sub(r"[.,]+$", "", n.strip())
        digitos = _sem_sep(tok)
        if not digitos:
            continue
        if digitos in alvo:
            continue
        if re.fullmatch(r"(19|20)\d\d", digitos) and f"/{digitos[2:]}" in bruto:
            continue
        achados[tok] = None
    return list(achados)

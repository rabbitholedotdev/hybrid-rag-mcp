"""Funil da recuperação (R5, spec 2026-10-20-rag-recall): em que degrau o trecho certo morre.

Puro: listas de docs (`.page_content`, `.metadata`) e dicts, sem Qdrant. Degraus:
0 aula-ouro sem evidência · 1 evidência partida entre chunks · 2 fora do pool ·
3 caiu no rerank · 4 corte do MCP (a CLI achou) · 5 geração ignorou · `viva`."""
from collections import Counter, defaultdict

from eval.metrics import facet_coverage

# 0.6: o degrau 0 deu 0% em todo L de 0.4 a 0.8 (tags base-<colecao>-calibracao), então o critério não
# discrimina; 0.6 evita o artefato de L alto (pergunta composta nunca cabe num chunk) no degrau 1.
L_PADRAO: float = 0.6

DEGRAUS = ("0", "1", "2", "3", "4", "5", "viva")
MARCAS_PRIMEIRA_BUSCA = ("denso", "esparso", "nenhum", "sintetico_na_vaga")


def cobertura_chunk(doc, keywords) -> float:
    return facet_coverage([doc], keywords) or 0.0


def melhor_cobertura(docs, keywords) -> float:
    """A maior cobertura de um chunk isolado (sintético conta 0)."""
    return max((cobertura_chunk(d, keywords) for d in docs), default=0.0)


def evidencia(docs, keywords, L) -> bool:
    return melhor_cobertura(docs, keywords) >= L


def diagnostico_aula(chunks, keywords, L) -> dict:
    ordem = sorted(chunks, key=lambda d: d.metadata.get("chunk") or 0)
    pares = zip(ordem, ordem[1:])
    return {"n_chunks": len(chunks),
            "uniao": facet_coverage(chunks, keywords) or 0.0,
            "melhor_chunk": melhor_cobertura(chunks, keywords),
            "melhor_janela2": max((facet_coverage([a, b], keywords) or 0.0 for a, b in pares), default=0.0)}


def classificar(vivo: dict, tem_ouro: bool, caminho: str, cli_final_vivo: bool | None = None,
                resposta_cobre: bool | None = None) -> str:
    if caminho == "mcp" and cli_final_vivo and not vivo["final"]:
        return "4"
    if not vivo["pool"]:
        if tem_ouro and not vivo["aula"]:
            return "0"
        if tem_ouro and not vivo["chunk"]:
            return "1"
        return "2"
    if not vivo["final"]:
        return "3"
    if resposta_cobre is False:
        return "5"
    return "viva"


def marcas(degrau, vivo, tem_ouro, trace_vivo: dict, n_sintetico_fusao: int, n_chunks_ouro) -> list[str]:
    """`n_chunks_ouro` é None quando a pergunta não tem `expected_ids`; 0 é aula-ouro não achada."""
    m = []
    if not tem_ouro:
        m.append("sem_ouro")
    if n_chunks_ouro == 0:
        m.append("ouro_vazio")
    if tem_ouro and vivo["pool"] and not vivo["chunk"]:
        m.append("outra_aula")
    if degrau == "2":
        achou = [modo for modo in ("denso", "esparso") if trace_vivo.get(modo)]
        m += achou or ([] if trace_vivo.get("fusao") else ["nenhum"])
        if n_sintetico_fusao >= 1:
            m.append("sintetico_na_vaga")
    return m


def resumo(itens: list[dict]) -> dict:
    """Contagens por `colecao|golden|caminho`. A soma dos degraus é `n` menos as `sem_facetas`."""
    grupos = defaultdict(list)
    for it in itens:
        grupos[f'{it["colecao"]}|{it["golden"]}|{it["caminho"]}'].append(it)
    out = {}
    for chave, its in sorted(grupos.items()):
        degraus = Counter(it["degrau"] for it in its if it["degrau"] is not None)
        mc = Counter(m for it in its for m in it["marcas"])
        assert sum(degraus.values()) == len(its) - mc["sem_facetas"], chave
        out[chave] = {"n": len(its), "degraus": {d: degraus.get(d, 0) for d in DEGRAUS}, "marcas": dict(mc)}
    return out


# ---------- regra de aceite da fase 2 (spec, «Regra de aceite») ----------

LAT_OK, LAT_DONO, GANHO_MIN, GANHO_DONO, PERDA_MAX = 1.10, 1.25, 3, 6, 1


def mcnemar(b: int, c: int) -> float:
    """p bilateral exato do teste de McNemar pareado (b e c são os discordantes)."""
    from math import comb
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def _vivas(itens, caminho) -> dict[str, bool]:
    return {i["id"]: i["degrau"] == "viva" for i in itens if i["caminho"] == caminho and i["degrau"] is not None}


def _bc(base_itens, novo_itens, caminho) -> tuple[int, int]:
    vb, vn = _vivas(base_itens, caminho), _vivas(novo_itens, caminho)
    comuns = vb.keys() & vn.keys()
    return sum(1 for q in comuns if vn[q] and not vb[q]), sum(1 for q in comuns if vb[q] and not vn[q])


def _taxa_sem_base(gate) -> float:
    fr = [f for it in gate["items"] for f in it.get("frases", [])]
    return sum(1 for f in fr if f["veredito"] == "nao_sustenta") / len(fr) if fr else 0.0


def aceite(base: dict, novo: dict, visado: str, gate_base: dict | None, gate_novo: dict | None,
           lat_ab: dict | None = None) -> dict:
    """`base`/`novo` = {colecao: (itens, resumo)}. Cada caminho é medido à parte; nunca se somam.
    `lat_ab` = {colecao: saída de `run_latency_ab`}: a latência vem da passada intercalada (razão novo/base),
    não das rodadas separadas, que misturam a carga do Mac com o efeito da mudança."""
    caminhos = ("cli", "mcp")
    crit = []

    def add(nome, ok, detalhe):
        crit.append({"nome": nome, "ok": bool(ok), "detalhe": detalhe})

    ganhos = {}
    det = {}
    for c in caminhos:
        b, cc = _bc(base[visado][0], novo[visado][0], c)
        ganhos[c] = b - cc
        det[c] = {"b": b, "c": cc, "p": round(mcnemar(b, cc), 4)}
    add("ganho_visado", all(g >= GANHO_MIN for g in ganhos.values()), det)

    perdas = {}
    for col in (k for k in novo if k != visado):
        for c in caminhos:
            b, cc = _bc(base[col][0], novo[col][0], c)
            perdas[f"{col}|{c}"] = cc - b
    add("perda_outro", all(p <= PERDA_MAX for p in perdas.values()), perdas)

    razoes = {}
    for col in novo:
        lb, ln = base[col][1].get("latencia") or {}, novo[col][1].get("latencia") or {}
        ab = ((lat_ab or {}).get(col) or {}).get("resultado") or {}
        for c in caminhos:
            for q in ("p50", "p95"):
                if ab.get(c):
                    razoes[f"{col}|{c}|{q}"] = ab[c]["razao"][q]
                elif lb.get(c) and ln.get(c) and lb[c][q]:
                    razoes[f"{col}|{c}|{q}"] = round(ln[c][q] / lb[c][q], 3)
    pior = max(razoes.values(), default=1.0)
    pede_dono = LAT_OK < pior <= LAT_DONO and all(g >= GANHO_DONO for g in ganhos.values())
    add("latencia", pior <= LAT_OK or pede_dono, {"pior": pior, "razoes": razoes})

    difs = []
    for col, (itens, _) in novo.items():
        por = {c: {i["id"]: i for i in itens if i["caminho"] == c and i["degrau"] is not None} for c in caminhos}
        for q in por["cli"].keys() & por["mcp"].keys():
            a, m = por["cli"][q], por["mcp"][q]
            if any(a["config"].get(x) != m["config"].get(x) for x in ("k", "cand", "cand_sintetico", "pools_separados",
                                                         "penalidade_sintetico")) \
                    or {tuple(x) for x in a["ids_final"]} != {tuple(x) for x in m["ids_final"]}:  # conjunto, não ordem
                difs.append(f"{col}:{q}")
    add("mcp_cli_iguais", not difs, {"n_diferentes": len(difs), "exemplos": difs[:5]})

    if gate_base is not None and gate_novo is not None:
        tb, tn = _taxa_sem_base(gate_base), _taxa_sem_base(gate_novo)
        add("frase_sem_base", tn <= tb, {"base": round(tb, 4), "novo": round(tn, 4)})

    ok = all(c["ok"] for c in crit)
    return {"ok": ok, "pede_dono": ok and pede_dono, "criterios": crit}

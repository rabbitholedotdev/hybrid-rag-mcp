"""MCP server — hybrid search + rerank over an indexed corpus.

An MCP client calls these tools to retrieve passages (hybrid search in Qdrant)
and then drafts an answer. Optional `answer` uses a local Claude CLI or API.

Run (stdio):
    python -m rag.mcp_server
"""
import logging
import os
import re
import sys
import time
from pathlib import Path

from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings

# Silencia logs verbosos das libs (em stdio, ruído no stderr atrapalha; mantém WARNING+).
for _n in ("httpx", "httpcore", "qdrant_client", "urllib3"):
    logging.getLogger(_n).setLevel(logging.WARNING)

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import (  # noqa: E402
    COLLECTION_NAME, MCP_HTTP_PATH, RAG_POOLS_SEPARADOS, RAG_PENALIDADE_SINTETICO,
    LLM_PROVIDER, RAG_DEADLINE_S,
)
from rag.answer import answer as run_answer, format_text  # noqa: E402
from rag.query import retrieve, get_lesson_chunks, expand_context, generator, verifier  # noqa: E402
from rag.vectorstore import get_client  # noqa: E402

# Segurança de transporte: quando servido atrás de um túnel/proxy (Cloudflare, etc.)
# o Host chega como o domínio público, disparando a proteção anti-DNS-rebinding (421).
# Como o acesso já é protegido por bearer token, permitimos os hosts esperados:
#   MCP_ALLOWED_HOSTS="a.trycloudflare.com,meu.dominio"  -> mantém a proteção só p/ esses
#   (vazio) -> desliga a proteção de Host (auth por token cobre o acesso).
_allowed = [h.strip() for h in os.getenv("MCP_ALLOWED_HOSTS", "").split(",") if h.strip()]
if _allowed:
    _ts = TransportSecuritySettings(
        enable_dns_rebinding_protection=True,
        allowed_hosts=_allowed + ["127.0.0.1:8000", "localhost:8000"],
        allowed_origins=["*"],
    )
else:
    _ts = TransportSecuritySettings(enable_dns_rebinding_protection=False)

mcp = FastMCP("rag-mcp", transport_security=_ts)

# Camada SINTÉTICA (resumo/qa) = gerada por LLM sobre a transcrição. Serve para
# MAPEAR/ORIENTAR, mas NÃO é citável: pode conter frameworks nomeados e números
# inventados. Fonte primária (citável) = video/material/texto.
_SINTETICO = {"resumo", "qa"}
_AVISO_SINTETICO = (
    "Resumo gerado por IA sobre a transcrição — pode conter estrutura, nomes de "
    "'framework' e NÚMEROS inventados. Use só para se orientar; verifique na "
    "transcrição crua (obter_aula) antes de citar qualquer número/passo/afirmação."
)

# URLs "genéricas" (sem id de aula) onde um ?t=Ns NÃO leva a lugar nenhum útil —
# o deep-link com timestamp só vale quando a URL identifica o recurso.
_URL_GENERICA = re.compile(r"/curso/?$")


def _link(m: dict) -> tuple[str, str, bool]:
    """Devolve (url, url_at, tem_deeplink) saneados. Se a URL é genérica (não aponta
    para a aula), descarta o ?t= inútil e sinaliza tem_deeplink=False — assim o LLM
    não cita um timestamp que não funciona."""
    url = m.get("url", "") or ""
    url_at = m.get("url_at", "") or ""
    if not url or _URL_GENERICA.search(url):
        return url, url, False
    return url, (url_at or url), bool(url_at and url_at != url)


@mcp.tool(
    annotations={
        "title": "Buscar nas transcrições do YouTube",
        "readOnlyHint": True,
        "openWorldHint": False,
    }
)
def buscar_transcricoes(
    query: str,
    k: int = 10,
    area: str | list[str] | None = None,
    channel_id: str | list[str] | None = None,
    source_type: str | list[str] | None = None,
    contexto: bool = True,
    diversificar: bool = False,
    module: str | None = None,
    hyde: bool = False,
) -> dict:
    """Busca trechos relevantes nas transcrições/materiais indexados.

    Usa busca híbrida (semântica + palavra-chave) + RERANK (cross-encoder multilingual)
    no Qdrant. Use sempre que o usuário perguntar algo sobre o conteúdo indexado.
    Depois, redija a resposta em português citando os vídeos e os links com timestamp
    (`url_at`). Para achar uma aula/material PELO NOME, use antes `encontrar_aula`.

    IMPORTANTE — áreas de conhecimento: o conteúdo é segregado por `area` (ex.:
    ofm, copywriting, mindset). Para "estudar separado", passe `area` e a busca
    fica restrita àquele assunto. Use `listar_areas` para descobrir as áreas.

    A busca mistura a camada sintética (resumo/qa) com a transcrição crua e remove
    duplicados (mesma aula republicada em vários cursos). ATENÇÃO — contrato de citação:
    cada resultado traz `citavel` (bool). `resumo`/`qa` (`citavel=False`) são gerados por
    IA e podem conter frameworks/números INVENTADOS: use-os para LOCALIZAR o assunto, mas
    NUNCA cite número/passo/afirmação deles sem conferir num resultado `citavel=True`
    (transcrição/material). Cada resultado traz também `tem_deeplink` (bool): quando False,
    NÃO cite timestamp — a fonte não tem link para o minuto.

    Args:
        query: A pergunta ou termos de busca.
        k: Quantos trechos retornar (padrão 10: pool de rerank 50+20).
        area: Opcional (recomendado). Restringe a busca a uma área — aceita UMA
            (`"constitucional"`) ou VÁRIAS (`["constitucional","trabalho"]`, casa qualquer uma).
        channel_id: Opcional. Um canal/curso ou lista deles.
        source_type: Opcional. Filtra o TIPO de conteúdo (str ou lista):
            - "video"    : transcrição da aula (fala crua, pode ser ruidosa)
            - "material" : PDFs/planilhas/docs/playbooks
            - "resumo"   : RESUMO destilado da aula (método limpo, frameworks nomeados)
            - "qa"       : PERGUNTAS E RESPOSTAS extraídas da aula
            - "texto"    : descrições/textos de aula
            DICA: para perguntas conceituais ("como faço X?"), `resumo` e `qa` costumam
            responder melhor que `video` (a transcrição crua é fala solta). Sem filtro,
            o rerank já mistura os tipos e escolhe o melhor.
        contexto: Se True (padrão), inclui `texto_expandido` (trecho + vizinhos, sem
            cortar a frase). Desligue para respostas mais enxutas.
        diversificar: Se True, no máx. 1 trecho por aula (cobre mais fontes).
        module: Opcional. Restringe a um MÓDULO/CURSO específico. Casamento EXATO —
            descubra o nome exato com `listar_modulos(area=...)`. Útil quando o acervo
            tem centenas de cursos e você quer buscar só dentro de um.
        hyde: Se True, expande a pergunta com uma hipótese de resposta antes de buscar
            (ajuda perguntas vagas/curtas). Custa uma chamada de LLM; deixe False p/ termos precisos.

    Returns:
        dict com `count` e `results`: cada item traz área, título, canal, tipo, módulo,
        link, link no minuto (`url_at`), `tem_deeplink`, timestamp, score e o texto.
    """
    docs = retrieve(query, k=max(k, k * 2 if diversificar else k),
                    channel=channel_id, area=area, source_type=source_type,
                    module=module, hyde=hyde,
                    fetch_k=max(k * 5, 30), fetch_k_sintetico=max(k * 2, 12),  # diversificar dobra o k, não o pool: o rerank custa ~0,04 s por candidato
                    pools_separados=RAG_POOLS_SEPARADOS,
                    penalidade_sintetico=RAG_PENALIDADE_SINTETICO)
    if diversificar:
        seen, uniq = set(), []
        for d in docs:
            vid = d.metadata.get("video_id")
            if vid in seen:
                continue
            seen.add(vid); uniq.append(d)
        docs = uniq[:k]
    else:
        docs = docs[:k]

    client = get_client() if contexto else None
    results = []
    for d in docs:
        m = d.metadata
        st = (m.get("source_type", "") or "").lower()
        citavel = st not in _SINTETICO
        item = {
            "area": m.get("area", ""),
            "title": m.get("title", ""),
            "channel": m.get("channel", ""),
            "channel_id": m.get("channel_id", ""),
            "video_id": m.get("video_id", ""),
            "source_type": m.get("source_type", ""),
            "citavel": citavel,
            "module": m.get("module", ""),
            **(lambda u, ua, dl: {"url": u, "url_at": ua, "tem_deeplink": dl})(*_link(m)),
            "start_seconds": m.get("start", 0) if _link(m)[2] else None,
            "score": m.get("_rerank", m.get("_score", 0)),
            **({"colecao": m["_colecao"]} if m.get("_colecao") else {}),
            "text": d.page_content,
        }
        if not citavel:
            item["aviso"] = _AVISO_SINTETICO
        if contexto:
            try:
                item["texto_expandido"] = expand_context(d, client=client)
            except Exception:  # noqa: BLE001
                pass
        results.append(item)
    return {
        "count": len(results),
        "contrato": (
            "Cite números/passos/afirmações APENAS de resultados citavel=True "
            "(transcrição/material). Itens citavel=False são resumos de IA — use só "
            "para localizar o assunto e confirme na transcrição antes de afirmar."
        ),
        "results": results,
    }


def _distinct_by_title(termo, source_type=None, area=None, channel_id=None, limit=40,
                       module=None):
    """Varre a coleção casando o TÍTULO (full-text) e devolve aulas/materiais
    DISTINTOS (dedup por video_id), com link. Filtros opcionais por tipo/área/canal."""
    from qdrant_client.http.models import (
        Filter, FieldCondition, MatchValue, MatchText,
    )

    client = get_client()
    if not client.collection_exists(COLLECTION_NAME):
        return []

    must = []
    if termo:
        must.append(FieldCondition(key="metadata.title", match=MatchText(text=termo)))
    if source_type:
        must.append(FieldCondition(key="metadata.source_type",
                                   match=MatchValue(value=source_type)))
    if area:
        must.append(FieldCondition(key="metadata.area", match=MatchValue(value=area)))
    if channel_id:
        must.append(FieldCondition(key="metadata.channel_id",
                                   match=MatchValue(value=channel_id)))
    if module:
        must.append(FieldCondition(key="metadata.module", match=MatchValue(value=module)))
    flt = Filter(must=must) if must else None

    seen, items, offset = {}, [], None
    # varre em páginas até juntar `limit` distintos (ou acabar)
    for _ in range(40):
        points, offset = client.scroll(
            collection_name=COLLECTION_NAME, scroll_filter=flt,
            limit=512, offset=offset, with_payload=True, with_vectors=False,
        )
        for p in points:
            m = (p.payload or {}).get("metadata", {}) or {}
            vid = m.get("video_id")
            if vid in seen:
                continue
            seen[vid] = True
            items.append({
                "video_id": vid,
                "title": m.get("title", ""),
                "area": m.get("area", ""),
                "channel": m.get("channel", ""),
                "channel_id": m.get("channel_id", ""),
                "source_type": m.get("source_type", ""),
                "module": m.get("module", ""),
                "url": m.get("url", ""),
                "url_at": m.get("url_at", ""),
            })
            if len(items) >= limit:
                return items
        if offset is None or not points:
            break
    return items


@mcp.tool(
    annotations={
        "title": "Encontrar aula/material pelo nome",
        "readOnlyHint": True,
        "openWorldHint": False,
    }
)
def encontrar_aula(
    termo: str,
    area: str | None = None,
    source_type: str | None = None,
    channel_id: str | None = None,
    limit: int = 40,
    module: str | None = None,
) -> dict:
    """Encontra AULAS e MATERIAIS específicos pelo NOME (título) — não busca no texto
    da transcrição, busca no título do item. Use quando o usuário pedir uma aula ou
    material pelo nome (ex.: "a aula de Neurovendas", "o material Matriz RFV", "o
    fluxograma da consultoria"). Retorna a LISTA de itens distintos que casam, cada um
    com link direto. Para o CONTEÚDO/trechos de uma aula, use `buscar_transcricoes`.

    O `termo` casa por palavras do título (full-text), não precisa ser exato.

    Args:
        termo: Palavras do título da aula/material (ex.: "matriz rfv", "neurovendas").
        area: Opcional. Restringe a uma área (ver `listar_areas`).
        source_type: Opcional. "material" (PDFs/planilhas/docs) ou "video" (aulas).
        channel_id: Opcional. Restringe a um canal/curso (ver `listar_canais`).
        limit: Máx. de itens distintos a retornar (padrão 40).
        module: Opcional. Restringe a um MÓDULO/CURSO (nome EXATO — ver `listar_modulos`).

    Returns:
        dict com `count` e `results`: cada item traz título, área, canal, tipo
        (`source_type`), módulo, link (`url`) e link com timestamp (`url_at`).
    """
    items = _distinct_by_title(
        termo, source_type=source_type, area=area,
        channel_id=channel_id, limit=limit, module=module,
    )
    return {"count": len(items), "results": items}


@mcp.tool(
    annotations={
        "title": "Listar materiais (PDFs/planilhas/docs)",
        "readOnlyHint": True,
        "openWorldHint": False,
    }
)
def listar_materiais(
    termo: str | None = None,
    area: str | None = None,
    channel_id: str | None = None,
    limit: int = 60,
    module: str | None = None,
) -> dict:
    """Lista os MATERIAIS indexados (PDFs, planilhas, docs, playbooks — tudo que é
    `source_type=material`, não vídeo). Útil para descobrir e localizar materiais
    específicos. Opcionalmente filtra por palavra no título, área e canal.

    Args:
        termo: Opcional. Palavra(s) do nome do material (ex.: "rfv", "fluxograma").
        area: Opcional. Restringe a uma área.
        channel_id: Opcional. Restringe a um canal/curso.
        limit: Máx. de materiais distintos a retornar (padrão 60).
        module: Opcional. Restringe a um MÓDULO/CURSO (nome EXATO — ver `listar_modulos`).

    Returns:
        dict com `count` e `results`: cada material com título, área, canal, módulo e link.
    """
    items = _distinct_by_title(
        termo or "", source_type="material", area=area,
        channel_id=channel_id, limit=limit, module=module,
    )
    return {"count": len(items), "results": items}


@mcp.tool(
    annotations={
        "title": "Obter aula/material completo",
        "readOnlyHint": True,
        "openWorldHint": False,
    }
)
def obter_aula(video_id: str, max_chars: int = 20000, offset: int = 0, colecao: str | None = None) -> dict:
    """Retorna a AULA ou MATERIAL INTEIRO (transcrição/texto completo, em ordem) a
    partir do `video_id`. Fluxo típico: `encontrar_aula(...)` para achar o item pelo
    nome, pegar o `video_id`, e então `obter_aula(video_id)` para ler tudo. Ideal para
    resumir uma aula específica ou responder com base no conteúdo completo dela.

    Args:
        video_id: O `video_id` do item (vem de `encontrar_aula`/`buscar_transcricoes`).
        max_chars: Tamanho do pedaço retornado (padrão 20000).
        colecao: Coleção do item, quando a busca veio de várias (campo `colecao` do
            resultado de `buscar_transcricoes`). Omitido = a coleção padrão.
        offset: Posição inicial em caracteres (padrão 0). Para PAGINAR uma aula longa:
            chame de novo com `offset = offset_anterior + max_chars` enquanto
            `tem_mais` for True. Use `total_chars` para saber o tamanho completo.

    Returns:
        dict com título, área, canal, tipo, módulo, link, nº de trechos, `text` (o
        pedaço pedido), `total_chars`, `offset`, `proximo_offset` e `tem_mais`.
    """
    chunks = get_lesson_chunks(video_id, collection=colecao)
    if not chunks:
        return {"found": False, "video_id": video_id, "note": "Nenhum trecho com esse video_id."}
    meta = chunks[0]["meta"]
    full = " ".join(c["text"] for c in chunks).strip()
    total = len(full)
    offset = max(0, int(offset))
    pedaco = full[offset:offset + max_chars]
    fim = offset + len(pedaco)
    return {
        "found": True,
        "video_id": video_id,
        "title": meta.get("title", ""),
        "area": meta.get("area", ""),
        "channel": meta.get("channel", ""),
        "module": meta.get("module", ""),
        "source_type": meta.get("source_type", ""),
        "url": meta.get("url", ""),
        "n_trechos": len(chunks),
        "total_chars": total,
        "offset": offset,
        "proximo_offset": fim if fim < total else None,
        "tem_mais": fim < total,
        "truncated": fim < total,  # compat
        "text": pedaco,
    }


@mcp.tool(
    annotations={
        "title": "Panorama do acervo",
        "readOnlyHint": True,
        "openWorldHint": False,
    }
)
def panorama() -> dict:
    """Visão geral do que está indexado: total de trechos, quebra por ÁREA e por TIPO
    (video/material). Use para responder "o que você tem?" e orientar buscas.

    Returns:
        dict com `total_chunks`, `por_area` e `por_tipo` (listas com contagens).
    """
    client = get_client()
    if not client.collection_exists(COLLECTION_NAME):
        return {"total_chunks": 0, "note": "Base vazia. Rode a ingestão."}
    total = client.count(COLLECTION_NAME, exact=True).count
    fa = client.facet(collection_name=COLLECTION_NAME, key="metadata.area", limit=100)
    por_area = sorted([{"area": h.value, "chunks": h.count} for h in fa.hits],
                      key=lambda x: x["chunks"], reverse=True)
    ft = client.facet(collection_name=COLLECTION_NAME, key="metadata.source_type", limit=20)
    por_tipo = sorted([{"tipo": h.value, "chunks": h.count} for h in ft.hits],
                      key=lambda x: x["chunks"], reverse=True)
    return {"total_chunks": total, "por_area": por_area, "por_tipo": por_tipo}


@mcp.tool(
    annotations={
        "title": "Listar áreas de conhecimento",
        "readOnlyHint": True,
        "openWorldHint": False,
    }
)
def listar_areas() -> dict:
    """Lista as ÁREAS DE CONHECIMENTO indexadas (o nível acima de canal), com
    contagem de trechos. Cada área é um assunto isolado para estudo separado
    (ex.: constitucional, trabalho, consumidor). Use a `area` retornada aqui em
    `buscar_transcricoes(area=...)` para restringir a busca a um assunto.

    Returns:
        dict com `count` e `areas`: cada item traz `area` e `chunks`.
    """
    client = get_client()
    if not client.collection_exists(COLLECTION_NAME):
        return {"count": 0, "areas": [], "note": "Base vazia. Rode a ingestão."}
    facets = client.facet(collection_name=COLLECTION_NAME, key="metadata.area", limit=100)
    areas = [{"area": h.value, "chunks": h.count} for h in facets.hits]
    areas.sort(key=lambda a: a["chunks"], reverse=True)
    return {"count": len(areas), "areas": areas}


@mcp.tool(
    annotations={
        "title": "Listar canais indexados",
        "readOnlyHint": True,
        "openWorldHint": False,
    }
)
def listar_canais(area: str | None = None) -> dict:
    """Lista os canais/fontes indexados, com contagem de trechos.

    Útil para descobrir o que está disponível e obter o `channel_id`.

    Args:
        area: Opcional. Restringe aos canais de uma área de conhecimento
            (use a `area` retornada por `listar_areas`).

    Returns:
        dict com `count` e `channels`: cada item traz `channel` (nome),
        `channel_id`, `area` e `chunks` (quantidade de trechos indexados).
    """
    from qdrant_client.http.models import Filter, FieldCondition, MatchValue

    client = get_client()
    if not client.collection_exists(COLLECTION_NAME):
        return {"count": 0, "channels": [], "note": "Base vazia. Rode a ingestão."}

    facet_filter = None
    if area:
        facet_filter = Filter(must=[FieldCondition(
            key="metadata.area", match=MatchValue(value=area))])

    facets = client.facet(
        collection_name=COLLECTION_NAME,
        key="metadata.channel_id",
        facet_filter=facet_filter,
        limit=100,
    )

    channels = []
    for hit in facets.hits:
        ch_id = hit.value
        # busca 1 ponto para descobrir nome legível do canal + área
        points, _ = client.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter={
                "must": [{"key": "metadata.channel_id", "match": {"value": ch_id}}]
            },
            limit=1,
            with_payload=True,
            with_vectors=False,
        )
        name = ch_area = ""
        if points:
            meta = points[0].payload.get("metadata", {})
            name = meta.get("channel", "")
            ch_area = meta.get("area", "")
        channels.append({"channel": name, "channel_id": ch_id,
                         "area": ch_area, "chunks": hit.count})

    channels.sort(key=lambda c: c["chunks"], reverse=True)
    return {"count": len(channels), "channels": channels}


@mcp.tool(
    annotations={
        "title": "Listar módulos/cursos",
        "readOnlyHint": True,
        "openWorldHint": False,
    }
)
def listar_modulos(area: str | None = None, channel_id: str | None = None,
                   termo: str | None = None, limit: int = 100) -> dict:
    """Lista os MÓDULOS/CURSOS indexados (o nível abaixo de canal), com contagem de
    trechos. Use para descobrir o NOME EXATO de um curso e então filtrar buscas com
    `buscar_transcricoes(..., module="<nome exato>")`.

    Essencial quando o acervo tem muitos cursos e o usuário quer estudar/perguntar
    dentro de UM curso específico.

    Args:
        area: Opcional. Restringe aos módulos de uma área (ver `listar_areas`).
        channel_id: Opcional. Restringe aos módulos de um canal (ver `listar_canais`).
        termo: Opcional. Filtra os nomes que contenham esta palavra (case-insensitive).
        limit: Máx. de módulos a retornar (padrão 100).

    Returns:
        dict com `count` e `modulos`: cada item traz `module` (nome exato) e `chunks`.
    """
    from qdrant_client.http.models import Filter, FieldCondition, MatchValue

    client = get_client()
    if not client.collection_exists(COLLECTION_NAME):
        return {"count": 0, "modulos": [], "note": "Base vazia. Rode a ingestão."}

    must = []
    if area:
        must.append(FieldCondition(key="metadata.area", match=MatchValue(value=area)))
    if channel_id:
        must.append(FieldCondition(key="metadata.channel_id",
                                   match=MatchValue(value=channel_id)))
    flt = Filter(must=must) if must else None

    facets = client.facet(collection_name=COLLECTION_NAME, key="metadata.module",
                          facet_filter=flt, limit=max(limit * 3, 300))
    mods = [{"module": h.value, "chunks": h.count} for h in facets.hits]
    if termo:
        t = termo.lower()
        mods = [m for m in mods if t in (m["module"] or "").lower()]
    mods.sort(key=lambda m: m["chunks"], reverse=True)
    mods = mods[:limit]
    return {"count": len(mods), "modulos": mods}


@mcp.tool(
    annotations={
        "title": "Documentação / manual do rag-mcp",
        "readOnlyHint": True,
        "openWorldHint": False,
    }
)
def documentacao() -> dict:
    """MANUAL do rag-mcp — leia ANTES de usar as outras ferramentas se você não
    conhece este acervo. Explica o que está indexado, o catálogo de ferramentas,
    os TIPOS de conteúdo (incl. a camada sintética `resumo`/`qa`) e os fluxos
    recomendados. Os números vêm ao vivo da base, então refletem o estado atual.

    Returns:
        dict com `o_que_e`, `acervo` (números ao vivo), `tipos_de_conteudo`,
        `ferramentas`, `fluxos_recomendados` e `dicas`.
    """
    acervo = {}
    try:
        acervo = panorama()
    except Exception as e:  # noqa: BLE001
        acervo = {"erro": str(e)[:120]}

    return {
        "o_que_e": (
            "RAG sobre transcrições de aulas (YouTube/cursos) + materiais (PDFs, planilhas, "
            "docs) de vários acervos. Busca HÍBRIDA (semântica + palavra-chave) com RERANK "
            "cross-encoder multilingual. Responda SEMPRE em português citando a fonte e o "
            "link com timestamp (`url_at`)."
        ),
        "acervo": acervo,
        "contrato_de_citacao": (
            "REGRA DE OURO. Fonte PRIMÁRIA (citável): video/material/texto. Camada "
            "SINTÉTICA (resumo/qa): gerada por LLM sobre a transcrição — ótima para MAPEAR "
            "e achar, mas pode conter frameworks nomeados e NÚMEROS INVENTADOS. NUNCA cite "
            "número, %, métrica, nome de framework ou passo específico vindo de resumo/qa "
            "sem confirmar na transcrição (`obter_aula`). Se só existir no resumo, diga que "
            "é 'não verificado na transcrição'."
        ),
        "tipos_de_conteudo": {
            "video": "PRIMÁRIA (citável). Transcrição CRUA da aula (fala falada, auto-legenda; pode ser ruidosa). Tem timestamp/deep-link.",
            "material": "PRIMÁRIA (citável). PDFs, planilhas, docs, playbooks.",
            "resumo": "SINTÉTICA — NÃO CITÁVEL. Resumo destilado por IA. Pode conter frameworks nomeados/números inventados. Use para ORIENTAR, não para citar. Aponta para a MESMA aula/URL.",
            "qa": "SINTÉTICA — NÃO CITÁVEL. Pares Pergunta→Resposta gerados por IA sobre a aula.",
            "texto": "PRIMÁRIA. Descrições/textos de aula (sem transcrição).",
            "NOTA": (
                "A camada sintética é ADITIVA: a transcrição crua continua inteira no índice. "
                "Fluxo seguro: use `resumo`/`qa` para LOCALIZAR o tema e a aula certa; depois "
                "puxe `video`/`obter_aula` e cite a transcrição. Sem filtro, o rerank mistura os "
                "tipos — por isso todo resultado vem com `citavel` (bool)."
            ),
        },
        "ferramentas": {
            "buscar_transcricoes": "Busca semântica+keyword+rerank. A principal. Filtros: area, channel_id, source_type.",
            "encontrar_aula": "Acha aula/material pelo NOME (título), não pelo conteúdo. Devolve lista com links.",
            "obter_aula": "Texto COMPLETO de uma aula/material pelo `video_id`.",
            "obter_resumo": "Resumo destilado + Q&A (camada sintética, NÃO CITÁVEL — só para orientar) de uma aula, pelo `video_id`.",
            "listar_materiais": "Lista só materiais (PDFs/planilhas/docs).",
            "listar_areas": "Áreas de conhecimento (assuntos isolados). Use para restringir buscas.",
            "listar_canais": "Canais/fontes indexados + `channel_id`.",
            "listar_modulos": "Módulos/CURSOS indexados. Use para achar o NOME EXATO e filtrar a busca com `module=`.",
            "panorama": "Visão geral do acervo (contagens por área e tipo).",
            "answer": "RESPOSTA PRONTA e verificada: cada frase vem com citação literal conferida por código e por um verificador (Sonnet); "
                      "frase sem prova é cortada; sem prova nenhuma → «Não encontrei isso nos trechos recuperados.»; falha → «Não foi possível verificar.». "
                      "Use `texto_final` como está. Risco residual: ao repassar, a sessão pode acrescentar texto que não foi verificado. ~40 s por chamada.",
            "saude": "Diagnóstico: a base está no ar? quantos pontos?",
        },
        "fluxos_recomendados": [
            "PERGUNTA SOBRE UM TEMA → `listar_areas` (descobrir a área) → `buscar_transcricoes(query, area=...)` → responder citando `url_at`.",
            "USUÁRIO CITA UMA AULA PELO NOME → `encontrar_aula(termo)` → pegar `video_id` → `obter_aula(video_id)` (conteúdo completo) ou `obter_resumo(video_id)` (destilado).",
            "QUER O MÉTODO/FRAMEWORK DE UMA AULA → `obter_resumo(video_id)` para SE ORIENTAR, depois `obter_aula(video_id)` e CITE a transcrição (o resumo pode ter inventado nomes/números).",
            "PROCURA UM MATERIAL (PDF/planilha) → `listar_materiais(termo=...)` ou `buscar_transcricoes(..., source_type='material')`.",
            "PERGUNTA AMPLA (cobrir várias fontes) → `buscar_transcricoes(query, k=10, diversificar=True)`.",
            "ESTUDAR/PERGUNTAR DENTRO DE UM CURSO → `listar_modulos(area=..., termo='nome parcial')` → pegar o nome EXATO → `buscar_transcricoes(query, module='<nome exato>')`.",
            "AULA MUITO LONGA → `obter_aula(video_id)` e, enquanto `tem_mais`, chamar de novo com `offset=proximo_offset`.",
        ],
        "dicas": [
            "SEMPRE passe `area` quando souber o assunto: evita misturar acervos diferentes.",
            "`diversificar=True` traz no máx. 1 trecho por aula — melhor para panoramas.",
            "`contexto=True` (padrão) devolve `texto_expandido` (trecho + vizinhos) — melhor para citar sem cortar frase.",
            "Cite sempre `url_at` (link no minuto) para o usuário pular direto ao ponto.",
            "Se a transcrição crua vier confusa (auto-legenda), use `resumo` para ENTENDER/localizar — mas volte à transcrição para CITAR número/passo/framework.",
            "Todo número, %, métrica ou nome de framework que você for afirmar precisa estar num resultado `citavel=True`. Na dúvida, `obter_aula` e confira.",
        ],
    }


@mcp.tool(
    annotations={
        "title": "Obter resumo destilado + Q&A da aula",
        "readOnlyHint": True,
        "openWorldHint": False,
    }
)
def obter_resumo(video_id: str) -> dict:
    """Retorna a CAMADA SINTÉTICA de uma aula: o RESUMO destilado e os pares
    PERGUNTA→RESPOSTA gerados por IA sobre a transcrição.

    NÃO CITÁVEL: use para ENTENDER e LOCALIZAR o método da aula rapidamente, mas o resumo
    pode ter imposto estrutura, nomeado "frameworks" e inventado números que o professor
    não disse. Para CITAR qualquer número/passo/afirmação, puxe a transcrição crua com
    `obter_aula(video_id)` e confira. Aponta para a mesma aula/URL original.

    Aceita tanto o `video_id` da aula original quanto o do record sintético
    (`syn-resumo-<id>` / `syn-qa-<id>`).

    Args:
        video_id: `video_id` da aula (de `encontrar_aula`/`buscar_transcricoes`).

    Returns:
        dict com `found`, `resumo`, `qa`, título, área, módulo e `url`. Se a aula
        ainda não tiver camada sintética, `found=False` (use `obter_aula`).
    """
    base = video_id
    for pref in ("syn-resumo-", "syn-qa-"):
        if base.startswith(pref):
            base = base[len(pref):]

    out = {"video_id": base, "found": False, "citavel": False, "aviso": _AVISO_SINTETICO}
    for kind in ("resumo", "qa"):
        chunks = get_lesson_chunks(f"syn-{kind}-{base}")
        if not chunks:
            continue
        out[kind] = " ".join(c["text"] for c in chunks).strip()
        out["found"] = True
        m = chunks[0]["meta"]
        out.setdefault("title", (m.get("title", "") or "")
                       .replace("[Resumo] ", "").replace("[Q&A] ", ""))
        out.setdefault("area", m.get("area", ""))
        out.setdefault("module", m.get("module", ""))
        out.setdefault("url", m.get("url", ""))

    if not out["found"]:
        out["note"] = (
            "Sem camada sintética para esta aula. Use `obter_aula(video_id)` para a "
            "transcrição completa."
        )
    return out


@mcp.tool(
    annotations={
        "title": "Estudar um tema (resumos + Q&A destilados)",
        "readOnlyHint": True,
        "openWorldHint": False,
    }
)
def estudar_tema(tema: str, area: str | list[str] | None = None, k: int = 8) -> dict:
    """Reúne o MÉTODO destilado sobre um tema — resumos + Q&A (frameworks, passos,
    números) já limpos, prontos para você redigir uma resposta/estudo estruturado.

    Diferente de `buscar_transcricoes` (que traz trechos CRUS com contexto para citar
    literalmente), esta tool foca na camada SINTÉTICA e devolve conteúdo compacto e
    organizado — use quando o usuário quer APRENDER/ENTENDER um assunto, não uma
    citação exata. Cada item vem sem `texto_expandido` (já é destilado).

    Args:
        tema: O assunto a estudar (ex.: "princípios da administração pública").
        area: Opcional. Uma área ou lista (ex.: ["constitucional","trabalho"]).
        k: Quantos blocos destilados trazer (padrão 8).

    Returns:
        dict com `count`, `fontes` (aulas distintas cobertas) e `blocos`: cada um traz
        tipo (resumo/qa), título, área, módulo, link da aula e o texto destilado.
    """
    docs = retrieve(tema, k=k, area=area, source_type=["resumo", "qa"])
    blocos, fontes = [], set()
    for d in docs:
        m = d.metadata
        ref = (m.get("source_ref") or m.get("video_id", "")).replace("syn-resumo-", "").replace("syn-qa-", "")
        fontes.add(ref)
        url, _, _ = _link(m)
        blocos.append({
            "tipo": m.get("source_type", ""),
            "title": re.sub(r"^\[(Resumo|Q&A)\]\s*", "", m.get("title", "")),
            "area": m.get("area", ""),
            "module": m.get("module", ""),
            "url": url,
            "score": m.get("_rerank", m.get("_score", 0)),
            "text": d.page_content,
        })
    return {"count": len(blocos), "fontes": len(fontes), "blocos": blocos}


@mcp.tool(
    annotations={
        "title": "Resposta verificada com prova por frase",
        "readOnlyHint": True,
        "openWorldHint": False,
    }
)
def answer(question: str, collection: str | None = None, k: int = 15) -> dict:
    """Responde a pergunta só com o que os trechos recuperados provam. Cada frase tem citação
    literal conferida por código e por um verificador de outro modelo; frase sem prova é cortada.

    Returns:
        dict de `rag.answer.answer` + `texto_final` (repasse como está). `cortadas` é só auditoria.
        `status`: ok | nao_encontrado | falha_verificacao (falha fechada, sem texto não verificado).
    """
    if collection and collection != COLLECTION_NAME:
        return {"erro": f"esta instância serve a coleção «{COLLECTION_NAME}», não «{collection}»"}
    t0 = time.monotonic()
    docs = retrieve(question, k)
    busca = round(time.monotonic() - t0, 2)
    res = run_answer(question, docs, generator(LLM_PROVIDER), verifier(), RAG_DEADLINE_S)
    res["tempos"]["busca_s"] = busca
    texto_final = format_text(res)
    # auditoria sem o texto do gerador: frase cortada não verificada nunca sai pelo MCP
    res["cortadas"] = [{"motivo": c["motivo"]} for c in res["cortadas"]]
    return {**res, "texto_final": texto_final}


@mcp.tool(
    annotations={
        "title": "Saúde da base (diagnóstico)",
        "readOnlyHint": True,
        "openWorldHint": False,
    }
)
def saude() -> dict:
    """Diagnóstico rápido: a base vetorial está no ar e respondendo? Quantos pontos?
    Use quando uma busca falhar ou vier vazia, para distinguir "não tem esse conteúdo"
    de "a base está fora do ar".

    Returns:
        dict com `ok`, `colecao`, `pontos`, `latencia_ms` e (se falhar) `erro`.
    """
    import time
    t0 = time.time()
    try:
        client = get_client()
        existe = client.collection_exists(COLLECTION_NAME)
        if not existe:
            return {"ok": False, "colecao": COLLECTION_NAME, "erro": "coleção não existe"}
        # exact=False: contagem aproximada, instantânea (o exato varria 605k pontos ~800ms).
        n = client.count(COLLECTION_NAME, exact=False).count
        return {
            "ok": True,
            "colecao": COLLECTION_NAME,
            "pontos": n,
            "latencia_ms": round((time.time() - t0) * 1000),
        }
    except Exception as e:  # noqa: BLE001
        return {
            "ok": False,
            "colecao": COLLECTION_NAME,
            "erro": str(e)[:200],
            "latencia_ms": round((time.time() - t0) * 1000),
        }


_PUBLIC_PATHS = frozenset({"/healthz", "/", "/favicon.ico", "/logo.png"})


class BearerAuthMiddleware:
    """Require `Authorization: Bearer <MCP_AUTH_TOKEN>` on every non-public path.

    Token-in-URL (path or query) is not accepted: it leaks into access logs.
    Pure ASGI so streamable HTTP / SSE is not buffered by BaseHTTPMiddleware.
    """

    def __init__(self, app, token: str):
        self.app = app
        self.token = token
        self._expected = f"Bearer {token}".encode("latin1")

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        path = scope.get("path", "")
        if path in _PUBLIC_PATHS:
            await self.app(scope, receive, send)
            return
        headers = scope.get("headers") or []
        auth = b""
        for key, value in headers:
            if key == b"authorization":
                auth = value
                break
        if auth == self._expected:
            await self.app(scope, receive, send)
            return
        await send({
            "type": "http.response.start",
            "status": 401,
            "headers": [
                (b"content-type", b"application/json"),
                (b"www-authenticate", b"Bearer"),
            ],
        })
        await send({"type": "http.response.body", "body": b'{"error":"unauthorized"}'})


def build_app():
    """Starlette app for streamable-http, without binding a port — used by tests
    (`TestClient`) and `_run_http`. Auth is `Authorization: Bearer <MCP_AUTH_TOKEN>`
    on `/mcp` (or `MCP_HTTP_PATH`). Missing/wrong token → 401. `/healthz` is public.
    """
    import os
    from pathlib import Path as _Path

    from starlette.middleware.base import BaseHTTPMiddleware
    from starlette.responses import HTMLResponse, JSONResponse, Response

    _root = _Path(__file__).resolve().parent.parent

    def _load_asset(env, default_name):
        p = os.getenv(env) or str(_root / default_name)
        try:
            b = _Path(p).read_bytes()
        except Exception:  # noqa: BLE001
            return None, None, None
        ct = ("image/svg+xml" if p.endswith(".svg")
              else "image/png" if p.endswith(".png")
              else "image/x-icon")
        return b, ct, p

    _favicon_bytes, _favicon_ct, _ = _load_asset("MCP_FAVICON", "favicon.ico")
    _logo_bytes, _logo_ct, _ = _load_asset("MCP_LOGO", "logo.png")

    token = os.environ.get("MCP_AUTH_TOKEN")
    if not token or len(token) < 16:
        raise SystemExit("MCP_AUTH_TOKEN ausente ou fraco (>=16 chars). Abortando.")

    mcp.settings.streamable_http_path = MCP_HTTP_PATH or "/mcp"

    landing_html = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>rag-mcp</title>
<meta name="theme-color" content="#081C36">
<link rel="icon" href="/favicon.ico">
<style>
  :root{color-scheme:dark}
  *{box-sizing:border-box}
  body{margin:0;min-height:100vh;display:flex;align-items:center;justify-content:center;
    font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
    background:radial-gradient(1100px 620px at 50% -12%,#0b2340,#081C36 55%,#05101f);color:#eaf0f8}
  .card{width:min(92vw,440px);padding:42px 34px;border-radius:24px;text-align:center;
    background:linear-gradient(180deg,rgba(20,44,78,.55),rgba(11,30,57,.55));
    border:1px solid rgba(232,195,106,.22);
    box-shadow:0 40px 90px -35px rgba(0,0,0,.7),inset 0 1px 0 rgba(255,255,255,.05);
    backdrop-filter:blur(14px)}
  .logo{width:72px;height:72px;margin:0 auto 20px;filter:drop-shadow(0 14px 30px rgba(0,0,0,.5))}
  .logo svg{width:100%;height:100%;display:block}
  h1{margin:0 0 6px;font-size:24px;font-weight:680;letter-spacing:-.02em;color:#fff}
  h1 b{color:#E8C36A;font-weight:680}
  .sub{margin:0 0 22px;font-size:13.5px;color:#9db0c9}
  .badge{display:inline-flex;align-items:center;gap:7px;font-size:12.5px;font-weight:600;
    padding:6px 13px;border-radius:999px;background:rgba(40,199,111,.16);color:#4fe08a;margin-bottom:22px;
    border:1px solid rgba(40,199,111,.28)}
  .dot{width:8px;height:8px;border-radius:50%;background:currentColor;
    box-shadow:0 0 0 4px rgba(40,199,111,.16);animation:pulse 2s infinite}
  @keyframes pulse{0%,100%{opacity:1}50%{opacity:.45}}
  code{display:block;font-size:13px;padding:11px 14px;border-radius:12px;margin:0 0 24px;
    font-family:ui-monospace,SFMono-Regular,Menlo,monospace;color:#c4d0df;
    background:rgba(0,0,0,.28);border:1px solid rgba(255,255,255,.07);word-break:break-all}
  a.btn{display:inline-flex;align-items:center;justify-content:center;gap:9px;width:100%;
    padding:13px 18px;border-radius:13px;text-decoration:none;font-weight:700;font-size:15px;
    background:linear-gradient(135deg,#F5DFA0,#E8C36A);color:#0a1e39;
    box-shadow:0 14px 30px -10px rgba(232,195,106,.55);transition:transform .12s ease,filter .12s ease}
  a.btn:hover{filter:brightness(1.05);transform:translateY(-1px)}
  .redir{margin-top:16px;font-size:12.5px;color:#7f92ab}
</style>
</head>
<body>
  <div class="card">
    <div class="logo">
      <svg viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg" aria-label="rag-mcp">
        <rect width="100" height="100" rx="22" fill="#081C36"/>
        <path d="M22 76 L50 20 L78 76" fill="none" stroke="#F8FAFC" stroke-width="7" stroke-linecap="round" stroke-linejoin="round"/>
        <path d="M33 62 L62 51" stroke="#E8C36A" stroke-width="7" stroke-linecap="round"/>
        <path d="M62 51 L53 48 M62 51 L59 60" fill="none" stroke="#E8C36A" stroke-width="7" stroke-linecap="round" stroke-linejoin="round"/>
        <circle cx="33" cy="62" r="5.5" fill="#E8C36A"/>
      </svg>
    </div>
    <h1>rag-<b>mcp</b></h1>
    <p class="sub">Hybrid search MCP — Bearer token on /mcp</p>
    <div class="badge"><span class="dot"></span> Server up</div>
    <code>endpoint: /mcp</code>
    <a class="btn" href="https://modelcontextprotocol.io/">MCP docs &rarr;</a>
    <p class="redir">Idle landing page</p>
  </div>
</body>
</html>"""

    class HealthMiddleware(BaseHTTPMiddleware):
        async def dispatch(self, request, call_next):
            if request.url.path == "/healthz":
                return JSONResponse({"status": "ok"})
            if request.url.path == "/favicon.ico" and _favicon_bytes:
                return Response(content=_favicon_bytes, media_type=_favicon_ct,
                                headers={"Cache-Control": "public, max-age=86400"})
            if request.url.path == "/logo.png" and _logo_bytes:
                return Response(content=_logo_bytes, media_type=_logo_ct,
                                headers={"Cache-Control": "public, max-age=86400"})
            if request.url.path == "/":
                return HTMLResponse(landing_html)
            return await call_next(request)

    app = mcp.streamable_http_app()
    app.add_middleware(HealthMiddleware)
    app.add_middleware(BearerAuthMiddleware, token=token)
    return app


def _run_http() -> None:
    """Sobe o app HTTP (streamable-http) via uvicorn. Config por env: MCP_HOST
    (padrão 127.0.0.1), MCP_PORT (8000). Montagem do app em `build_app()`."""
    import os

    import uvicorn

    host = os.getenv("MCP_HOST", "127.0.0.1")
    port = int(os.getenv("MCP_PORT", "8000"))
    uvicorn.run(build_app(), host=host, port=port, log_level="info")


if __name__ == "__main__":
    import os

    if os.getenv("MCP_TRANSPORT", "stdio").lower() == "http":
        _run_http()
    else:
        mcp.run()

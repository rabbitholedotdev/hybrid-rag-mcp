from eval.expansao import base_id, documentos_do_ranking, por_documento, vizinhos


class Doc:
    def __init__(self, vid, chunk, st="material", texto=None):
        self.page_content = texto or f"{vid}#{chunk}"
        self.metadata = {"video_id": vid, "chunk": chunk, "source_type": st}


def test_base_id_resolve_resumo_qa_e_janela():
    assert base_id("syn-resumo-ebj-m-abc-p2") == "ebj-m-abc"
    assert base_id("syn-qa-ebj-v-123") == "ebj-v-123"
    assert base_id("ebj-m-abc") == "ebj-m-abc" and base_id("") == ""


def test_documentos_do_ranking_agrupa_sintetico_com_primario_na_ordem():
    docs = [Doc("syn-resumo-ebj-m-a-p1", 0, "resumo"), Doc("ebj-m-b", 3), Doc("ebj-m-a", 7), Doc("ebj-m-c", 1)]
    assert documentos_do_ranking(docs, 2) == ["ebj-m-a", "ebj-m-b"]


def test_vizinhos_acrescenta_chunk_anterior_e_seguinte_sem_repetir_e_tira_sintetico():
    chunks = {"ebj-m-a": [{"chunk": i, "text": f"a{i}", "start": 0, "meta": {}} for i in range(6)]}
    docs = [Doc("syn-qa-ebj-m-a", 0, "qa"), Doc("ebj-m-a", 2), Doc("ebj-m-a", 3)]
    out = vizinhos(docs, k_semente=2, janela=1, teto=10, chunks_fn=lambda v: chunks[v])
    assert [d.metadata["chunk"] for d in out] == [1, 2, 3, 4]  # sem repetir o 2 e o 3; sintético fora


def test_vizinhos_respeita_o_teto():
    chunks = {"ebj-m-a": [{"chunk": i, "text": "x", "start": 0, "meta": {}} for i in range(20)]}
    docs = [Doc("ebj-m-a", 5), Doc("ebj-m-a", 12)]
    assert len(vizinhos(docs, 2, 1, teto=4, chunks_fn=lambda v: chunks[v])) == 4


def test_por_documento_busca_dentro_dos_melhores_documentos():
    primeira = [Doc("ebj-m-a", 0), Doc("syn-resumo-ebj-m-b-p1", 0, "resumo"), Doc("ebj-m-c", 0)]
    chamadas = []

    def no_doc(q, vid, k):
        chamadas.append((vid, k))
        return [Doc(vid, i) for i in range(k)]
    out = por_documento("pergunta", k_docs=2, k_doc=3, retrieve_fn=lambda q, k: primeira, busca_no_doc=no_doc)
    assert chamadas == [("ebj-m-a", 3), ("ebj-m-b", 3)] and len(out) == 6


def test_por_documento_sem_resultado_no_doc_cai_para_os_primarios_da_primeira_busca():
    primeira = [Doc("ebj-m-a", 4), Doc("syn-qa-ebj-m-a", 0, "qa")]
    out = por_documento("p", retrieve_fn=lambda q, k: primeira, busca_no_doc=lambda q, v, k: [])
    assert [d.metadata["video_id"] for d in out] == ["ebj-m-a"]


def test_subperguntas_no_documento_busca_cada_faceta_so_nos_melhores_documentos():
    from eval.expansao import por_subperguntas_no_documento
    primeira = [Doc("ebj-m-a", 0), Doc("syn-qa-ebj-m-b", 0, "qa"), Doc("ebj-m-c", 0)]
    visto = []

    def em_docs(q, vids, k):
        visto.append((q, tuple(vids), k))
        return [Doc(vids[0], hash(q) % 50 + i) for i in range(k)]
    gen = lambda p: '{"sub": ["faceta 1", "faceta 2"]}'
    out = por_subperguntas_no_documento("composta", k_docs=2, k_sub=3, teto=5, gen=gen, retrieve_fn=lambda q, k: primeira, busca_em_docs=em_docs)
    assert [v[:2] for v in visto] == [("faceta 1", ("ebj-m-a", "ebj-m-b")), ("faceta 2", ("ebj-m-a", "ebj-m-b"))]
    assert len(out) <= 5


def test_subperguntas_no_documento_pergunta_simples_busca_uma_vez_com_a_pergunta():
    from eval.expansao import por_subperguntas_no_documento
    visto = []
    out = por_subperguntas_no_documento("simples", gen=lambda p: '{"sub": ["simples"]}', retrieve_fn=lambda q, k: [Doc("ebj-m-a", 0)],
                                        busca_em_docs=lambda q, v, k: visto.append((q, k)) or [Doc("ebj-m-a", 1)])
    assert visto == [("simples", 15)] and len(out) == 1

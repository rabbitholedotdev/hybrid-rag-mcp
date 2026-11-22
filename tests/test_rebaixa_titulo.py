from rag.query import TITULOS_REBAIXADOS


def test_regex_pega_pagina_de_menu_e_poupa_aula():
    assert TITULOS_REBAIXADOS.search("[Resumo] Por favor, avalie o MUP - MENTORIA UBIRAJARA PARA PROCURADORIAS.")
    assert not TITULOS_REBAIXADOS.search("AÇÃO RESCISÓRIA")
    assert not TITULOS_REBAIXADOS.search("MUP - MENTORIA UBIRAJARA PARA PROCURADORIAS")

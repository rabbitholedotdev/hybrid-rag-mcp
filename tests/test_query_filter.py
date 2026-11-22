from qdrant_client.http import models as qm

from rag.query import SINTETICO, _build_filter


def test_sem_sintetico_exclui_resumo_e_qa():
    f = _build_filter(area="constitucional", sem_sintetico=True)
    assert f.must[0].key == "metadata.area"
    (cond,) = f.must_not
    assert cond.key == "metadata.source_type"
    assert isinstance(cond.match, qm.MatchAny) and set(cond.match.any) == set(SINTETICO)


def test_sem_filtro_devolve_none():
    assert _build_filter() is None

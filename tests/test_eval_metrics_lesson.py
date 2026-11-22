from types import SimpleNamespace as D

from eval import metrics as M


def doc(vid, st="video", text="x"):
    return D(page_content=text, metadata={"video_id": vid, "source_type": st})


def test_lesson_recall_ignora_prefixo_sintetico():
    assert M.primary_recall_at_k([doc("syn-qa-abc", "qa")], "abc") == 0.0
    assert M.lesson_recall_at_k([doc("syn-qa-abc", "qa")], "abc") == 1.0
    assert M.lesson_recall_at_k([doc("syn-resumo-abc", "resumo")], "abc") == 1.0
    assert M.lesson_recall_at_k([doc("outra")], "abc") == 0.0
    assert M.lesson_recall_at_k([doc("abc")], "") == 0.0


def test_content_recall_casa_copia_republicada():
    gold = ["um dois tres quatro cinco seis sete oito nove dez onze doze"]
    copia = doc("mcd-vimeo-1", text="Um dois tres quatro cinco seis sete oito nove dez")
    outra = doc("z", text="nada a ver com a aula original de copy aqui")
    sintetico = doc("syn-qa-x", "qa", text=gold[0])
    assert M.content_recall_at_k([outra, copia], gold) == 1.0
    assert M.content_recall_at_k([outra], gold) == 0.0
    assert M.content_recall_at_k([sintetico], gold) == 0.0  # sintético não conta
    assert M.content_recall_at_k([copia], []) == 0.0

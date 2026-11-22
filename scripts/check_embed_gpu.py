"""Confere o app Modal `rag-embed-e5`: os vetores da GPU têm de ser os da CPU local (cosseno mínimo
>= 0,999) e mede textos/s. Sai 1 se divergir. Uso: `python scripts/check_embed_gpu.py`."""
import sys
import time

import numpy as np

MODELO = "intfloat/multilingual-e5-large"
AMOSTRA = [
    "O controle concentrado de constitucionalidade é exercido pelo STF por meio da ADI, da ADC e da ADPF.",
    "Funil de vendas: atrair, converter, reter e fazer o cliente indicar, cada etapa com uma métrica própria.",
    "O inquérito policial é procedimento administrativo, inquisitivo e escrito, presidido pelo delegado.",
    "Na contabilidade, o balanço patrimonial separa o ativo circulante do não circulante pelo prazo de um ano.",
    "Mentoria de carreira: defina o cargo, o edital provável, a rotina de estudo e a revisão espaçada.",
    "A tese central do capítulo é que o hábito vence a motivação quando o ambiente reduz o atrito inicial.",
    "Considerando os princípios da Administração Pública, assinale a alternativa que contém a exceção.",
    "O SUS organiza-se de forma regionalizada e hierarquizada, com direção única em cada esfera de governo.",
]


def main() -> int:
    from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
    from rag.modal_embeddings import ModalDenseEmbeddings

    cpu = np.array(FastEmbedEmbeddings(model_name=MODELO).embed_documents(AMOSTRA))
    gpu_emb = ModalDenseEmbeddings(MODELO)
    gpu_emb.embed_documents(AMOSTRA)  # container frio fora da medida
    t0 = time.time()
    gpu = np.array(gpu_emb.embed_documents(AMOSTRA * 32))[: len(AMOSTRA)]
    dt = time.time() - t0
    cos = (cpu * gpu).sum(1) / (np.linalg.norm(cpu, axis=1) * np.linalg.norm(gpu, axis=1))
    print(f"cosseno mínimo {cos.min():.6f} | {len(AMOSTRA) * 32 / dt:.0f} textos/s")
    if cos.min() < 0.999:
        print("erro: vetores da GPU divergem dos da CPU", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

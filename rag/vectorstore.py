"""Coleção do Qdrant em modo HÍBRIDO: denso (semântico) + esparso (BM25).

- Denso: fastembed + multilingual-e5-large (bom em PT), vetor sem-nome ('').
- Esparso: fastembed + Qdrant/bm25 (palavra-chave), vetor 'langchain-sparse'.

Ambos rodam local e de graça. Os nomes dos vetores seguem o padrão do
langchain-qdrant para que a busca híbrida funcione sem ajustes.
"""
import os
import sys
from pathlib import Path

from langchain_qdrant import QdrantVectorStore, RetrievalMode, FastEmbedSparse
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_core.embeddings import Embeddings
from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance,
    VectorParams,
    SparseVectorParams,
    PayloadSchemaType,
    ScalarQuantization,
    ScalarQuantizationConfig,
    ScalarType,
    TextIndexParams,
    TokenizerType,
)

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import (  # noqa: E402
    QDRANT_URL,
    QDRANT_API_KEY,
    QDRANT_CF_ACCESS_CLIENT_ID,
    QDRANT_CF_ACCESS_CLIENT_SECRET,
    QDRANT_CF_ACCESS_TOKEN,
    COLLECTION_NAME,
    EMBED_MODEL,
    RERANK_MODEL,
    RAG_RERANK,
)

# Modelos/nomes fixos para casar com o schema do langchain-qdrant.
SPARSE_MODEL = "Qdrant/bm25"
DENSE_VECTOR_NAME = ""              # vetor denso sem nome (padrão do langchain)
SPARSE_VECTOR_NAME = "langchain-sparse"

BGE_M3_MODEL = "BAAI/bge-m3"

_dense = None
_sparse = None
_reranker = None  # None=não tentado, False=indisponível, obj=carregado


class BgeM3Embeddings(Embeddings):
    """Lado da QUERY do BGE-M3, na mesma sessão ONNX do lado do documento.

    Por que não fastembed aqui (o caminho e5 continua nele):

    1. POOLING. O lado do documento usa late chunking, que mean-pooleia os
       embeddings de token por span (rag/late_chunking.pool_spans) — não
       existe CLS por span arbitrário. Query e documento precisam do MESMO
       pooling pro cosseno valer. O uso oficial do BGE-M3 é CLS
       (1_Pooling/config.json: pooling_mode_cls_token true); este código
       desvia disso DE PROPÓSITO, por causa do late chunking. Usando o
       mesmo LateChunkEncoder dos dois lados, a paridade é estrutural, não
       um par de constantes que alguém pode dessincronizar de novo.

    2. PESOS SHARDED. O export ONNX do BGE-M3 tem os pesos num
       onnx/model.onnx_data separado. O download do fastembed (via
       snapshot_download) deixa cada arquivo como symlink pro blob store
       do HF, e o onnxruntime rejeita isso: "External data path escapes
       model directory" — o mesmo erro da Task 2. A saída do fastembed
       seria specific_model_path apontando pro local_dir real, mas o
       FastEmbedEmbeddings do langchain_community não repassa esse kwarg
       (só model_name/max_length/cache_dir/threads/providers).

    Bônus: uma cópia só do modelo (~2.3GB) no volume, em vez de uma no
    cache do fastembed e outra no local_dir do LateChunkEncoder.
    """

    def __init__(self, providers: list[str] | None = None):
        from rag.late_chunking import get_encoder
        self._encoder = get_encoder(providers=providers)

    def embed_query(self, text: str) -> list[float]:
        return self._encoder.embed_whole(text).tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        # a ingestão BGE-M3 não passa por aqui (usa late chunking direto,
        # ver ingest/ingest.late_chunk_points); é só o contrato de Embeddings.
        return [self.embed_query(t) for t in texts]


def get_reranker():
    """Cross-encoder p/ reordenar candidatos (precisão do top-k). Desligar com
    RAG_RERANK=0. Retorna None se desligado/indisponível (busca segue sem rerank)."""
    global _reranker
    if _reranker is None:
        if not RAG_RERANK:
            _reranker = False
        else:
            try:
                from fastembed.rerank.cross_encoder import TextCrossEncoder
                _reranker = TextCrossEncoder(model_name=RERANK_MODEL)
            except Exception:  # noqa: BLE001 — sem reranker, busca híbrida pura
                _reranker = False
    return _reranker or None


def get_dense_embeddings():
    global _dense
    if _dense is None:
        # GPU (pod RunPod): mesmo forward pass do CPU, só acelerado.
        # Vetores idênticos aos existentes — apenas o provider muda.
        providers = ["CUDAExecutionProvider"] if os.getenv("FASTEMBED_CUDA") == "1" else None
        if os.getenv("EMBED_BACKEND") == "modal" and EMBED_MODEL != BGE_M3_MODEL:
            from rag.modal_embeddings import ModalDenseEmbeddings
            _dense = ModalDenseEmbeddings(EMBED_MODEL)
        elif EMBED_MODEL == BGE_M3_MODEL:
            _dense = BgeM3Embeddings(providers=providers)
        else:
            kw = {"providers": providers} if providers else {}
            _dense = FastEmbedEmbeddings(model_name=EMBED_MODEL, **kw)
    return _dense


def get_sparse_embeddings():
    global _sparse
    if _sparse is None:
        _sparse = FastEmbedSparse(model_name=SPARSE_MODEL)
    return _sparse


def _cf_access_headers() -> dict[str, str] | None:
    # Testado 2026-09-23 contra o túnel público: o User-Agent do qdrant-client
    # passa no Cloudflare (não é bot management). O que travava era a PORTA:
    # QdrantClient(url="https://host") usa port=6333 por padrão e o Cloudflare
    # não proxia 6333 — com QDRANT_URL público, use porta explícita
    # (https://host:443).
    headers = {}
    if QDRANT_CF_ACCESS_CLIENT_ID and QDRANT_CF_ACCESS_CLIENT_SECRET:
        headers["CF-Access-Client-Id"] = QDRANT_CF_ACCESS_CLIENT_ID
        headers["CF-Access-Client-Secret"] = QDRANT_CF_ACCESS_CLIENT_SECRET
    if QDRANT_CF_ACCESS_TOKEN:
        headers["Cf-Access-Token"] = QDRANT_CF_ACCESS_TOKEN
    return headers or None


def get_client() -> QdrantClient:
    # timeout alto: upserts de lotes de embeddings + Qdrant recém-iniciado passam dos 5s default
    return QdrantClient(
        url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=120, headers=_cf_access_headers()
    )


def _dense_size(embeddings) -> int:
    return len(embeddings.embed_query("dimensão"))


def _create_collection(client: QdrantClient, size: int, name: str = COLLECTION_NAME) -> None:
    client.create_collection(
        collection_name=name,
        # denso original on-disk (usado no rescoring); COSINE.
        vectors_config=VectorParams(size=size, distance=Distance.COSINE, on_disk=True),
        sparse_vectors_config={SPARSE_VECTOR_NAME: SparseVectorParams()},
        # Quantização escalar int8 mantida em RAM (~4x menos memória, recall ~intacto
        # via rescoring com os originais on-disk). Deixa a VPS crescer muito mais.
        quantization_config=ScalarQuantization(
            scalar=ScalarQuantizationConfig(type=ScalarType.INT8, always_ram=True)
        ),
    )
    ensure_payload_indexes(client, name)
    print(f"Coleção '{name}' criada HÍBRIDA ({size}d + BM25) + int8 quantization.")


def ensure_payload_indexes(client: QdrantClient, name: str = COLLECTION_NAME) -> None:
    """Cria índices de payload (idempotente):
    - KEYWORD p/ filtrar/facetar por área, canal, vídeo, tipo de fonte,
      curso e módulo (rag.query filtra `metadata.module` com MatchValue).
    - TEXT em `metadata.title` p/ busca por NOME de aula/material (MatchText)."""
    for field in ("metadata.area", "metadata.channel_id",
                  "metadata.video_id", "metadata.source_type", "metadata.module",
                  "metadata.course"):
        try:
            client.create_payload_index(
                collection_name=name,
                field_name=field,
                field_schema=PayloadSchemaType.KEYWORD,
            )
        except Exception:  # noqa: BLE001 — já existe, tudo bem
            pass
    # índice full-text no título: achar aulas/materiais pelo nome (MatchText).
    # Tokenizer multilingual + lowercase p/ casar bem PT (acentos, maiúsc/minúsc).
    try:
        client.create_payload_index(
            collection_name=name,
            field_name="metadata.title",
            field_schema=TextIndexParams(
                type="text",
                tokenizer=TokenizerType.MULTILINGUAL,
                min_token_len=2,
                max_token_len=30,
                lowercase=True,
            ),
        )
    except Exception:  # noqa: BLE001
        pass


def _is_hybrid(client: QdrantClient, collection: str | None = None) -> bool:
    """True se a coleção já tem o vetor esparso configurado."""
    info = client.get_collection(collection or COLLECTION_NAME)
    sparse = info.config.params.sparse_vectors or {}
    return SPARSE_VECTOR_NAME in sparse


def get_vectorstore(create: bool = True, collection: str | None = None,
                    retrieval_mode: RetrievalMode = RetrievalMode.HYBRID) -> QdrantVectorStore:
    collection = collection or COLLECTION_NAME
    dense = get_dense_embeddings()
    sparse = get_sparse_embeddings()
    client = get_client()

    if not client.collection_exists(collection):
        if not create:
            raise RuntimeError(
                f"Coleção '{collection}' não existe. Rode a ingestão primeiro."
            )
        _create_collection(client, _dense_size(dense))
    elif not _is_hybrid(client, collection):
        # Coleção antiga (só densa). Precisa recriar para busca híbrida.
        raise RuntimeError(
            f"Coleção '{collection}' foi criada só com vetor denso.\n"
            f"Recrie em modo híbrido:  python -m ingest.reset  (depois reindexe)."
        )

    return QdrantVectorStore(
        client=client,
        collection_name=collection,
        embedding=dense,
        sparse_embedding=sparse,
        retrieval_mode=retrieval_mode,
        vector_name=DENSE_VECTOR_NAME,
        sparse_vector_name=SPARSE_VECTOR_NAME,
    )

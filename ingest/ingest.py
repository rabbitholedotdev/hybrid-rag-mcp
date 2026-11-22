"""Lê os JSON baixados, quebra em chunks e envia pro Qdrant.

Uso:
    python -m ingest.ingest                # ingere tudo em data/
    python -m ingest.ingest --channel <channel_id>
"""
import argparse
import json
import os
import re
import sys
import uuid
from pathlib import Path

from tqdm import tqdm
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import CHUNK_SIZE, CHUNK_OVERLAP, COLLECTION_NAME, LATE_CHUNK_MAX_TOKENS, EMBED_MODEL  # noqa: E402
from rag.vectorstore import (  # noqa: E402
    get_vectorstore,
    get_client,
    get_sparse_embeddings,
    DENSE_VECTOR_NAME,
    SPARSE_VECTOR_NAME,
)
from rag.late_chunking import get_encoder  # noqa: E402
from qdrant_client.http import models as qmodels  # noqa: E402

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def find_time_for_offset(segments, char_offset: int) -> float:
    """Aproxima o timestamp (segundos) onde o chunk começa dentro da transcrição."""
    if not segments:
        return 0.0
    pos = 0
    for seg in segments:
        seg_len = len(seg["text"]) + 1  # +1 do espaço do join
        if pos + seg_len > char_offset:
            return float(seg.get("start", 0.0))
        pos += seg_len
    return float(segments[-1].get("start", 0.0))


def load_records(channel: str | None):
    files = []
    if channel:
        files = list((DATA_DIR / channel).glob("*.json"))
    else:
        files = list(DATA_DIR.glob("*/*.json"))
    for f in files:
        if f.name.startswith("._"):  # AppleDouble (macOS tar), não é JSON
            continue
        try:
            yield json.loads(f.read_text(encoding="utf-8", errors="replace"))
        except (json.JSONDecodeError, ValueError):
            continue  # arquivo vazio/corrompido: pula em vez de abortar a ingestão


CTX_MARK = "[Contexto] "
# linha que parece título de seção: numerada ("8.2. Termos iniciais") ou toda em caixa alta
_SECAO = re.compile(r"^\s*(\d+(\.\d+)*\.?\s+\S.{2,80}|[A-ZÁÉÍÓÚÂÊÔÃÕÇ0-9 ,\-–:]{6,80})\s*$")


def cabecalho(title: str, text: str, offset: int) -> str:
    """Contexto que o chunk perde ao ser cortado: título do documento e a última seção antes do trecho."""
    secao = ""
    for ln in reversed(text[:offset].splitlines()[-200:]):
        if _SECAO.match(ln):
            secao = ln.strip()
            break
    return f"{CTX_MARK}{title}" + (f" > {secao}" if secao else "") + "\n\n"


def sem_cabecalho(texto: str) -> str:
    """Tira o cabeçalho de contexto (para medir cobertura só no conteúdo, sem o título inflar a métrica)."""
    if texto.startswith(CTX_MARK):
        return texto.split("\n\n", 1)[-1]
    return texto


def build_documents(record, splitter) -> list[Document]:
    text = record["text"]
    segments = record.get("segments", [])
    docs = []

    # create_documents preserva o offset via metadata? Não — então dividimos manualmente
    # para saber onde cada chunk começa e mapear pro timestamp.
    start = 0
    chunks = splitter.split_text(text)
    for idx, chunk in enumerate(chunks):
        # localiza o offset real do chunk no texto (a partir da posição anterior)
        offset = text.find(chunk[:60], start)
        if offset == -1:
            offset = start
        start = offset + 1

        ts = find_time_for_offset(segments, offset)
        vid = record["video_id"]
        base_url = record.get("url", "")
        is_yt = (not base_url) or ("youtube.com" in base_url) or ("youtu.be" in base_url)
        if is_yt:
            url_at = f"https://www.youtube.com/watch?v={vid}&t={int(ts)}s"
        elif segments:
            sep = "&" if "?" in base_url else "?"
            url_at = f"{base_url}{sep}t={int(ts)}s"
        else:
            url_at = base_url  # material escrito: sem timestamp
        if os.getenv("INGEST_CHUNK_HEADER") == "1":
            chunk = cabecalho(record.get("title", ""), text, offset) + chunk
        docs.append(
            Document(
                page_content=chunk,
                metadata={
                    "video_id": vid,
                    "title": record.get("title", ""),
                    "channel": record.get("channel", ""),
                    "channel_id": record.get("channel_id", ""),
                    "url": base_url,
                    "start": round(ts, 1),
                    "url_at": url_at,
                    "chunk": idx,
                    "source_type": record.get("source_type", "video"),
                    "area": record.get("area") or "geral",
                    "course": record.get("course", ""),
                    "module": record.get("module") or "(sem módulo)",
                },
            )
        )
    return docs


def build_documents_late_chunk(record, encoder) -> list[Document]:
    text = record["text"]
    segments = record.get("segments", [])
    # O encoder faz tudo com UMA encode() do documento: spans, texto do
    # chunk e offset de char saem todos dos mesmos offsets de token.
    chunks = encoder.encode_document(text, max_chunk_tokens=LATE_CHUNK_MAX_TOKENS)

    vid = record["video_id"]
    base_url = record.get("url", "")
    is_yt = (not base_url) or ("youtube.com" in base_url) or ("youtu.be" in base_url)

    docs = []
    for idx, (chunk_text, char_pos, vector) in enumerate(chunks):
        ts = find_time_for_offset(segments, char_pos)
        if is_yt:
            url_at = f"https://www.youtube.com/watch?v={vid}&t={int(ts)}s"
        elif segments:
            sep = "&" if "?" in base_url else "?"
            url_at = f"{base_url}{sep}t={int(ts)}s"
        else:
            url_at = base_url
        docs.append(
            Document(
                page_content=chunk_text,
                metadata={
                    "video_id": vid,
                    "title": record.get("title", ""),
                    "channel": record.get("channel", ""),
                    "channel_id": record.get("channel_id", ""),
                    "url": base_url,
                    "start": round(ts, 1),
                    "url_at": url_at,
                    "chunk": idx,
                    "source_type": record.get("source_type", "video"),
                    "area": record.get("area") or "geral",
                    "course": record.get("course", ""),
                    "module": record.get("module") or "(sem módulo)",
                    # consumido por late_chunk_points(); NUNCA vai pro payload
                    # do Qdrant (np.ndarray não serializa em Pydantic).
                    "embedding_vector": vector,
                },
            )
        )
    return docs


def point_id(doc: Document) -> str:
    """ID de ponto derivado de canal+vídeo+chunk: reenviar o mesmo record
    sobrescreve em vez de duplicar (upsert interrompido e rodadas paralelas).
    Não serve pro reindex_area, onde sintéticos repetem o mesmo `chunk`."""
    m = doc.metadata
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"rag:{m.get('channel_id')}:{m['video_id']}:{m['chunk']}"))


def expected_chunks(record, splitter) -> int:
    """Quantos pontos o caminho não-late (e5) gera pra este record."""
    return len(splitter.split_text(record["text"]))


def make_splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )


def late_chunk_points(docs: list[Document], sparse, ids: list[str] | None = None) -> list[qmodels.PointStruct]:
    """Pontos prontos pro upsert direto, com o vetor denso que o late
    chunking já calculou.

    vs.add_documents() não serve aqui: ele re-embeda page_content pelo
    FastEmbed (jogando fora o contexto de documento inteiro, que é o
    ponto todo do late chunking) e ainda quebra a serialização do payload
    por causa do np.ndarray em metadata. Schema (payload keys + shape do
    vetor híbrido) copiado de QdrantVectorStore._build_vectors /
    _build_payloads pra retrieval via rag/query.py seguir funcionando."""
    sparse_vectors = sparse.embed_documents([d.page_content for d in docs])
    points = []
    for n, (doc, sv) in enumerate(zip(docs, sparse_vectors)):
        metadata = {k: v for k, v in doc.metadata.items() if k != "embedding_vector"}
        points.append(qmodels.PointStruct(
            id=ids[n] if ids else str(uuid.uuid4()),
            vector={
                DENSE_VECTOR_NAME: [float(x) for x in doc.metadata["embedding_vector"]],
                SPARSE_VECTOR_NAME: qmodels.SparseVector(
                    values=sv.values, indices=sv.indices),
            },
            payload={
                QdrantVectorStore.CONTENT_KEY: doc.page_content,
                QdrantVectorStore.METADATA_KEY: metadata,
            },
        ))
    return points


def delete_channel_points(channel_id: str) -> None:
    """Apaga todos os pontos de um canal na coleção (idempotência: reindexar substitui)."""
    client = get_client()
    if not client.collection_exists(COLLECTION_NAME):
        return
    flt = qmodels.Filter(must=[qmodels.FieldCondition(
        key="metadata.channel_id", match=qmodels.MatchValue(value=channel_id))])
    client.delete(collection_name=COLLECTION_NAME,
                  points_selector=qmodels.FilterSelector(filter=flt))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--channel", default=None, help="channel_id específico (pasta em data/)")
    ap.add_argument("--batch", type=int, default=128, help="tamanho do lote de upsert")
    ap.add_argument("--no-replace", action="store_true",
                    help="não apagar pontos existentes do canal antes de indexar (append)")
    ap.add_argument("--area", default=None,
                    help="área de conhecimento (ex.: ofm, copywriting). Sobrepõe a do JSON.")
    args = ap.parse_args()

    splitter = make_splitter()

    vs = get_vectorstore(create=True)

    records = list(load_records(args.channel))
    if not records:
        print("Nenhum arquivo encontrado em data/. Rode o download primeiro.")
        return

    if args.area:  # --area sobrepõe a área de todos os registros desta rodada
        for rec in records:
            rec["area"] = args.area

    all_docs = []
    late_chunk = EMBED_MODEL == "BAAI/bge-m3"
    if late_chunk:
        encoder = get_encoder()
        for rec in records:
            all_docs.extend(build_documents_late_chunk(rec, encoder))
        # upsert direto: os vetores densos já existem (com contexto do
        # documento inteiro), vs.add_documents() os descartaria.
        client = get_client()
        sparse = get_sparse_embeddings()

        def send(batch):
            client.upsert(collection_name=COLLECTION_NAME,
                          points=late_chunk_points(batch, sparse, [point_id(d) for d in batch]))
    else:
        for rec in records:
            all_docs.extend(build_documents(rec, splitter))
        def send(batch):
            vs.add_documents(batch, ids=[point_id(d) for d in batch])

    # Idempotência: apaga os pontos dos canais que vamos reindexar (a menos que --no-replace).
    if not args.no_replace:
        channel_ids = {r.get("channel_id") for r in records if r.get("channel_id")}
        for cid in channel_ids:
            delete_channel_points(cid)
        print(f"Reindexando {len(channel_ids)} canal(is) (pontos antigos apagados).")

    print(f"{len(records)} vídeos -> {len(all_docs)} chunks. Enviando ao Qdrant...")

    import time
    for i in tqdm(range(0, len(all_docs), args.batch)):
        batch = all_docs[i : i + args.batch]
        for attempt in range(10):  # resiliência: aguenta blip de DNS/rede longo
            try:
                send(batch)
                break
            except Exception as e:  # noqa: BLE001
                if attempt == 9:
                    raise
                wait = min(60, 10 * (attempt + 1))
                print(f"\n  ! upsert falhou (batch {i}), retry {attempt+1}/9 em {wait}s: {str(e)[:110]}", flush=True)
                time.sleep(wait)

    print("Ingestão concluída.")


if __name__ == "__main__":
    main()

"""Reindexa uma área (e5-large, chunk de 1000 chars) numa
coleção nova BGE-M3 + late chunking, embedando no app Modal
`rag-embed-bgem3` (scripts/modal_embed.py).

Não relê fonte (legenda/PDF): reconstrói o documento a partir dos chunks
que já estão no Qdrant de origem (removendo o overlap de 150 chars do
splitter antigo). Sintéticos (resumo/qa) são reembedados um a um, sem
late chunking.

Uso (QDRANT_URL / QDRANT_CF_ACCESS_TOKEN no ambiente):
    python scripts/reindex_area.py --area constitucional --dry-run --limit-videos 20
    python scripts/reindex_area.py --area constitucional --limit-videos 20
    python scripts/reindex_area.py --area constitucional --resume

`--limit-videos N` corta a LISTA ordenada (determinística) antes de pular os
já feitos: `--resume --limit-videos 20` depois do piloto não faz nada novo.
"""
import argparse
import itertools
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from qdrant_client.http import models as qmodels  # noqa: E402
from langchain_core.documents import Document  # noqa: E402

from config import LATE_CHUNK_MAX_TOKENS  # noqa: E402
from ingest.ingest import build_documents_late_chunk, late_chunk_points  # noqa: E402
from rag.query import SINTETICO  # noqa: E402
from rag.vectorstore import _create_collection, get_client, get_sparse_embeddings  # noqa: E402

BGE_M3_DIM = 1024
OVERLAP_MIN, OVERLAP_MAX = 20, 200  # CHUNK_OVERLAP antigo = 150 chars
UPSERT_BATCH = 256
DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def _flt(**kw) -> qmodels.Filter:
    return qmodels.Filter(must=[
        qmodels.FieldCondition(key=f"metadata.{k}", match=qmodels.MatchValue(value=v))
        for k, v in kw.items()])


def strip_overlap(prev: str, cur: str) -> str:
    """Tira de `cur` o prefixo que repete o fim de `prev` (maior sufixo/prefixo
    comum entre OVERLAP_MIN e OVERLAP_MAX chars). Sem match, devolve `cur`."""
    for k in range(min(OVERLAP_MAX, len(prev), len(cur)), OVERLAP_MIN - 1, -1):
        if prev.endswith(cur[:k]):
            return cur[k:].lstrip()
    return cur


def reconstruct(points) -> tuple[dict, int]:
    """Chunks primários de UM video_id -> registro no formato de
    build_documents_late_chunk (`text == ' '.join(seg["text"])`, contrato de
    find_time_for_offset). Devolve (registro, nº de overlaps removidos)."""
    by_chunk = {}
    for p in points:  # chunk duplicado (ingestão repetida): fica o primeiro
        by_chunk.setdefault(int(float(p.payload["metadata"].get("chunk", 0))), p)
    ordered = [by_chunk[k] for k in sorted(by_chunk)]
    segments, prev, stripped = [], None, 0
    for p in ordered:
        text = p.payload.get("page_content") or ""
        seg = text if prev is None else strip_overlap(prev, text)
        stripped += seg != text
        prev = text
        if seg:
            # start=0 em tudo (legenda do YouTube) continua 0: não inventa timestamp
            segments.append({"start": float(p.payload["metadata"].get("start") or 0.0), "text": seg})
    m0 = ordered[0].payload["metadata"]
    record = {k: m0.get(k) for k in
              ("video_id", "title", "channel", "channel_id", "url", "source_type", "area", "module")}
    record["text"] = " ".join(s["text"] for s in segments)
    # material/texto não tem tempo: sem segments, url_at = url (sem &t=)
    record["segments"] = segments if record["source_type"] == "video" else []
    return record, stripped


def parse_source_video_id(vid: str, known: set[str]) -> str | None:
    """`syn-resumo-<id>` / `syn-qa-[<canal>-]<id>` -> <id>, só se <id> for um
    video_id primário que existe na área (senão None: não chuta)."""
    for prefix in ("syn-resumo-", "syn-qa-"):
        if vid.startswith(prefix):
            rest = vid[len(prefix):]
            cands = [rest] + [rest[i + 1:] for i, ch in enumerate(rest) if ch == "-"]
            return next((c for c in cands if c in known), None)
    return None


def synthetic_docs(points, vectors, known: set[str]) -> list[Document]:
    docs = []
    for p, vec in zip(points, vectors):
        m = dict(p.payload["metadata"])
        m["chunk"] = int(float(m.get("chunk") or 0))    # origem tem string em parte dos sintéticos
        m["start"] = float(m.get("start") or 0.0)
        m["synthetic"] = True
        src = parse_source_video_id(m["video_id"], known)
        if src:
            m["source_video_id"] = src
        m["embedding_vector"] = vec  # consumido e removido por late_chunk_points
        docs.append(Document(page_content=p.payload.get("page_content") or "", metadata=m))
    return docs


class _Precomputed:
    """Encoder de mentira pro build_documents_late_chunk: os chunks já vieram
    do Modal, só devolve."""

    def __init__(self, chunks):
        self.chunks = chunks

    def encode_document(self, text, max_chunk_tokens=None):
        return [(c["text"], c["char_pos"], c["vector"]) for c in self.chunks]


def list_videos(client, source: str, area: str) -> tuple[list[str], set[str]]:
    """video_ids da área em ordem determinística e misturada (round-robin por
    canal x primário/sintético). Devolve também o set de ids primários."""
    chans = client.facet(source, key="metadata.channel_id", facet_filter=_flt(area=area),
                         limit=1000, exact=True).hits
    syn = qmodels.FieldCondition(key="metadata.source_type", match=qmodels.MatchAny(any=list(SINTETICO)))
    groups, primary = [], set()
    for ch in sorted(h.value for h in chans):
        base = _flt(area=area, channel_id=ch).must
        for is_syn in (False, True):
            flt = qmodels.Filter(must=base + [syn]) if is_syn else qmodels.Filter(must=base, must_not=[syn])
            ids = sorted(h.value for h in client.facet(
                source, key="metadata.video_id", facet_filter=flt, limit=1_000_000, exact=True).hits)
            groups.append(ids)
            if not is_syn:
                primary.update(ids)
    order = [v for tup in itertools.zip_longest(*groups) for v in tup if v]
    return list(dict.fromkeys(order)), primary


def fetch_points(client, source: str, area: str, vid: str):
    pts, offset = [], None
    while True:
        batch, offset = client.scroll(source, scroll_filter=_flt(area=area, video_id=vid),
                                      limit=1024, offset=offset, with_payload=True, with_vectors=False)
        pts += batch
        if offset is None:
            return pts


def write_video(client, target: str, area: str, vid: str, rec, syn, chunks, syn_vecs,
                 sparse, ckpt: Path, primary: set[str]) -> int:
    """Corpo por vídeo: monta os pontos, substitui no destino (delete+upsert) e
    marca no checkpoint. Levanta ValueError sem marcar checkpoint se `rec` tinha
    texto reconstruído de verdade mas o Modal não devolveu nenhum chunk — assim
    cai no failures/--resume em vez de sumir silenciosamente."""
    if rec and rec["text"].strip() and not chunks:
        raise ValueError(f"0 chunks retornados pro vídeo {vid}")
    docs = build_documents_late_chunk(rec, _Precomputed(chunks)) if rec else []
    docs += synthetic_docs(syn, syn_vecs, primary)
    points = late_chunk_points(docs, sparse) if docs else []
    # idempotência: reprocessar substitui os pontos do vídeo no destino
    client.delete(target, points_selector=qmodels.FilterSelector(
        filter=_flt(area=area, video_id=vid)), wait=True)
    for i in range(0, len(points), UPSERT_BATCH):
        client.upsert(target, points=points[i:i + UPSERT_BATCH], wait=True)
    with ckpt.open("a") as f:
        f.write(vid + "\n")
    return len(points)


def embed(embedder, items: dict) -> dict:
    """items {vid: (record|None, syn_points)} -> {vid: (chunks, syn_vectors)}
    numa chamada remota por método."""
    docs = [{"id": v, "text": rec["text"]} for v, (rec, _) in items.items() if rec]
    texts = [p.payload.get("page_content") or "" for _, (_, syn) in items.items() for p in syn]
    chunks = {d["id"]: d["chunks"] for d in embedder.embed_documents.remote(
        docs, max_chunk_tokens=LATE_CHUNK_MAX_TOKENS)} if docs else {}
    vecs = iter(embedder.embed_texts.remote(texts) if texts else [])
    return {v: (chunks.get(v, []), [next(vecs) for _ in syn]) for v, (_, syn) in items.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--area", required=True)
    ap.add_argument("--source", default="legal_rag")
    ap.add_argument("--target", default="legal_rag_bgem3")
    ap.add_argument("--limit-videos", type=int, default=None)
    ap.add_argument("--resume", action="store_true", help="pula video_ids já no checkpoint")
    ap.add_argument("--dry-run", action="store_true", help="só reconstrói e conta (sem Modal, sem escrita)")
    ap.add_argument("--batch-videos", type=int, default=10, help="vídeos por chamada ao Modal")
    ap.add_argument("--checkpoint", type=Path, default=None)
    args = ap.parse_args()
    if args.target == args.source:
        sys.exit("--target igual a --source: a origem é somente leitura.")
    if not args.target.endswith("_bgem3"):
        sys.exit(f"--target {args.target!r} não termina em '_bgem3': recusando gravar (delete+upsert) "
                 f"numa coleção que não é claramente um destino de reindex BGE-M3 — protege coleções "
                 f"que não são destino de reindex (legal_rag, youtube_rag, ...) de apagamento silencioso.")
    ckpt = args.checkpoint or DATA_DIR / f"reindex-{args.area}-{args.target}.done"

    client = get_client()
    videos, primary = list_videos(client, args.source, args.area)
    todo = videos[:args.limit_videos] if args.limit_videos else videos
    done = set(ckpt.read_text().split()) if args.resume and ckpt.exists() else set()
    pending = [v for v in todo if v not in done]
    print(f"área={args.area}: {len(videos)} video_ids; selecionados {len(todo)}; "
          f"já feitos {len(todo) - len(pending)}; a processar {len(pending)}"
          f"{' [DRY-RUN]' if args.dry_run else ''}", flush=True)

    embedder = sparse = None
    if not args.dry_run and pending:
        import modal
        embedder = modal.Cls.from_name("rag-embed-bgem3", "Embedder")()
        sparse = get_sparse_embeddings()
        if not client.collection_exists(args.target):
            _create_collection(client, BGE_M3_DIM, name=args.target)
        ckpt.parent.mkdir(parents=True, exist_ok=True)

    failures, n_points, n_src, embed_secs, t0 = [], 0, 0, 0.0, time.time()
    for bi in range(0, len(pending), args.batch_videos):
        items = {}
        for vid in pending[bi:bi + args.batch_videos]:
            try:
                pts = fetch_points(client, args.source, args.area, vid)
                prim = [p for p in pts if p.payload["metadata"].get("source_type") not in SINTETICO]
                syn = [p for p in pts if p.payload["metadata"].get("source_type") in SINTETICO]
                rec, stripped = reconstruct(prim) if prim else (None, 0)
                items[vid] = (rec, syn)
                n_src += len(pts)
                zero = rec and rec["segments"] and all(s["start"] == 0 for s in rec["segments"])
                print(f"  {vid}: {len(prim)} prim ({stripped}/{max(len(prim) - 1, 0)} overlaps, "
                      f"{len(rec['text']) if rec else 0} chars{', start=0' if zero else ''}) + {len(syn)} sint",
                      flush=True)
            except Exception as e:  # noqa: BLE001 — um vídeo não derruba o lote
                failures.append((vid, f"reconstrução: {e}"))
        if args.dry_run or not items:
            continue

        t = time.time()
        try:
            results = embed(embedder, items)
        except Exception as e:  # noqa: BLE001 — lote falhou: tenta vídeo a vídeo
            print(f"  ! lote falhou no Modal ({str(e)[:120]}); refazendo vídeo a vídeo", flush=True)
            results = {}
            for vid, item in items.items():
                try:
                    results.update(embed(embedder, {vid: item}))
                except Exception as e2:  # noqa: BLE001
                    failures.append((vid, f"modal: {str(e2)[:200]}"))
        embed_secs += time.time() - t

        for vid, (chunks, syn_vecs) in results.items():
            rec, syn = items[vid]
            try:
                n = write_video(client, args.target, args.area, vid, rec, syn, chunks, syn_vecs,
                                 sparse, ckpt, primary)
                n_points += n
                print(f"  ok {vid}: {len(chunks)} chunks late + {len(syn_vecs)} sint -> {n} pontos",
                      flush=True)
            except Exception as e:  # noqa: BLE001
                failures.append((vid, f"escrita: {str(e)[:200]}"))
        print(f"[{min(bi + args.batch_videos, len(pending))}/{len(pending)}] {n_points} pontos, "
              f"{time.time() - t0:.0f}s total, {embed_secs:.0f}s no Modal", flush=True)

    print(f"FIM: {len(pending) - len(failures)} ok, {len(failures)} falhas; {n_src} pontos lidos da origem, "
          f"{n_points} escritos em {args.target}; {time.time() - t0:.0f}s total, {embed_secs:.0f}s no Modal")
    for vid, err in failures:
        print(f"  FALHA {vid}: {err}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()

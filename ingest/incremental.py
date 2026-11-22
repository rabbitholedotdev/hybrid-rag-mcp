"""Ingestão INCREMENTAL genérica e resumível para qualquer canal já materializado
em data/<channel_dir>/*.json.

Consulta a nuvem (Qdrant) pela `area` e ingere SOMENTE os video_id que ainda não
existem. Idempotente/resumível: rode quantas vezes quiser, em quantas máquinas
quiser (com --no-replace); cada uma pega o que falta. Ignora tudo já ingerido.

Uso:
  python -m ingest.incremental --dir legal --area constitucional --dry-run
  python -m ingest.incremental --dir legal --area constitucional
  # só conferir cada record contra a coleção e gravar o estado no ledger (sai 1 se faltar algo):
  python -m ingest.incremental --dir legal --area constitucional --verify
  # opcional: processar só uma fatia (para distribuir entre máquinas):
  python -m ingest.incremental --dir legal --area constitucional --shard 0/3
"""
import argparse, glob, json, os, shutil, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))
from rag.vectorstore import get_client  # noqa: E402
from config import COLLECTION_NAME, EMBED_MODEL  # noqa: E402
from ingest.ingest import expected_chunks, make_splitter  # noqa: E402
from qdrant_client.http import models as qm  # noqa: E402


FACET_LIMIT = 1_000_000


def _flt(**kv) -> qm.Filter:
    return qm.Filter(must=[qm.FieldCondition(key=f"metadata.{k}", match=qm.MatchValue(value=v))
                           for k, v in kv.items()])


def _facet(flt: qm.Filter) -> dict:
    fac = get_client().facet(collection_name=COLLECTION_NAME, key="metadata.video_id",
                             facet_filter=flt, limit=FACET_LIMIT, exact=True)
    if len(fac.hits) >= FACET_LIMIT:
        raise SystemExit(f"passou de {FACET_LIMIT} vídeos: facet truncado")
    return {h.value: h.count for h in fac.hits}


def cloud_counts(area: str, channel_ids) -> tuple[set, dict]:
    """(video_ids na área, {(channel_id, video_id): nº de pontos}). A presença é
    por área (como sempre foi); a contagem é por canal, porque o mesmo video_id
    pode existir em dois canais da mesma área."""
    presentes = set(_facet(_flt(area=area)))
    counts = {(ch, v): n for ch in channel_ids for v, n in _facet(_flt(area=area, channel_id=ch)).items()}
    return presentes, counts


def pendentes(records: dict, presentes: set, counts: dict, esperado) -> tuple[list, list]:
    """(faltam, parciais). Parcial = no índice, no canal do record, com menos
    pontos que o record gera (upsert interrompido no meio); `esperado=None` ou
    record sem channel_id pulam essa conferência."""
    faltam, parciais = [], []
    for f, rec in records.items():
        vid, ch = rec["video_id"], rec.get("channel_id")
        if vid not in presentes:
            faltam.append((f, vid))
        elif esperado and ch and counts.get((ch, vid), 0) < esperado(rec):
            parciais.append((f, vid))
    return faltam, parciais


def classificar(records: dict, presentes: set, counts: dict, esperado) -> list[dict]:
    """Estado de CADA record na coleção: ok | parcial | faltando | vazio. `vazio` = o record gera 0 chunks
    (texto vazio): nunca vai aparecer na coleção, então não é "faltando" e não volta a cada rodada."""
    faltam, parciais = pendentes(records, presentes, counts, esperado)
    falta, parcial = {f for f, _ in faltam}, {f for f, _ in parciais}
    out = []
    for f, rec in records.items():
        vid, ch = rec["video_id"], rec.get("channel_id") or ""
        esp = esperado(rec) if esperado else None
        if f in parcial:
            st = "parcial"
        elif f in falta:
            st = "vazio" if esp == 0 else "faltando"
        else:
            st = "ok"
        out.append({"f": f, "video_id": vid, "canal": ch, "status": st, "esperado": esp,
                    "na_colecao": counts.get((ch, vid), 0) if vid in presentes else 0})
    return out


def apaga_parcial(client, area: str, rec: dict) -> None:
    client.delete(collection_name=COLLECTION_NAME, wait=True, points_selector=qm.FilterSelector(
        filter=_flt(area=area, channel_id=rec["channel_id"], video_id=rec["video_id"])))


class _LedgerBestEffort:
    """Ledger de auditoria (ingest/ledger.py) que nunca derruba a ingestão: erro vira aviso. Abre só na
    primeira gravação, então --dry-run não cria rodada."""

    def __init__(self, args, backend: str, modo: str):
        self.args, self.backend, self.modo, self.con, self.rid = args, backend, modo, None, None

    def itens(self, estados: list[dict], tocados: set, falhou: int | None = None) -> None:
        try:
            from ingest import ledger
            if self.con is None:
                self.con = ledger.abrir()
                self.rid = ledger.iniciar_rodada(self.con, COLLECTION_NAME, self.args.dir, self.args.area,
                                                 EMBED_MODEL, self.backend, self.modo)
            itens = []
            for e in estados:
                st, motivo, tocado = e["status"], None, e["f"] in tocados
                if tocado and falhou and st in ("faltando", "parcial"):
                    st, motivo = "falhou", f"ingest.ingest saiu {falhou}"
                itens.append({"video_id": e["video_id"], "canal": e["canal"], "status": st, "motivo": motivo,
                              "esperado": e["esperado"], "na_colecao": e["na_colecao"], "tocado": tocado})
            ledger.gravar_itens(self.con, self.rid, COLLECTION_NAME, self.args.dir, self.args.area, itens,
                                EMBED_MODEL, self.backend)
        except Exception as ex:  # noqa: BLE001
            print(f"aviso: ledger não gravou: {ex}", file=sys.stderr)

    def fecha(self, rc: int) -> None:
        try:
            if self.con is not None:
                from ingest import ledger
                ledger.fechar_rodada(self.con, self.rid, rc)
        except Exception as ex:  # noqa: BLE001
            print(f"aviso: ledger não fechou a rodada: {ex}", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="pasta em data/ com os records")
    ap.add_argument("--area", required=True)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--shard", default=None, help="i/N — processa só a fatia i de N (distribuir)")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--verify", action="store_true",
                    help="só confere cada record contra a coleção e grava o estado no ledger; sai 1 se faltar algo")
    args = ap.parse_args()

    src = ROOT / "data" / args.dir
    files = sorted(f for f in src.glob("*.json") if not f.name.startswith("._"))
    records = {}
    for f in files:
        try:
            rec = json.loads(f.read_text(encoding="utf-8", errors="replace"))
            rec["video_id"]
        except Exception:
            continue
        records[f] = rec
    print(f"records locais: {len(records)}")

    channel_ids = {r["channel_id"] for r in records.values() if r.get("channel_id")}
    presentes, counts = cloud_counts(args.area, channel_ids)
    print(f"já na nuvem (area {args.area}): {len(presentes)}")
    splitter = make_splitter()
    if EMBED_MODEL == "BAAI/bge-m3":
        esperado = None
        print("aviso: late chunking (bge-m3) não confere vídeo parcial, só presença")
    else:
        memo = {}

        def esperado(r):  # divide o texto uma vez só por record (pendentes e classificar o chamam)
            k = (r["video_id"], r.get("channel_id"))
            if k not in memo:
                memo[k] = expected_chunks(r, splitter)
            return memo[k]
    estados = classificar(records, presentes, counts, esperado)
    por_f = {e["f"]: e for e in estados}
    parciais = [(f, e["video_id"]) for f, e in por_f.items() if e["status"] == "parcial"]
    for f, vid in parciais:
        ch = records[f]["channel_id"]
        print(f"   parcial: {vid} ({counts.get((ch, vid), 0)} de {esperado(records[f])} pontos)")
    missing = [(f, e["video_id"]) for f, e in por_f.items() if e["status"] in ("faltando", "parcial")]
    vazios = sum(1 for e in estados if e["status"] == "vazio")
    if vazios:
        print(f"vazios (0 chunks, não entram): {vazios}")
    backend = os.getenv("EMBED_BACKEND", "local")
    ledger = _LedgerBestEffort(args, backend, "verify" if args.verify else "ingest")

    if args.verify:
        ledger.itens(estados, tocados=set())
        ledger.fecha(0 if not missing else 1)
        cont = {}
        for e in estados:
            cont[e["status"]] = cont.get(e["status"], 0) + 1
        print("verify:", ", ".join(f"{k}={v}" for k, v in sorted(cont.items())))
        sys.exit(1 if missing else 0)

    if args.shard:
        i, n = map(int, args.shard.split("/"))
        missing = [x for k, x in enumerate(missing) if k % n == i]
        print(f"shard {i}/{n}")
    print(f"FALTAM ingerir: {len(missing)}")

    if args.dry_run:
        for f, vid in missing[:5]:
            print("   -", f.name)
        print("dry-run: nada ingerido.")
        return
    if not missing:
        ledger.itens(estados, tocados=set())
        ledger.fecha(0)
        print("nada a fazer — já completo.")
        return

    # staging só dos que faltam (nome único por dir+shard)
    tag = f"{args.dir}-up" + (f"-{args.shard.replace('/','of')}" if args.shard else "")
    stage = ROOT / "data" / tag
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    for f, vid in missing:
        shutil.copy(f, stage / f.name)
    # parcial pode ter pontos com ID aleatório (antes do point_id): apaga os
    # pontos do vídeo e reenvia o record inteiro, que está no staging
    c = get_client()
    for f, vid in (x for x in parciais if x in missing):  # só os desta shard
        apaga_parcial(c, args.area, records[f])

    print(f"ingerindo {len(missing)} (staging '{tag}')...")
    rc = subprocess.run([sys.executable, "-m", "ingest.ingest", "--channel", tag,
                         "--no-replace", "--batch", str(args.batch)], cwd=str(ROOT)).returncode
    tocados = {f for f, _ in missing}
    if rc:
        ledger.itens(estados, tocados=tocados, falhou=rc)
        ledger.fecha(rc)
        sys.exit(f"ingest.ingest saiu {rc}: records ficam em data/{args.dir}")
    # o Qdrant confirma o que acabou de subir: o ledger grava o estado lido de lá, não o que se esperava
    presentes2, counts2 = cloud_counts(args.area, channel_ids)
    ledger.itens(classificar(records, presentes2, counts2, esperado), tocados=tocados)
    ledger.fecha(0)
    shutil.rmtree(stage, ignore_errors=True)  # o staging só serve à rodada; deixá-lo gerava -up-up
    print("incremental concluído.")


if __name__ == "__main__":
    main()

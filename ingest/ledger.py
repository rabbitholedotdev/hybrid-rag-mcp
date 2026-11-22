"""Registro de AUDITORIA da ingestão (SQLite, stdlib): por aula, em que estado ficou, com que modelo e
backend, em que rodada e quando; por rodada, o que rodou e como terminou.

NÃO decide o que ingerir: quem decide é o Qdrant (ingest.incremental compara os video_id e a contagem de
pontos da coleção). O ledger guarda o que o Qdrant não guarda: falha, record vazio (0 chunks) e proveniência.
Nunca derruba a ingestão: quem o chama trata erro como aviso. Reconstruível: `ingest.incremental --verify`
regrava o estado a partir da coleção.

  python -m ingest.ledger                 # resumo por coleção e status
  python -m ingest.ledger --falhas        # lista as aulas que falharam ou ficaram parciais
"""
import argparse
import os
import socket
import sqlite3
import subprocess
import time
from pathlib import Path

CAMINHO = Path(os.getenv("RAG_LEDGER") or Path.home() / ".local/share/rag-ledger/ledger.sqlite")
STATUS = ("ok", "parcial", "faltando", "vazio", "falhou")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS rodada (
  id INTEGER PRIMARY KEY, inicio TEXT, fim TEXT, colecao TEXT, pasta TEXT, area TEXT, host TEXT,
  commit_rag TEXT, embed_model TEXT, embed_backend TEXT, modo TEXT, rc INTEGER);
CREATE TABLE IF NOT EXISTS item (
  colecao TEXT NOT NULL, video_id TEXT NOT NULL, canal TEXT NOT NULL DEFAULT '', area TEXT, pasta TEXT,
  status TEXT NOT NULL CHECK (status IN ('ok','parcial','faltando','vazio','falhou')),
  esperado INTEGER, na_colecao INTEGER, rodada_id INTEGER REFERENCES rodada(id),
  embed_model TEXT, embed_backend TEXT, motivo TEXT, atualizado_em TEXT,
  PRIMARY KEY (colecao, video_id, canal));
"""


def agora() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def abrir(caminho: Path | None = None) -> sqlite3.Connection:
    caminho = Path(caminho or CAMINHO)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(caminho, timeout=30)
    con.execute("PRAGMA journal_mode=WAL")
    con.executescript(_SCHEMA)
    return con


def _commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                              cwd=Path(__file__).resolve().parent.parent).stdout.strip()
    except Exception:  # noqa: BLE001 — proveniência é best-effort
        return ""


def iniciar_rodada(con, colecao: str, pasta: str, area: str, embed_model: str, backend: str, modo: str) -> int:
    cur = con.execute(
        "INSERT INTO rodada (inicio, colecao, pasta, area, host, commit_rag, embed_model, embed_backend, modo)"
        " VALUES (?,?,?,?,?,?,?,?,?)",
        (agora(), colecao, pasta, area, socket.gethostname(), _commit(), embed_model, backend, modo))
    con.commit()
    return cur.lastrowid


def fechar_rodada(con, rodada_id: int, rc: int) -> None:
    con.execute("UPDATE rodada SET fim=?, rc=? WHERE id=?", (agora(), rc, rodada_id))
    con.commit()


def gravar_itens(con, rodada_id: int, colecao: str, pasta: str, area: str, itens: list[dict],
                 embed_model: str, backend: str) -> None:
    """itens: {video_id, canal, status, esperado, na_colecao, motivo, tocado}. `tocado` = esta rodada
    (re)ingeriu a aula: só então modelo, backend e rodada de proveniência são sobrescritos."""
    for it in itens:
        assert it["status"] in STATUS, it["status"]
        con.execute(
            "INSERT INTO item (colecao, video_id, canal, area, pasta, status, esperado, na_colecao, rodada_id,"
            " embed_model, embed_backend, motivo, atualizado_em) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)"
            " ON CONFLICT (colecao, video_id, canal) DO UPDATE SET status=excluded.status,"
            " esperado=excluded.esperado, na_colecao=excluded.na_colecao, motivo=excluded.motivo,"
            " area=excluded.area, pasta=excluded.pasta, atualizado_em=excluded.atualizado_em,"
            " rodada_id=CASE WHEN ? THEN excluded.rodada_id ELSE item.rodada_id END,"
            " embed_model=CASE WHEN ? THEN excluded.embed_model ELSE item.embed_model END,"
            " embed_backend=CASE WHEN ? THEN excluded.embed_backend ELSE item.embed_backend END",
            (colecao, it["video_id"], it.get("canal") or "", area, pasta, it["status"], it.get("esperado"),
             it.get("na_colecao"), rodada_id, embed_model, backend, it.get("motivo"), agora(),
             bool(it.get("tocado")), bool(it.get("tocado")), bool(it.get("tocado"))))
    con.commit()


def resumo(con, colecao: str | None = None) -> list[tuple]:
    q = "SELECT colecao, status, COUNT(*) FROM item" + (" WHERE colecao=?" if colecao else "") \
        + " GROUP BY colecao, status ORDER BY colecao, status"
    return con.execute(q, (colecao,) if colecao else ()).fetchall()


def problemas(con, colecao: str | None = None) -> list[tuple]:
    q = ("SELECT colecao, pasta, video_id, status, esperado, na_colecao, motivo FROM item"
         " WHERE status IN ('falhou','parcial','faltando')" + (" AND colecao=?" if colecao else "")
         + " ORDER BY colecao, pasta, video_id")
    return con.execute(q, (colecao,) if colecao else ()).fetchall()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--colecao")
    ap.add_argument("--falhas", action="store_true")
    a = ap.parse_args()
    if not CAMINHO.exists():
        print(f"sem ledger em {CAMINHO}: nenhuma ingestão registrada ainda")
        return 0
    con = abrir()
    if a.falhas:
        for r in problemas(con, a.colecao):
            print("\t".join("" if x is None else str(x) for x in r))
        return 0
    for col, st, n in resumo(con, a.colecao):
        print(f"{col}\t{st}\t{n}")
    for r in con.execute("SELECT id, inicio, colecao, pasta, modo, embed_backend, rc FROM rodada"
                         " ORDER BY id DESC LIMIT 5"):
        print("rodada", *r, sep="\t")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

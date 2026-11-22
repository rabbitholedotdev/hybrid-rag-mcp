"""Scanner de segredos p/ o pipeline de ingest.

Barra documentos com credenciais em texto puro ANTES de embedar/upsertar
(credentials in source text must not be embedded).

Uso como lib:  from ingest.secret_scan import scan_text, scan_record
Uso como CLI:  python -m ingest.secret_scan <dir-de-records>   (relatório; exit 1 se achar)
               python -m ingest.secret_scan <dir> --quarantine  (move os suspeitos p/ _quarentena/)
"""
import argparse
import glob
import json
import os
import re
import shutil

# (nome, regex). Foco em segredos ACIONÁVEIS, não em menções genéricas.
PATTERNS = [
    ("openai_key", re.compile(r"\bsk-(?:proj|ant|live)?-?[A-Za-z0-9_-]{20,}")),
    ("stripe_pagarme_live", re.compile(r"\b(?:sk|pk|rk)_live_[A-Za-z0-9]{16,}")),
    ("stripe_test", re.compile(r"\b(?:sk|pk)_test_[A-Za-z0-9]{16,}")),
    ("aws_akid", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("gh_token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{36,}")),
    ("google_api", re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b")),
    ("slack_token", re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}")),
    ("jwt", re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}")),
    ("private_key", re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA |PGP )?PRIVATE KEY-----")),
    ("db_url_creds", re.compile(r"\b(?:mysql|postgres|postgresql|mongodb(?:\+srv)?|redis)://[^\s:@/]+:[^\s:@/]+@")),
    ("cf_tunnel_token", re.compile(r"\beyJhIjoi[A-Za-z0-9+/=]{40,}")),
    ("generic_secret_kv", re.compile(
        r"(?i)\b(?:password|senha|passwd|secret|api[_-]?key|access[_-]?token|client[_-]?secret)"
        r"\s*[:=]\s*['\"]?[A-Za-z0-9!@#$%^&*_\-./+]{8,}")),
]

# reduz falso-positivo: linhas de placeholder/exemplo
_PLACEHOLDER = re.compile(r"(?i)(exemplo|example|placeholder|troque|your[_-]?|xxxx|\.\.\.|<[^>]+>|sk-\.\.\.)")


def scan_text(text: str):
    """Retorna lista de (tipo, trecho_mascarado) dos segredos encontrados."""
    hits = []
    for name, rx in PATTERNS:
        for m in rx.finditer(text or ""):
            s = m.group(0)
            # janela p/ checar placeholder
            i = m.start()
            ctx = text[max(0, i - 20): i + len(s) + 5]
            if _PLACEHOLDER.search(ctx) and name == "generic_secret_kv":
                continue
            masked = s[:6] + "…" + s[-3:] if len(s) > 14 else s[:3] + "…"
            hits.append((name, masked))
    # dedup
    seen = set(); out = []
    for h in hits:
        if h not in seen:
            seen.add(h); out.append(h)
    return out


def scan_record(path: str):
    try:
        d = json.loads(open(path, encoding="utf-8", errors="replace").read())
    except Exception:  # noqa: BLE001
        return []
    return scan_text(d.get("text", ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("records_dir")
    ap.add_argument("--quarantine", action="store_true", help="move suspeitos p/ <dir>/_quarentena")
    a = ap.parse_args()
    qdir = os.path.join(a.records_dir, "_quarentena")
    flagged = 0
    for f in sorted(glob.glob(a.records_dir + "/*.json")):
        hits = scan_record(f)
        if hits:
            flagged += 1
            print(f"⚠️  {os.path.basename(f)}: {hits}")
            if a.quarantine:
                os.makedirs(qdir, exist_ok=True)
                shutil.move(f, os.path.join(qdir, os.path.basename(f)))
    if flagged:
        print(f"\n{flagged} record(s) com possíveis segredos"
              + (f" → movidos p/ {qdir}" if a.quarantine else " (use --quarantine p/ isolar)"))
        raise SystemExit(1)
    print("OK — nenhum segredo detectado.")


if __name__ == "__main__":
    main()

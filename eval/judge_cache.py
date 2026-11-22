"""Cache de julgamentos em disco, formato JSONL append-only (uma escrita por
chamada, sem reescrever o arquivo inteiro — corte de energia no meio de um
run de horas não perde as linhas já gravadas). Chave = hash de
(pergunta, variante, prompt_hash, modelo): mesma pergunta+config+modelo nunca
paga duas vezes o juiz, então um run interrompido é retomável.

ponytail: sem compactação — arquivo só cresce (linha repetida = a última
prevalece, carregada por último no load()). Nesta escala (milhares de linhas)
não compensa comprimir; se crescer demais, reescrever de vez em quando com
`{k: v for k,v in entradas}` já resolve.
"""
import hashlib
import json
import threading
from pathlib import Path
from typing import Optional


class JudgeCache:
    def __init__(self, path: Path):
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._mem: dict[str, float] = {}
        if self._path.exists():
            for ln in self._path.read_text(encoding="utf-8").splitlines():
                ln = ln.strip()
                if not ln:
                    continue
                d = json.loads(ln)
                self._mem[d["k"]] = d["v"]

    @staticmethod
    def key(question_id: str, variant: str, prompt_hash: str, model: str) -> str:
        raw = "|".join([question_id, variant, prompt_hash, model])
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def get(self, key: str) -> Optional[float]:
        return self._mem.get(key)

    def set(self, key: str, score: float) -> None:
        with self._lock:
            self._mem[key] = score
            with self._path.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"k": key, "v": score}) + "\n")

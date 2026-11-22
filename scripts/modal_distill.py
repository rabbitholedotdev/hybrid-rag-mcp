"""App Modal `rag-distill`: destilação (resumo + Q&A) de transcrições de aula
numa GPU sob demanda, via vLLM. Gera a CAMADA SINTÉTICA (source_type
`resumo`/`qa`) que `mcp_server.obter_resumo`/`estudar_tema` consomem.

LLM na GPU (Modal) só gera o TEXTO do resumo/qa; o embedding continua no
attivorag (e5-large), via `ingest.ingest` normal — este app não embedda.

Sem endpoint HTTP: uso interno via SDK
(`modal.Cls.from_name("rag-distill", "Distiller")` ou `modal run`).

Deploy:  modal deploy scripts/modal_distill.py
Smoke:   modal run scripts/modal_distill.py
"""
import json

import modal

MODEL = "Qwen/Qwen2.5-7B-Instruct"
HF_HOME = "/cache/hf"
MAX_LEN = 16384

image = (
    modal.Image.debian_slim(python_version="3.11")
    # vllm 0.6.6 traz torch 2.5.1; transformers >=4.48 exige torch.accelerator
    # (torch 2.6) -> "module 'torch' has no attribute 'accelerator'" no boot.
    # 4.47.1 é o par validado com vllm 0.6.6 / torch 2.5.1.
    .pip_install("vllm==0.6.6", "transformers==4.47.1")
    .env({"HF_HOME": HF_HOME, "VLLM_WORKER_MULTIPROC_METHOD": "spawn"})
)

model_volume = modal.Volume.from_name("qwen25-7b-cache", create_if_missing=True)
app = modal.App("rag-distill", image=image)

# ponytail: um prompt só, pedindo JSON {resumo, qa}. Resumo = método em
# bullets com frameworks nomeados; qa = pares P:/R:. Avisa o modelo que é
# material de estudo (pode impor estrutura), espelhando _AVISO_SINTETICO.
SYS = (
    "Você destila transcrições de aulas de negócios/marketing em português. "
    "Produz material de ESTUDO (não citação literal). Responda SEMPRE com um "
    "único objeto JSON válido, sem markdown, sem texto fora do JSON."
)
PROMPT = """\
Aula: {title}

Transcrição (pode ter ruído de ASR):
\"\"\"
{text}
\"\"\"

Gere um objeto JSON com exatamente duas chaves:

"resumo": uma STRING ÚNICA (não objeto, não lista) com o MÉTODO da aula em
bullets. Cada bullet numa linha começando com "- ", linhas separadas por \\n.
6 a 14 bullets. Nomeie frameworks/etapas. Não invente números.

"qa": uma STRING ÚNICA (não objeto, não lista) com 4 a 8 pares, formato exato
"P: <pergunta>\\nR: <resposta>", pares separados por linha em branco (\\n\\n).

Ambos os valores são strings de texto puro. Responda só o JSON.
"""


# Domínio jurídico (concurso): chave "domain": "juridico" no item. Sem a chave, vale o prompt de negócios acima.
SYS_JUR = (
    "Você destila aulas e materiais de Direito para concurso público (AGU, procuradorias) em português. "
    "Produz material de ESTUDO fiel ao texto: não invente artigo, súmula, tema ou número. A transcrição pode ter "
    "ruído de reconhecimento de voz em números de artigos; quando duvidoso, omita o número. Responda SEMPRE com um "
    "único objeto JSON válido, sem markdown, sem texto fora do JSON."
)
PROMPT_JUR = """\
Título: {title}

Texto (aula transcrita ou material de apoio):
\"\"\"
{text}
\"\"\"

Gere um objeto JSON com exatamente duas chaves:

"resumo": uma STRING ÚNICA com as TESES e REGRAS do trecho em bullets. Cada bullet numa linha começando com "- ",
linhas separadas por \\n. 6 a 14 bullets: conceito, requisitos, prazos, competência, procedimento, exceções,
normas e julgados citados (só os que o texto cita).

"qa": uma STRING ÚNICA com 4 a 8 pares no estilo de questão de prova, formato exato
"P: <pergunta>\\nR: <resposta>", pares separados por linha em branco (\\n\\n).

Ambos os valores são strings de texto puro. Responda só o JSON.
"""


def _as_text(v) -> str:
    """Coage o valor (str | dict com bullets | lista) em texto limpo."""
    if isinstance(v, str):
        return v.strip()
    if isinstance(v, list):
        return "\n".join(_as_text(x) for x in v).strip()
    if isinstance(v, dict):
        # modelo às vezes devolve {"-": "bullet", ...} ou {"P":..,"R":..}
        parts = []
        for k, val in v.items():
            s = _as_text(val)
            parts.append(f"- {s}" if k in ("-", "*", "•") else
                         (f"{k}: {s}" if k in ("P", "R", "pergunta", "resposta") else s))
        return "\n".join(parts).strip()
    return str(v).strip() if v is not None else ""


def _parse(raw: str) -> dict:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```", 2)[1].lstrip("json").strip() if "```" in raw else raw
    i, j = raw.find("{"), raw.rfind("}")
    if i >= 0 and j > i:
        raw = raw[i:j + 1]
    try:
        d = json.loads(raw)
        return {"resumo": _as_text(d.get("resumo", "")),
                "qa": _as_text(d.get("qa", ""))}
    except (json.JSONDecodeError, TypeError):
        return {"resumo": "", "qa": ""}


@app.cls(
    gpu="A10G",
    volumes={HF_HOME: model_volume},
    timeout=3600,
    scaledown_window=60,  # escala a zero ocioso: custo parado
)
class Distiller:
    @modal.enter()
    def load(self):
        from vllm import LLM
        from transformers import AutoTokenizer
        self._tok = AutoTokenizer.from_pretrained(MODEL)
        self._llm = LLM(model=MODEL, max_model_len=MAX_LEN,
                        gpu_memory_utilization=0.92, dtype="float16")

    @modal.method()
    def distill(self, lessons: list[dict]) -> list[dict]:
        """lessons: [{"id","title","text"}] -> [{"id","resumo","qa"}]."""
        from vllm import SamplingParams
        sp = SamplingParams(temperature=0.2, max_tokens=2200)
        prompts = []
        for ls in lessons:
            # trunca transcrição pra caber no contexto (deixa folga pro output)
            text = ls["text"][:28000]
            sys_, prompt = (SYS_JUR, PROMPT_JUR) if ls.get("domain") == "juridico" else (SYS, PROMPT)
            msgs = [{"role": "system", "content": sys_},
                    {"role": "user", "content": prompt.format(title=ls["title"], text=text)}]
            prompts.append(self._tok.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True))
        outs = self._llm.generate(prompts, sp)
        res = []
        for ls, o in zip(lessons, outs):
            d = _parse(o.outputs[0].text)
            res.append({"id": ls["id"], **d})
        return res


@app.local_entrypoint()
def main():
    """Smoke: `modal run scripts/modal_distill.py`."""
    d = Distiller()
    sample = [{"id": "smoke-1", "title": "Como prospectar",
               "text": "Então galera hoje vou falar de prospecção ativa. "
                       "Primeiro passo é listar 20 contas ideais. Depois "
                       "mandar mensagem personalizada no primeiro toque. "
                       "Não venda na primeira mensagem, gere curiosidade."}]
    out = d.distill.remote(sample)
    print("resumo:", out[0]["resumo"][:200])
    print("qa:", out[0]["qa"][:200])

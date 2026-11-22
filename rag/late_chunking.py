"""Late chunking (arXiv:2409.04701): embeda o documento inteiro por
token, agrupa em spans DEPOIS, faz mean-pooling por span. Diferente de
chunking ingênuo (que corta o texto ANTES de embedar) — cada chunk
carrega contexto do documento inteiro no embedding, não só do seu
próprio pedaço.

Bypassa a API pública do fastembed (`TextEmbedding.embed()`), que só
devolve vetor final por documento — sem acesso a embedding por token.

INVARIANTE: todo índice de token usado pra fatiar a matriz de embeddings
vem dos `offsets` da MESMA (única) chamada de encode() que gerou a
matriz. Nunca de contagens de token obtidas encodando pedaços separados —
cada encode() adiciona <s>/</s>, e somar essas contagens desalinha do
índice real (confirmado no modelo real: 12 sentenças = +20 tokens de
drift; 240 sentenças = +477, spans do fim caindo fora da matriz e
virando NaN silencioso).

Dois casos que o corte por sentença sozinho não cobre:
- Legenda automática do YouTube não tem pontuação: o vídeo inteiro é UMA
  "sentença". Sentença maior que max_chunk_tokens é subdividida em `\n`
  (linha de legenda) ou, sem `\n`, em fronteira de palavra (`split_long_span`).
- Documento maior que MAX_TOKENS (janela do modelo) é embedado em
  macro-janelas com overlap (`plan_windows`); cada chunk usa o vetor da
  janela em que fica mais centralizado. Sem perda de cauda.

DEPENDÊNCIAS: este arquivo é enviado sozinho pra imagem Modal. Só pode
importar stdlib, numpy, onnxruntime, tokenizers, huggingface_hub e
`rag.sentence_splitting` (que é stdlib pura; enviar os dois arquivos).
"""
import os
from pathlib import Path

import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

from rag.sentence_splitting import split_sentences_with_offsets

BGE_M3_REPO = "BAAI/bge-m3"
BGE_M3_ONNX_FILE = "onnx/model.onnx"
BGE_M3_ONNX_DATA_FILE = "onnx/model.onnx_data"
BGE_M3_TOKENIZER_FILE = "tokenizer.json"
# Respeita HF_HOME (Dockerfile e docker-compose.yml apontam
# ele pra /models, que é volume persistente). Com Path.home() fixo, o
# download caía na camada efêmera do container e os ~2.3GB voltavam a cada
# recriação.
BGE_M3_LOCAL_DIR = Path(os.getenv("HF_HOME") or (Path.home() / ".cache" / "huggingface")) / "bge-m3-onnx"
# Janela do modelo. Configurável porque o forward de 8192 tokens materializa
# atenção 16 heads x 8192^2 (GBs por camada): numa máquina sem memória de sobra
# (CPU de dev) vira swap. Default = janela nativa do BGE-M3.
MAX_TOKENS = int(os.getenv("LATE_CHUNK_WINDOW_TOKENS", "8192"))
# Macro-janelas (documento > MAX_TOKENS): janela de conteúdo = MAX_TOKENS - 2
# (sobra lugar pro <s>/</s>), avanço de 3/4 da janela (6144 no default) ->
# overlap de ~2046 tokens, bem maior que qualquer chunk (256), então todo
# chunk cabe inteiro em alguma janela.
WINDOW_STRIDE = MAX_TOKENS * 3 // 4


def pool_spans(token_embeddings: np.ndarray, spans: list[tuple[int, int]]) -> list[np.ndarray]:
    """Mean-pooling + normalização L2 por span de token.

    token_embeddings: [seq_len, dim], embedding de cada token (pré-pooling).
    spans: lista de (start, end) exclusivo, em índice de token.
    """
    out = []
    for start, end in spans:
        pooled = token_embeddings[start:end].mean(axis=0)
        norm = np.linalg.norm(pooled)
        out.append(pooled / norm if norm > 0 else pooled)
    return out


def sentence_token_bounds(offsets: list[tuple[int, int]], char_starts: list[int]) -> list[int]:
    """Traduz offset de char de cada sentença -> índice de token real,
    caminhando pelos `offsets` da encode() do documento inteiro.

    Devolve len(char_starts)+1 posições: o início de cada sentença mais a
    sentinela len(offsets) no fim (o chamador ajusta essa sentinela pro
    limite real da janela/documento em uso — não há mais truncamento fixo
    aqui, documento grande vira macro-janelas em vez de cortado).
    """
    bounds = []
    t = 0
    for c in char_starts:
        # primeiro token cujo trecho de char termina DEPOIS do início da
        # sentença (offsets são monotônicos; <s>/</s> vêm como (0, 0)).
        while t < len(offsets) and offsets[t][1] <= c:
            t += 1
        bounds.append(t)
    bounds.append(len(offsets))
    return bounds


def group_sentences(bounds: list[int], max_tokens: int = 256) -> list[tuple[int, int]]:
    """Agrupa unidades em chunks de até max_tokens, em índice de UNIDADE
    (start, end) exclusivo. Unidade sozinha maior que max_tokens vira grupo
    próprio (esta função nunca corta no meio de uma unidade — quem chama
    já entrega unidades ≤ max_tokens quando possível; em encode_document
    uma "unidade" é uma sentença inteira, ou, se a sentença for maior que
    max_chunk_tokens, um dos sub-spans que split_long_span produziu dela).

    bounds: saída de sentence_token_bounds, ou os bounds de unidade
    (n+1 posições)."""
    groups: list[tuple[int, int]] = []
    start = 0
    for i in range(len(bounds) - 1):
        if bounds[i] > bounds[start] and bounds[i + 1] - bounds[start] > max_tokens:
            groups.append((start, i))
            start = i
    if len(bounds) - 1 > start and bounds[-1] > bounds[start]:
        groups.append((start, len(bounds) - 1))
    return groups


def split_long_span(text: str, offsets: list[tuple[int, int]], a: int, b: int,
                    max_tokens: int) -> list[int]:
    """Pontos de corte (índices de token, início de cada sub-span depois do
    primeiro) pra dividir o span [a, b) em pedaços de até max_tokens tokens.

    Prefere quebra de linha (legenda), desde que o pedaço tenha ao menos
    metade de max_tokens; senão a última fronteira de palavra; só no fim da
    fila (palavra sozinha maior que max_tokens) corta no meio de palavra.
    Fronteira = espaço em branco entre o token anterior e este, medido nos
    offsets de char (o tokenizer inclui o espaço inicial no offset do token).
    """
    def gap(t: int) -> str:
        c = offsets[t][0]
        while c < len(text) and text[c].isspace():
            c += 1
        return text[offsets[t - 1][1]:c]

    cuts = []
    cur = a
    while b - cur > max_tokens:
        limit = cur + max_tokens
        best_word = best_line = None
        for t in range(limit, cur, -1):
            g = gap(t)
            if not g:
                continue
            if best_word is None:
                best_word = t
            if "\n" in g:
                best_line = t
                break
        if best_line is not None and best_line - cur >= max_tokens // 2:
            cut = best_line
        elif best_word is not None:
            cut = best_word
        else:
            cut = limit  # ponytail: palavra > max_tokens, corte forçado no token
        cuts.append(cut)
        cur = cut
    return cuts


def plan_windows(head: int, tail: int, n_tokens: int, max_tokens: int,
                 stride: int) -> list[tuple[int, int, int]]:
    """Janelas de embedding sobre os tokens de conteúdo [head, tail) de um
    documento com n_tokens (incluindo especiais). Devolve (ws, we, pre):
    a janela cobre os tokens globais [ws, we) e `pre` é quantos tokens
    especiais são prefixados na entrada do modelo (linha do token global p
    na matriz da janela = p - ws + pre).

    Cabe em max_tokens -> uma janela só, sequência inteira (pre=0), idêntico
    ao comportamento sem janela."""
    if n_tokens <= max_tokens:
        return [(0, n_tokens, 0)]
    width = max_tokens - 2
    windows = []
    ws = head
    while True:
        we = min(ws + width, tail)
        windows.append((ws, we, 1))
        if we >= tail:
            return windows
        ws += stride


def pick_window(windows: list[tuple[int, int, int]], a: int, b: int) -> int:
    """Índice da janela onde o span [a, b) fica mais centralizado, entre as
    que o contêm inteiro (senão, entre as que contêm o início)."""
    inside = [i for i, (ws, we, _) in enumerate(windows) if ws <= a and b <= we]
    cands = inside or [i for i, (ws, we, _) in enumerate(windows) if ws <= a < we] or [0]
    mid = (a + b) / 2
    return min(cands, key=lambda i: abs((windows[i][0] + windows[i][1]) / 2 - mid))


class LateChunkEncoder:
    """Roda BGE-M3 via ONNX Runtime direto, expondo embedding por token
    (a API pública do fastembed não expõe isso — ver Task 2 do plano)."""

    def __init__(self, repo_id: str = BGE_M3_REPO, onnx_file: str = BGE_M3_ONNX_FILE,
                 tokenizer_file: str = BGE_M3_TOKENIZER_FILE,
                 local_dir: str | Path = BGE_M3_LOCAL_DIR,
                 providers: list[str] | None = None):
        # local_dir (em vez do default do hf_hub_download) é necessário:
        # sem ele, hf_hub_download devolve um symlink pro blob store do
        # cache do HF, e o export ONNX do BGE-M3 é sharded (os pesos ficam
        # num onnx/model.onnx_data separado, referenciado por caminho
        # relativo) — o validador de external-data do onnxruntime rejeita
        # esse symlink ("External data path escapes model directory").
        # Baixando os três arquivos pro mesmo local_dir, o .onnx_data cai
        # como sibling de verdade do .onnx e a resolução relativa funciona.
        model_path = hf_hub_download(repo_id, onnx_file, local_dir=local_dir)
        hf_hub_download(repo_id, BGE_M3_ONNX_DATA_FILE, local_dir=local_dir)
        tokenizer_path = hf_hub_download(repo_id, tokenizer_file, local_dir=local_dir)
        self.session = ort.InferenceSession(
            model_path, providers=providers or ["CPUExecutionProvider"])
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self._input_names = {i.name for i in self.session.get_inputs()}

    def _run(self, ids: list[int]) -> np.ndarray:
        """Forward pass numa sequência de ids já com especiais. Devolve
        [len(ids), dim] (saída [0] da sessão = last_hidden_state pré-pooling)."""
        feed = {
            "input_ids": np.array([ids], dtype=np.int64),
            "attention_mask": np.ones((1, len(ids)), dtype=np.int64),
        }
        if "token_type_ids" in self._input_names:
            feed["token_type_ids"] = np.zeros_like(feed["input_ids"])
        return self.session.run(None, feed)[0][0]

    def token_embeddings(self, text: str) -> tuple[list[tuple[int, int]], np.ndarray]:
        """Tokeniza (truncando em MAX_TOKENS) e roda o forward pass. Retorna
        (offsets de char por token, matriz [seq_len, dim]). Usado pra
        texto curto (query); documento longo passa por encode_document."""
        enc = self.tokenizer.encode(text)
        return list(enc.offsets[:MAX_TOKENS]), self._run(list(enc.ids[:MAX_TOKENS]))

    def embed_whole(self, text: str) -> np.ndarray:
        """Mean-pooling + L2 do documento/query inteiro — o equivalente,
        nesta sessão ONNX, a um TextEmbedding(pooling=MEAN).embed()."""
        offsets, embeddings = self.token_embeddings(text)
        return pool_spans(embeddings, [(0, len(offsets))])[0]

    def encode_document(self, text: str, max_chunk_tokens: int = 256
                        ) -> list[tuple[str, int, np.ndarray]]:
        """Late chunking de ponta a ponta. Devolve (texto do chunk, offset
        de char onde ele começa, vetor). `chunk_text == text[char_pos:...]`
        (substring exata, sem espaço nas pontas): os chunks cobrem o texto
        de ponta a ponta, sem lacuna nem perda de cauda.

        Uma única encode() do documento inteiro; os spans saem dos offsets
        DESSA encode. Documento > MAX_TOKENS roda em macro-janelas."""
        sent_data = split_sentences_with_offsets(text)
        if not sent_data:
            return []
        enc = self.tokenizer.encode(text)
        ids, offsets = list(enc.ids), list(enc.offsets)
        n = len(ids)
        head = 1 if offsets[0] == (0, 0) else 0
        tail = n - 1 if n > head and offsets[-1] == (0, 0) else n
        bounds = sentence_token_bounds(offsets, [c for _, c in sent_data])
        bounds[-1] = tail  # sem o </s> no último span (o <s> já fica fora do 1º)

        # unidades atômicas de agrupamento: sentenças; sentença > max vira
        # várias unidades. (token inicial, char inicial) de cada uma.
        unit_tok: list[int] = []
        unit_char: list[int] = []
        for i, (_, c) in enumerate(sent_data):
            a, b = bounds[i], bounds[i + 1]
            if b <= a:
                continue
            unit_tok.append(a)
            unit_char.append(c)
            for cut in (split_long_span(text, offsets, a, b, max_chunk_tokens)
                        if b - a > max_chunk_tokens else []):
                k = offsets[cut][0]
                while k < len(text) and text[k].isspace():
                    k += 1
                # `cut` pode ter caído num token que é SÓ whitespace (ex.: em
                # legenda, "\n" vira token próprio em vez de virar prefixo do
                # token seguinte, diferente do espaço normal) — k já andou
                # pra depois dele, mas cut ainda aponta pro token de espaço.
                # Sem isso, o embedding desse token de espaço entra no chunk
                # ERRADO (o seguinte), enquanto o char dele já ficou no texto
                # do chunk anterior: mesmo resync que sentence_token_bounds
                # já faz (token cujo fim passa de k).
                while cut < len(offsets) and offsets[cut][1] <= k:
                    cut += 1
                unit_tok.append(cut)
                unit_char.append(k)
        if not unit_tok:
            return []
        ubounds = unit_tok + [tail]
        groups = group_sentences(ubounds, max_tokens=max_chunk_tokens)

        # Limite de char de cada grupo, já sem espaço/quebra de linha nas
        # pontas (era um .strip() solto no fim, DEPOIS do pooling — o vetor
        # podia incluir o token de um "\n" de fim de linha que o texto do
        # chunk não tinha mais, um mismatch real achado pelo teste
        # discriminativo). Resync (a, b) de token pro MESMO corte de char,
        # senão o vetor pooled representa texto diferente do chunk_text.
        bounds_char = []
        for g, (i, j) in enumerate(groups):
            raw_start = unit_char[i]
            raw_end = unit_char[groups[g + 1][0]] if g + 1 < len(groups) else len(text)
            raw = text[raw_start:raw_end]
            lead = len(raw) - len(raw.lstrip())
            trail = len(raw) - len(raw.rstrip())
            bounds_char.append((raw_start + lead, raw_end - trail))

        spans = []
        for (i, j), (cs, ce) in zip(groups, bounds_char):
            a, b = ubounds[i], ubounds[j]
            while a < b and offsets[a][1] <= cs:
                a += 1
            while b > a and offsets[b - 1][0] >= ce:
                b -= 1
            spans.append((a, b))

        windows = plan_windows(head, tail, n, MAX_TOKENS, WINDOW_STRIDE)
        by_window: dict[int, list[int]] = {}
        for k, (a, b) in enumerate(spans):
            by_window.setdefault(pick_window(windows, a, b), []).append(k)
        vectors: list[np.ndarray | None] = [None] * len(spans)
        for w, ks in by_window.items():
            ws, we, pre = windows[w]
            win_ids = ids if pre == 0 else [ids[0]] + ids[ws:we] + [ids[-1]]
            hidden = self._run(win_ids)
            local = [(max(spans[k][0], ws) - ws + pre, min(spans[k][1], we) - ws + pre)
                     for k in ks]
            for k, v in zip(ks, pool_spans(hidden, local)):
                vectors[k] = v

        return [(text[cs:ce], cs, v) for (cs, ce), v in zip(bounds_char, vectors)]


_encoder = None


def get_encoder(providers: list[str] | None = None) -> LateChunkEncoder:
    """Singleton: cada InferenceSession carrega os ~2.3GB de peso de novo.
    Na ingestão, o lado do documento e o lado da query (BgeM3Embeddings)
    caem os dois aqui e dividem uma sessão só."""
    global _encoder
    if _encoder is None:
        _encoder = LateChunkEncoder(providers=providers)
    return _encoder

FROM python:3.13-slim
WORKDIR /app
# HF_HUB_DISABLE_SHARED_BLOBS: o cache-wide shared blob store do huggingface_hub
# guarda cada arquivo baixado em blobs/<prefixo-do-hash>/<hash> e symlinka pro
# snapshot do repo. Modelo ONNX sharded (.onnx + .onnx_data, ex. e5-large e o
# reranker Jina) acaba com os dois arquivos symlinkados em blobs/ de PREFIXOS
# diferentes (hashes diferentes) — o onnxruntime valida que o external-data
# resolve dentro do MESMO diretório do .onnx e rejeita ("External data path
# escapes model directory"), quebrando a busca. Desligando o
# blob store compartilhado, o download cai pro modo repo-local (arquivo real,
# sem symlink cruzando diretório) — mesma causa raiz já corrigida pro BGE-M3
# em rag/late_chunking.py (download com local_dir explícito), aqui resolvida
# na camada de baixo (huggingface_hub) porque o e5/reranker passam pelo
# FastEmbedEmbeddings/TextCrossEncoder, que não repassam local_dir.
ENV PYTHONUNBUFFERED=1 HF_HOME=/models FASTEMBED_CACHE_DIR=/models HF_HUB_DISABLE_SHARED_BLOBS=1 MCP_TRANSPORT=http MCP_HOST=0.0.0.0 MCP_PORT=8000
RUN apt-get update && apt-get install -y --no-install-recommends curl nodejs npm && rm -rf /var/lib/apt/lists/*
# Claude Code CLI para a tool `answer` (`claude -p`, Haiku gera e Sonnet verifica). Versão fixa; o login vem de
# CLAUDE_CODE_OAUTH_TOKEN em runtime (nunca no build).
RUN npm install -g @anthropic-ai/claude-code@2.1.302 && npm cache clean --force
ENV RAG_CLAUDE_HOME=/home/rag-claude
RUN mkdir -p /home/rag-claude
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --retries=10 CMD curl -fsS http://127.0.0.1:8000/healthz || exit 1
CMD ["python","-m","rag.mcp_server"]

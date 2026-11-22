"""Configuração central lida do .env."""
import os
from pathlib import Path
from dotenv import load_dotenv

# Carrega o .env da raiz do projeto (não deste pacote) pelo caminho absoluto
# (funciona seja qual for o cwd, importante quando o MCP server é lançado
# pelo Claude Code).
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY") or None
# Optional Cloudflare Access headers when Qdrant is exposed behind Access.
# Leave unset on a local Docker network. If the URL is HTTPS on 443, include
# `:443` explicitly — qdrant-client otherwise defaults to 6333.
QDRANT_CF_ACCESS_CLIENT_ID = os.getenv("QDRANT_CF_ACCESS_CLIENT_ID") or None
QDRANT_CF_ACCESS_CLIENT_SECRET = os.getenv("QDRANT_CF_ACCESS_CLIENT_SECRET") or None
QDRANT_CF_ACCESS_TOKEN = os.getenv("QDRANT_CF_ACCESS_TOKEN") or None
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "legal_rag")
# Busca em várias coleções de uma vez (lista separada por vírgula). Só a busca usa a lista;
# ingestão, status e leitura de aula seguem COLLECTION_NAME. Vazio = só COLLECTION_NAME.
COLLECTION_NAMES = [c.strip() for c in os.getenv("COLLECTION_NAMES", "").split(",") if c.strip()] or [COLLECTION_NAME]

EMBED_MODEL = os.getenv("EMBED_MODEL", "intfloat/multilingual-e5-large")
RERANK_MODEL = os.getenv("RERANK_MODEL", "jinaai/jina-reranker-v2-base-multilingual")
RAG_RERANK = os.getenv("RAG_RERANK", "1") not in ("0", "false", "no")
# Off by default. Turn on per instance after measuring on that collection.
RAG_POOLS_SEPARADOS = os.getenv("RAG_POOLS_SEPARADOS", "0") not in ("0", "false", "no")
RAG_PENALIDADE_SINTETICO = float(os.getenv("RAG_PENALIDADE_SINTETICO", "0.0"))

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
LATE_CHUNK_MAX_TOKENS = int(os.getenv("LATE_CHUNK_MAX_TOKENS", "256"))

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "claude-cli").lower()
RAG_GEN_MODEL = os.getenv("RAG_GEN_MODEL", "claude-haiku-5-5")
RAG_VERIFY_MODEL = os.getenv("RAG_VERIFY_MODEL", "claude-sonnet-5-5")
RAG_DEADLINE_S = float(os.getenv("RAG_DEADLINE_S", "90"))
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY") or None
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-5")

TRANSCRIPT_LANGS = [
    s.strip() for s in os.getenv("TRANSCRIPT_LANGS", "pt,pt-BR,en").split(",") if s.strip()
]

# HTTP path of the MCP endpoint. Auth is the Authorization header, not this path.
MCP_HTTP_PATH = os.getenv("MCP_HTTP_PATH", "/mcp")

# rag-mcp

MCP server: hybrid search + rerank over Qdrant, ingest pipeline, and an eval
harness. Demo corpus: public-domain Brazilian legal excerpts in `sample/corpus/`.
Overview and environment variables: `README.md`.

- `rag/`: retrieve, rerank, MCP server
- `ingest/`: incremental ingest and ledger
- `eval/`: golden sets and harness; see `eval/README.md`
- `sample/`: public sample records
- `tests/`: `pytest` (skips `slow` and `live` by default; `-m live` uses a real LLM)
- `data/`: ingest records at runtime; gitignored; copy from `sample/corpus/`
- Secrets only via environment variables; never in the repo

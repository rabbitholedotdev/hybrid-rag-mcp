# Sample corpus

Public-domain excerpts of Brazilian federal law, small enough to ingest on a laptop
and large enough for hybrid search + rerank to return real passages.

| `video_id` | Source | Area |
|---|---|---|
| `cf-art5` | Constituição Federal/1988, art. 5º (excerpt) | constitucional |
| `cf-art37` | CF/1988, art. 37 (excerpt, LIMPE) | constitucional |
| `lei-greve-7783` | Lei 7.783/1989 (right to strike, excerpt) | trabalho |
| `cdc-direitos` | Lei 8.078/1990, CDC (excerpt) | consumidor |
| `cc-contratos` | Lei 10.406/2002, arts. 421–427, 441 (excerpt) | civil |

Official texts: [Planalto](https://www.planalto.gov.br/). Records follow the ingest JSON
schema (`text`, `video_id`, `title`, `area`, `source_type=material`).

Eval goldens live in `eval/golden_legal.jsonl` and `eval/traps_legal.jsonl`.
They are questions about these excerpts, not a private client collection.

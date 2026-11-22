# eval/ — retrieval and citation harness

Measures the real pipeline (`rag.query.retrieve` + optional answer) on a golden
set, with a bias toward **citation safety** — not only “did it find the lesson?”
but “did it claim something that is not in the source?”

The public goldens (`golden_legal.jsonl`, `traps_legal.jsonl`) target the sample
legal excerpts in `sample/corpus/`. They are a demo of the harness, not a
published benchmark.

## Flow

1. Curate questions in `eval/golden_legal.jsonl` (`expected_ids` = `video_id` of
   the source record). Add traps in `eval/traps_legal.jsonl`.
2. Freeze the split once: `python -m eval.split` (writes `split.json`).
3. Run:

```bash
python -m eval.run_eval --k 6            # answer + faithfulness judge
python -m eval.run_eval --no-judge       # retrieval + number guard
python -m eval.run_eval --no-answer      # retrieval only
python -m eval.run_eval --primary-only   # drop resumo/qa from search
```

Reports go under `eval/reports/` (gitignored).

## Metrics

- **primary_recall@k** — gold lesson among *citable* hits
- **lesson_recall@k** — gold lesson anywhere in top-k (including synthetic)
- **facet coverage** — fraction of `expected_keywords` present in the context
- **numbers_hallucination_rate** — digits asserted in the answer but absent from
  primary text
- **faithfulness** — LLM judge, temperature 0, lexical fallback in `eval/judge.py`
- **correct abstention** on traps

## Gate and funnel

- `python -m eval.split` / `python -m eval.funil_dados` write frozen id lists.
- `python -m eval.run_gate run --mode baseline|assertive --colecao legal_rag --split teste --tag T`
- `python -m eval.run_funnel run --colecao legal_rag --split calibracao --tag T`

These need a live Qdrant populated with the sample (or your own) collection.
CI does not run them.

## Limits

Small n, one language, one domain. Do not treat sample scores as evidence that
the same hyperparameters win on a different corpus or embedding.

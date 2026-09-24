# Ticket triage pipeline — Phases 1–5

Phases 1–2 load and validate `tickets.json` and `label_schema.json` with
Pydantic, sort tickets by `ticket_id`, preserve each original message, and
create a deterministic normalized message and stage history. Phase 3 adds
OpenAI Structured Outputs with Pydantic parsing. Phase 4 routes on the lower
of category and urgency confidence; human-review tickets receive a null
suggested reply. The pipeline writes deterministic `predictions.json`, `errors.json`,
`state_history.json`, and `model_responses.json` artifacts to `outputs/` after
classification. Phase 5 adds `metrics.json` and final validation of routes,
labels, replies, stage histories, and output files.

Run from this directory:

```bash
python3 main.py
```

Optional arguments:

```bash
python3 main.py --tickets path/to/tickets.json \
  --label-schema path/to/label_schema.json \
  --config config.json --confidence-threshold 0.75
```

Classify up to three tickets with the OpenAI API (requires `OPENAI_API_KEY`):

```bash
python3 main.py --classify --limit 3
```

Use `--output-dir` to choose a different artifact directory. Re-running with
`--mock` and the same inputs produces byte-identical JSON artifacts.

For offline, deterministic verification without an API call:

```bash
python3 main.py --classify --mock --limit 3
```

The optional config JSON may contain `confidence_threshold`, `tickets_path`,
`label_schema_path`, and `output_dir`.

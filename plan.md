# Implementation Plan: Customer Support Ticket Urgency Pipeline

## Phase 1 — Inputs, configuration, and validation

- Define the expected ticket and output schemas.
- Load `ticket.json` and validate required fields: `ticket_id`, `message`, expected category, and expected urgency level.
- Add explicit configuration for the confidence threshold, input/output paths, and deterministic mock/fallback mode.
- Fail clearly on malformed JSON, missing fields, duplicate ticket IDs, or unsupported labels.

**Exit criteria:** valid inputs load successfully, invalid inputs produce actionable errors, and configuration is explicit.

## Phase 2 — Deterministic preprocessing and model interface

- Normalize each message deterministically by trimming whitespace, normalizing case/spacing, and handling empty text consistently.
- Preserve stable ticket ordering from the input, with a deterministic tie-breaker if needed.
- Define a structured model request and response schema containing category, urgency, confidence, and suggested reply.
- Support a deterministic mock/fallback model so runs do not require network access or a live model.
- Store the raw structured model response for replayability and debugging.

**Exit criteria:** identical inputs and configuration produce identical preprocessed text, model requests, and parsed predictions.

## Phase 3 — Structured parsing and confidence-based routing

- Parse model responses into the required typed fields.
- Validate category and urgency against the supported label sets.
- Normalize or reject invalid confidence values.
- Route tickets as `auto` when confidence is at or above the configured threshold; otherwise route to `human_review`.
- Set `suggested_reply` to `null` for human-review tickets and retain an error when parsing or validation fails.

**Exit criteria:** every ticket has a deterministic routing decision, and no low-confidence ticket receives an automated reply.

## Phase 4 — Reply generation, persistence, and stage tracking

- Generate a short reply only for safely automated tickets.
- Track the required lifecycle stages:
  `INIT → INPUTS_LOADED → TEXT_PREPROCESSED → MODEL_PROMPTED → STRUCTURED_OUTPUT_PARSED → CONFIDENCE_CHECKED → ROUTED → RESPONSE_GENERATED → RESULTS_SAVED`.
- Save per-ticket results with `ticket_id`, category, urgency, confidence, route, suggested reply, and error when applicable.
- Write a run summary containing configuration, counts, and any failures.

**Exit criteria:** a clean run produces complete, machine-readable output files and a traceable stage transition record.

## Phase 5 — Evaluation, validation, and reproducibility checks

- Compare predictions with expected labels when they are present.
- Compute category accuracy, urgency accuracy, coverage, and human-review rate.
- Complete `EVALUATION_COMPUTED` and `VALIDATION_COMPLETED` stages only after outputs and metrics pass schema checks.
- Add tests for parsing, preprocessing, routing at the threshold boundary, malformed model output, empty messages, and replay consistency.
- Document the run command and expected output locations.

**Definition of done:** the pipeline runs from a clean checkout, produces deterministic structured outputs and stage traces, routes uncertain tickets to human review, saves raw/model-derived results, and reports the required evaluation metrics.

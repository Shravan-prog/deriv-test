"""Phases 1–5: validate, preprocess, classify, route, save, and evaluate tickets.

The pipeline is deterministic in mock mode and validates its final artifacts.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, ValidationError, create_model, field_validator


class LabelSchema(BaseModel):
    """Allowed labels used by the eventual classifier."""

    model_config = ConfigDict(extra="forbid")

    categories: list[str] = Field(min_length=1)
    urgency_levels: list[str] = Field(min_length=1)

    @field_validator("categories", "urgency_levels")
    @classmethod
    def validate_labels(cls, values: list[str]) -> list[str]:
        cleaned = [value.strip() for value in values]
        if any(not value for value in cleaned):
            raise ValueError("labels must be non-empty strings")
        if len(set(cleaned)) != len(cleaned):
            raise ValueError("labels must not contain duplicates")
        return cleaned


class Ticket(BaseModel):
    """A source ticket, accepting the names used by the project fixtures."""

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    ticket_id: str = Field(min_length=1)
    message: str = Field(
        min_length=1,
        validation_alias=AliasChoices("message", "customer_message"),
    )
    expected_category: str = Field(min_length=1)
    expected_urgency: str = Field(min_length=1)

    @field_validator("ticket_id", "expected_category", "expected_urgency")
    @classmethod
    def strip_text(cls, value: str) -> str:
        return value.strip() if isinstance(value, str) else value

    @field_validator("ticket_id", "message")
    @classmethod
    def reject_blank_required_text(cls, value: str) -> str:
        if not value:
            raise ValueError("must not be blank")
        return value


class PipelineConfig(BaseModel):
    """Explicit, validated configuration for the pipeline."""

    model_config = ConfigDict(extra="forbid")

    confidence_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    tickets_path: str = "tickets.json"
    label_schema_path: str = "label_schema.json"
    output_dir: str = "outputs"


class PipelineStage(StrEnum):
    """Stages reached by each ticket during the implemented phases."""

    INIT = "INIT"
    INPUTS_LOADED = "INPUTS_LOADED"
    TEXT_PREPROCESSED = "TEXT_PREPROCESSED"
    MODEL_PROMPTED = "MODEL_PROMPTED"
    STRUCTURED_OUTPUT_PARSED = "STRUCTURED_OUTPUT_PARSED"
    CONFIDENCE_CHECKED = "CONFIDENCE_CHECKED"
    ROUTED = "ROUTED"
    RESPONSE_GENERATED = "RESPONSE_GENERATED"
    RESULTS_SAVED = "RESULTS_SAVED"
    EVALUATION_COMPUTED = "EVALUATION_COMPUTED"
    VALIDATION_COMPLETED = "VALIDATION_COMPLETED"


class PreprocessedTicket(BaseModel):
    """Ticket data plus immutable source text and preprocessing trace."""

    model_config = ConfigDict(extra="forbid")

    ticket: Ticket
    original_message: str
    normalized_message: str
    stage_history: list[PipelineStage]


class ClassificationOutput(BaseModel):
    """Structured model result; label membership is checked against the schema."""

    model_config = ConfigDict(extra="forbid")

    category: str
    urgency: str
    category_confidence: float = Field(ge=0.0, le=1.0)
    urgency_confidence: float = Field(ge=0.0, le=1.0)
    suggested_reply: str = Field(min_length=1, max_length=500)

    @field_validator("category", "urgency", "suggested_reply")
    @classmethod
    def strip_output_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be blank")
        return value


class ClassifiedTicket(BaseModel):
    """A parsed result or a safe error, with the ticket's stage history."""

    model_config = ConfigDict(extra="forbid")

    ticket_id: str
    classification: ClassificationOutput | None = None
    raw_model_response: str | None = None
    error: str | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    route: Literal["auto", "human_review"] | None = None
    suggested_reply: str | None = None
    stage_history: list[PipelineStage]


MISSING_TEXT = "[missing]"


def normalize_message(value: str | None) -> str:
    """Apply deterministic whitespace, casing, and missing-value rules."""

    if value is None or not value.strip():
        return MISSING_TEXT
    # ``casefold`` is deterministic and more complete than lower() for Unicode.
    return re.sub(r"\s+", " ", value.strip()).casefold()


def preprocess_tickets(tickets: list[Ticket]) -> list[PreprocessedTicket]:
    """Create sorted, replayable ticket states without making model calls."""

    processed: list[PreprocessedTicket] = []
    for ticket in tickets:
        processed.append(
            PreprocessedTicket(
                ticket=ticket,
                original_message=ticket.message,
                normalized_message=normalize_message(ticket.message),
                stage_history=[
                    PipelineStage.INIT,
                    PipelineStage.INPUTS_LOADED,
                    PipelineStage.TEXT_PREPROCESSED,
                ],
            )
        )
    return processed


def classification_model(schema: LabelSchema) -> type[BaseModel]:
    """Build an API/Pydantic schema whose label fields are schema-derived."""

    # Literal values become JSON-schema enums, so the API is instructed to use
    # exactly the labels loaded from label_schema.json.
    category_type = Literal[tuple(schema.categories)]
    urgency_type = Literal[tuple(schema.urgency_levels)]
    return create_model(
        "TicketClassification",
        __base__=ClassificationOutput,
        category=(category_type, ...),
        urgency=(urgency_type, ...),
    )


def _raw_response_text(response: Any) -> str:
    """Capture a stable raw response representation for replay/debugging."""

    output_text = getattr(response, "output_text", None)
    if isinstance(output_text, str) and output_text:
        return output_text
    if hasattr(response, "model_dump"):
        try:
            return json.dumps(response.model_dump(mode="json"), sort_keys=True)
        except Exception:  # pragma: no cover - SDK response variations
            pass
    return str(response)


def _classification_prompt(schema: LabelSchema, message: str) -> list[dict[str, Any]]:
    return [
        {
            "role": "system",
            "content": (
                "Classify the support ticket. Return only the requested structured fields. "
                f"Allowed categories: {', '.join(schema.categories)}. "
                f"Allowed urgency levels: {', '.join(schema.urgency_levels)}. "
                "Confidence values must be numbers from 0 to 1. Keep the reply concise."
            ),
        },
        {"role": "user", "content": message},
    ]


def classify_ticket(
    client: Any,
    ticket: PreprocessedTicket,
    schema: LabelSchema,
    model: str,
) -> ClassifiedTicket:
    """Call OpenAI Structured Outputs and safely parse one ticket."""

    history = [*ticket.stage_history, PipelineStage.MODEL_PROMPTED]
    output_schema = classification_model(schema)
    try:
        response = client.responses.parse(
            model=model,
            input=_classification_prompt(schema, ticket.normalized_message),
            text_format=output_schema,
        )
        raw_response = _raw_response_text(response)
        parsed = getattr(response, "output_parsed", None)
        if parsed is None:
            raise ValueError("model returned no parsed structured output (possibly a refusal)")
        parsed_dict = parsed.model_dump() if hasattr(parsed, "model_dump") else parsed
        classification = ClassificationOutput.model_validate(parsed_dict)
        if classification.category not in schema.categories:
            raise ValueError(f"model returned unsupported category: {classification.category}")
        if classification.urgency not in schema.urgency_levels:
            raise ValueError(f"model returned unsupported urgency: {classification.urgency}")
        return ClassifiedTicket(
            ticket_id=ticket.ticket.ticket_id,
            classification=classification,
            raw_model_response=raw_response,
            stage_history=[*history, PipelineStage.STRUCTURED_OUTPUT_PARSED],
        )
    except Exception as exc:
        return ClassifiedTicket(
            ticket_id=ticket.ticket.ticket_id,
            raw_model_response=locals().get("raw_response"),
            error=f"structured classification failed: {exc}",
            stage_history=history,
        )


def route_ticket(ticket: ClassifiedTicket, confidence_threshold: float) -> ClassifiedTicket:
    """Route on the lower of category and urgency confidence.

    This function never routes malformed classifications: those remain errors
    without an automated reply.
    """

    if ticket.classification is None:
        return ticket
    confidence = min(
        ticket.classification.category_confidence,
        ticket.classification.urgency_confidence,
    )
    route = "auto" if confidence >= confidence_threshold else "human_review"
    reply = ticket.classification.suggested_reply if route == "auto" else None
    return ticket.model_copy(
        update={
            "confidence": confidence,
            "route": route,
            "suggested_reply": reply,
            "stage_history": [
                *ticket.stage_history,
                PipelineStage.CONFIDENCE_CHECKED,
                PipelineStage.ROUTED,
                PipelineStage.RESPONSE_GENERATED,
            ],
        }
    )


def _write_deterministic_json(path: Path, payload: Any) -> None:
    """Write canonical JSON so repeated runs are byte-for-byte comparable."""

    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def save_outputs(results: list[ClassifiedTicket], output_dir: Path) -> list[ClassifiedTicket]:
    """Save reproducibility artifacts and mark every result as persisted."""

    finalized = [
        result.model_copy(
            update={"stage_history": [*result.stage_history, PipelineStage.RESULTS_SAVED]}
        )
        for result in sorted(results, key=lambda item: item.ticket_id)
    ]
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions = []
    errors = []
    state_history = []
    model_responses = []
    for result in finalized:
        state_history.append(
            {"ticket_id": result.ticket_id, "stages": [str(stage) for stage in result.stage_history]}
        )
        model_responses.append(
            {
                "ticket_id": result.ticket_id,
                "raw_response": result.raw_model_response,
                "structured_response": (
                    result.classification.model_dump() if result.classification is not None else None
                ),
            }
        )
        if result.classification is not None and result.route is not None:
            predictions.append(
                {
                    "ticket_id": result.ticket_id,
                    "category": result.classification.category,
                    "urgency": result.classification.urgency,
                    "confidence": result.confidence,
                    "route": result.route,
                    "suggested_reply": result.suggested_reply,
                }
            )
        if result.error is not None:
            errors.append({"ticket_id": result.ticket_id, "error": result.error})

    _write_deterministic_json(output_dir / "predictions.json", predictions)
    _write_deterministic_json(output_dir / "errors.json", errors)
    _write_deterministic_json(output_dir / "state_history.json", state_history)
    _write_deterministic_json(output_dir / "model_responses.json", model_responses)
    return finalized


def evaluate_results(results: list[ClassifiedTicket], tickets: list[Ticket]) -> dict[str, Any]:
    """Compute label accuracy and routing rates for this deterministic run."""

    total = len(tickets)
    by_id = {ticket.ticket_id: ticket for ticket in tickets}
    labeled = [ticket for ticket in tickets if ticket.expected_category and ticket.expected_urgency]
    successful = [result for result in results if result.classification is not None]
    category_correct = sum(
        result.classification is not None
        and result.ticket_id in by_id
        and result.classification.category == by_id[result.ticket_id].expected_category
        for result in successful
    )
    urgency_correct = sum(
        result.classification is not None
        and result.ticket_id in by_id
        and result.classification.urgency == by_id[result.ticket_id].expected_urgency
        for result in successful
    )
    auto_count = sum(result.route == "auto" for result in results)
    review_count = sum(result.route == "human_review" for result in results)
    return {
        "total_tickets": total,
        "labeled_tickets": len(labeled),
        "classified_tickets": len(successful),
        "category_accuracy": (category_correct / len(labeled)) if labeled else None,
        "urgency_accuracy": (urgency_correct / len(labeled)) if labeled else None,
        "coverage": (auto_count / total) if total else 0.0,
        "human_review_rate": (review_count / total) if total else 0.0,
        "auto_routed_tickets": auto_count,
        "human_review_tickets": review_count,
        "error_tickets": sum(result.error is not None for result in results),
    }


def validate_final_results(
    results: list[ClassifiedTicket],
    tickets: list[Ticket],
    schema: LabelSchema,
) -> list[str]:
    """Return actionable invariant violations; an empty list means valid."""

    issues: list[str] = []
    expected_ids = [ticket.ticket_id for ticket in tickets]
    result_ids = [result.ticket_id for result in results]
    if result_ids != expected_ids:
        issues.append("results must contain every ticket exactly once in ticket_id order")
    allowed_routes = {"auto", "human_review"}
    required_prefix = [
        PipelineStage.INIT,
        PipelineStage.INPUTS_LOADED,
        PipelineStage.TEXT_PREPROCESSED,
        PipelineStage.MODEL_PROMPTED,
        PipelineStage.STRUCTURED_OUTPUT_PARSED,
        PipelineStage.CONFIDENCE_CHECKED,
        PipelineStage.ROUTED,
        PipelineStage.RESPONSE_GENERATED,
        PipelineStage.RESULTS_SAVED,
        PipelineStage.EVALUATION_COMPUTED,
    ]
    for result in results:
        if result.error is not None or result.classification is None:
            issues.append(f"{result.ticket_id}: classification error prevents final validation")
            continue
        if result.route not in allowed_routes:
            issues.append(f"{result.ticket_id}: invalid route {result.route!r}")
        if result.classification.category not in schema.categories:
            issues.append(f"{result.ticket_id}: category is not in label schema")
        if result.classification.urgency not in schema.urgency_levels:
            issues.append(f"{result.ticket_id}: urgency is not in label schema")
        if result.confidence is None or not 0.0 <= result.confidence <= 1.0:
            issues.append(f"{result.ticket_id}: confidence must be between 0 and 1")
        if result.route == "human_review" and result.suggested_reply is not None:
            issues.append(f"{result.ticket_id}: human-review ticket must have a null reply")
        if result.route == "auto" and not result.suggested_reply:
            issues.append(f"{result.ticket_id}: auto-routed ticket must have a reply")
        if result.stage_history[: len(required_prefix)] != required_prefix:
            issues.append(f"{result.ticket_id}: incomplete or out-of-order stage history")
    return issues


def append_stage(results: list[ClassifiedTicket], stage: PipelineStage) -> list[ClassifiedTicket]:
    return [
        result.model_copy(update={"stage_history": [*result.stage_history, stage]})
        for result in results
    ]


def write_state_history(results: list[ClassifiedTicket], output_dir: Path) -> None:
    state_history = [
        {"ticket_id": result.ticket_id, "stages": [str(stage) for stage in result.stage_history]}
        for result in sorted(results, key=lambda item: item.ticket_id)
    ]
    _write_deterministic_json(output_dir / "state_history.json", state_history)
def mock_classify_ticket(ticket: PreprocessedTicket, schema: LabelSchema) -> ClassifiedTicket:
    """Deterministic local demonstration path; it never calls the API."""

    text = ticket.normalized_message
    keyword_map = (
        ("login_access", ("login", "password", "sign in")),
        ("billing", ("payment", "charge", "withdraw", "withdrew", "refund")),
        ("verification", ("verify", "verification", "document")),
        ("technical_issue", ("slow", "error", "freeze", "bug")),
        ("account_closure", ("close my account", "delete my account")),
        ("feature_request", ("would be great", "feature", "dark mode")),
    )
    category = next(
        (name for name, words in keyword_map if name in schema.categories and any(word in text for word in words)),
        "other" if "other" in schema.categories else schema.categories[0],
    )
    urgency = "high" if any(word in text for word in ("can't", "cannot", "urgent", "missing")) else (
        "low" if "low" in schema.urgency_levels and category == "feature_request" else schema.urgency_levels[0]
    )
    if urgency not in schema.urgency_levels:
        urgency = schema.urgency_levels[0]
    classification = ClassificationOutput(
        category=category,
        urgency=urgency,
        category_confidence=0.75,
        urgency_confidence=0.70,
        suggested_reply="Thanks for contacting us. We’re reviewing your request and will follow up shortly.",
    )
    return ClassifiedTicket(
        ticket_id=ticket.ticket.ticket_id,
        classification=classification,
        raw_model_response=classification.model_dump_json(),
        stage_history=[*ticket.stage_history, PipelineStage.MODEL_PROMPTED, PipelineStage.STRUCTURED_OUTPUT_PARSED],
    )


def _read_json(path: Path, description: str) -> Any:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError as exc:
        raise ValueError(f"{description} not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"{description} contains malformed JSON at line {exc.lineno}, column {exc.colno}: {exc.msg}"
        ) from exc
    except OSError as exc:
        raise ValueError(f"could not read {description} {path}: {exc}") from exc


def load_inputs(tickets_path: Path, label_schema_path: Path) -> tuple[list[Ticket], LabelSchema]:
    """Read both files and validate their complete Phase 1 data contract."""

    raw_schema = _read_json(label_schema_path, "label schema")
    try:
        schema = LabelSchema.model_validate(raw_schema)
    except ValidationError as exc:
        raise ValueError(f"invalid label schema {label_schema_path}: {exc}") from exc

    raw_tickets = _read_json(tickets_path, "tickets file")
    if not isinstance(raw_tickets, list):
        raise ValueError(f"invalid tickets file {tickets_path}: expected a JSON array")

    tickets: list[Ticket] = []
    errors: list[str] = []
    for index, raw_ticket in enumerate(raw_tickets):
        try:
            ticket = Ticket.model_validate(raw_ticket)
            if ticket.expected_category is not None and ticket.expected_category not in schema.categories:
                raise ValueError(f"expected_category '{ticket.expected_category}' is not in label schema")
            if ticket.expected_urgency is not None and ticket.expected_urgency not in schema.urgency_levels:
                raise ValueError(f"expected_urgency '{ticket.expected_urgency}' is not in label schema")
            tickets.append(ticket)
        except (ValidationError, ValueError) as exc:
            errors.append(f"ticket index {index}: {exc}")

    ids = [ticket.ticket_id for ticket in tickets]
    duplicates = sorted({ticket_id for ticket_id in ids if ids.count(ticket_id) > 1})
    if duplicates:
        errors.append(f"duplicate ticket_id values: {', '.join(duplicates)}")
    if errors:
        raise ValueError("input validation failed:\n- " + "\n- ".join(errors))

    tickets.sort(key=lambda ticket: ticket.ticket_id)
    return tickets, schema


def load_config(config_path: Path | None) -> PipelineConfig:
    if config_path is None:
        return PipelineConfig()
    raw_config = _read_json(config_path, "config file")
    try:
        return PipelineConfig.model_validate(raw_config)
    except ValidationError as exc:
        raise ValueError(f"invalid config {config_path}: {exc}") from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run ticket-triage phases 1–5.")
    parser.add_argument("--tickets", type=Path, help="path to tickets.json")
    parser.add_argument("--label-schema", type=Path, help="path to label_schema.json")
    parser.add_argument("--config", type=Path, help="optional JSON config file")
    parser.add_argument("--output-dir", type=Path, help="directory for reproducibility artifacts")
    parser.add_argument("--confidence-threshold", type=float, help="override configured threshold")
    parser.add_argument("--classify", action="store_true", help="classify tickets with OpenAI Structured Outputs")
    parser.add_argument("--mock", action="store_true", help="use deterministic local classification instead of the API")
    parser.add_argument("--model", default="gpt-4o-mini", help="OpenAI model used with --classify")
    parser.add_argument("--limit", type=int, default=None, help="optional maximum tickets to classify")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = load_config(args.config)
        if args.tickets is not None:
            config = config.model_copy(update={"tickets_path": str(args.tickets)})
        if args.label_schema is not None:
            config = config.model_copy(update={"label_schema_path": str(args.label_schema)})
        if args.output_dir is not None:
            config = config.model_copy(update={"output_dir": str(args.output_dir)})
        if args.confidence_threshold is not None:
            config = config.model_copy(update={"confidence_threshold": args.confidence_threshold})
            config = PipelineConfig.model_validate(config.model_dump())

        tickets, schema = load_inputs(Path(config.tickets_path), Path(config.label_schema_path))
        processed_tickets = preprocess_tickets(tickets)
        if args.limit is not None and args.limit < 1:
            raise ValueError("--limit must be at least 1")
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print("Phases 1–5 validation, preprocessing, classification, routing, saving, and evaluation pipeline ready.")
    print(f"Validated ticket count: {len(tickets)}")
    print(f"Allowed categories: {', '.join(schema.categories)}")
    print(f"Allowed urgency levels: {', '.join(schema.urgency_levels)}")
    print(f"Confidence threshold: {config.confidence_threshold:g}")
    print("Model calls: not made." if not args.classify else ("Model calls: deterministic mock." if args.mock else "Model calls: OpenAI Responses API."))
    print("\nPreprocessing samples:")
    for item in processed_tickets[:3]:
        print(f"- Ticket {item.ticket.ticket_id}")
        print(f"  Original text: {item.original_message!r}")
        print(f"  Normalized text: {item.normalized_message!r}")
        print(f"  Stage history: {' -> '.join(item.stage_history)}")

    if args.classify:
        selected = processed_tickets if args.limit is None else processed_tickets[: args.limit]
        run_tickets = [item.ticket for item in selected]
        if args.mock:
            classified = [mock_classify_ticket(item, schema) for item in selected]
        else:
            try:
                from openai import OpenAI
            except ImportError:
                print("ERROR: install the OpenAI SDK (`pip install -r requirements.txt`) before using --classify.", file=sys.stderr)
                return 1
            try:
                client = OpenAI()
                classified = [classify_ticket(client, item, schema, args.model) for item in selected]
            except Exception as exc:
                print(f"ERROR: could not initialize OpenAI classification: {exc}", file=sys.stderr)
                return 1

        routed = [route_ticket(result, config.confidence_threshold) for result in classified]
        try:
            routed = save_outputs(routed, Path(config.output_dir))
        except OSError as exc:
            print(f"ERROR: could not save reproducibility outputs: {exc}", file=sys.stderr)
            return 1
        metrics = evaluate_results(routed, run_tickets)
        evaluated = append_stage(routed, PipelineStage.EVALUATION_COMPUTED)
        validation_errors = validate_final_results(evaluated, run_tickets, schema)
        metrics["validation_passed"] = not validation_errors
        metrics["validation_errors"] = validation_errors
        _write_deterministic_json(Path(config.output_dir) / "metrics.json", metrics)
        if not validation_errors:
            finalized = append_stage(evaluated, PipelineStage.VALIDATION_COMPLETED)
            write_state_history(finalized, Path(config.output_dir))

        print("\nParsed structured classification and routing outputs:")
        display_results = finalized if not validation_errors else evaluated
        for result in display_results:
            print(f"- Ticket {result.ticket_id}")
            if result.classification is not None:
                print(json.dumps(result.classification.model_dump(), sort_keys=True))
                print(f"  Effective confidence: {result.confidence:g}")
                print(f"  Route: {result.route}")
                print(f"  Suggested reply: {result.suggested_reply!r}")
            else:
                print(f"  ERROR: {result.error}")
            print(f"  Stage history: {' -> '.join(result.stage_history)}")
        print("\nFinal metrics:")
        print(json.dumps(metrics, indent=2, sort_keys=True))
        if validation_errors:
            print("\nFinal validation: FAILED", file=sys.stderr)
            for issue in validation_errors:
                print(f"- {issue}", file=sys.stderr)
            return 1
        required_outputs = (
            "predictions.json",
            "errors.json",
            "state_history.json",
            "model_responses.json",
            "metrics.json",
        )
        missing_outputs = [
            name for name in required_outputs if not (Path(config.output_dir) / name).is_file()
        ]
        if missing_outputs:
            print(f"ERROR: missing output files: {', '.join(missing_outputs)}", file=sys.stderr)
            return 1
        print("Final validation: PASSED")
        print("Output files verified: " + ", ".join(required_outputs))
        print(f"Results saved to: {config.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

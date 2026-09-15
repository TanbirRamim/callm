"""Output validator: parse responses into ``output_schema``; re-ask with the errors on failure."""

from __future__ import annotations

import logging

from callm.errors import OutputValidationError
from callm.pipeline import CallState, Handler, Step
from callm.types import LLMResponse, Message
from callm.validation import ValidationFailure, feedback_message, schema_name, validate_text

logger = logging.getLogger("callm")


class ValidatorMiddleware:
    name = "validator"

    def handle(self, state: CallState, call_next: Handler) -> Step[LLMResponse]:
        schema = state.config.output_schema
        if schema is None or state.request.stream:
            return (yield from call_next(state))

        original = state.request
        attempts = state.config.validation_retries + 1
        errors: list[str] = []
        response: LLMResponse | None = None
        try:
            for attempt in range(attempts):
                response = yield from call_next(state)
                try:
                    response.parsed = validate_text(response.text, schema)
                    return response
                except ValidationFailure as failure:
                    errors = failure.errors
                if attempt + 1 >= attempts:
                    break
                state.record.validation_retries = attempt + 1
                logger.info(
                    "callm: output failed %s validation (attempt %d/%d): %s",
                    schema_name(schema),
                    attempt + 1,
                    attempts,
                    "; ".join(errors[:3]),
                )
                state.request = state.request.append(
                    Message("assistant", response.text or "(empty response)"),
                    Message("user", feedback_message(errors, schema)),
                )
        finally:
            state.request = original
        raise OutputValidationError(
            f"Model output did not match {schema_name(schema)} after {attempts} attempt(s): "
            + "; ".join(errors[:5]),
            errors=errors,
            raw_text=response.text if response is not None else "",
            attempts=attempts,
        )


__all__ = ["ValidatorMiddleware"]

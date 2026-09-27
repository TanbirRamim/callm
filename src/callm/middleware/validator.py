"""Output validator: parse responses into ``output_schema``; re-ask with the errors on failure."""

from __future__ import annotations

import logging

from callm.errors import OutputValidationError
from callm.pipeline import CallState, Handler, Step
from callm.types import LLMResponse, Message, Usage
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
        # Every attempt is paid for, so usage and cost are summed across attempts and
        # reported on the final response (or on the record if all attempts fail).
        spent = _Spend()
        try:
            for attempt in range(attempts):
                response = yield from call_next(state)
                spent.add(response)
                try:
                    response.parsed = validate_text(response.text, schema)
                    if attempt:
                        spent.apply_to(response)
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
        spent.apply_to_record(state)
        raise OutputValidationError(
            f"Model output did not match {schema_name(schema)} after {attempts} attempt(s): "
            + "; ".join(errors[:5]),
            errors=errors,
            raw_text=response.text if response is not None else "",
            attempts=attempts,
        )


class _Spend:
    """Running total of usage and cost over validation attempts."""

    def __init__(self) -> None:
        self.usage = Usage()
        self.cost: float | None = 0.0

    def add(self, response: LLMResponse) -> None:
        if response.cached:
            return
        usage = response.usage
        self.usage = Usage(
            input_tokens=self.usage.input_tokens + usage.input_tokens,
            output_tokens=self.usage.output_tokens + usage.output_tokens,
            cache_read_tokens=self.usage.cache_read_tokens + usage.cache_read_tokens,
            cache_write_tokens=self.usage.cache_write_tokens + usage.cache_write_tokens,
            reported=self.usage.reported and usage.reported,
        )
        # One attempt with an unknown cost makes the total unknown too.
        self.cost = (
            None if self.cost is None or response.cost is None else self.cost + response.cost
        )

    def apply_to(self, response: LLMResponse) -> None:
        response.usage = self.usage
        response.cost = self.cost

    def apply_to_record(self, state: CallState) -> None:
        record = state.record
        record.input_tokens = self.usage.input_tokens
        record.output_tokens = self.usage.output_tokens
        record.cache_read_tokens = self.usage.cache_read_tokens
        record.cache_write_tokens = self.usage.cache_write_tokens
        record.cost_usd = self.cost


__all__ = ["ValidatorMiddleware"]

"""Structured output validation with Pydantic v2."""

from __future__ import annotations

import json
import re
import threading
from typing import Any

from callm.errors import ConfigurationError, MissingDependencyError

_FENCE = re.compile(r"```(?:json|JSON)?[ \t]*\n?(.*?)```", re.DOTALL)
_adapters: dict[Any, Any] = {}
_adapters_lock = threading.Lock()


def _pydantic() -> Any:
    try:
        import pydantic
    except ImportError as exc:
        raise MissingDependencyError("output_schema validation", "pydantic", "validation") from exc
    if not pydantic.VERSION.startswith("2"):
        raise MissingDependencyError(
            "output_schema validation (Pydantic v2)", "pydantic>=2", "validation"
        )
    return pydantic


def type_adapter(schema: Any) -> Any:
    with _adapters_lock:
        try:
            cached = _adapters.get(schema)
        except TypeError:  # unhashable annotation
            cached = None
        if cached is not None:
            return cached
        pydantic = _pydantic()
        try:
            adapter = pydantic.TypeAdapter(schema)
        except Exception as exc:
            raise ConfigurationError(
                f"output_schema {schema!r} is not a valid Pydantic type: {exc}"
            ) from exc
        try:
            _adapters[schema] = adapter
        except TypeError:
            pass
        return adapter


def check_schema(schema: Any) -> None:
    """Fail fast (at decoration time) for unusable schemas."""
    type_adapter(schema)


def schema_name(schema: Any) -> str:
    name = getattr(schema, "__qualname__", None) or getattr(schema, "__name__", None)
    return str(name) if name else repr(schema)


def json_schema(schema: Any) -> dict[str, Any]:
    try:
        result: dict[str, Any] = type_adapter(schema).json_schema()
        return result
    except Exception:
        return {}


def schema_fingerprint(schema: Any) -> str:
    if schema is None:
        return ""
    return schema_name(schema) + ":" + json.dumps(json_schema(schema), sort_keys=True, default=str)


def candidate_json(text: str) -> list[str]:
    """Plausible JSON substrings of a model response, most likely first."""
    stripped = text.strip()
    candidates: list[str] = []
    if stripped:
        candidates.append(stripped)
    for match in _FENCE.finditer(text):
        inner = match.group(1).strip()
        if inner:
            candidates.append(inner)
    for opener, closer in (("{", "}"), ("[", "]")):
        start = stripped.find(opener)
        end = stripped.rfind(closer)
        if start != -1 and end > start:
            candidates.append(stripped[start : end + 1])
    return list(dict.fromkeys(candidates))


class ValidationFailure(Exception):
    """Internal: a single failed validation attempt."""

    def __init__(self, errors: list[str], text: str) -> None:
        self.errors = errors
        self.text = text
        super().__init__("; ".join(errors))


def _format_errors(exc: Any) -> list[str]:
    errors = getattr(exc, "errors", None)
    if callable(errors):
        formatted = []
        for error in errors():
            location = ".".join(str(part) for part in error.get("loc", ())) or "(root)"
            formatted.append(f"{location}: {error.get('msg')}")
        return formatted
    return [str(exc)]


def validate_text(text: str, schema: Any) -> Any:
    """Parse ``text`` into ``schema``. Raises :class:`ValidationFailure`."""
    pydantic = _pydantic()
    adapter = type_adapter(schema)
    if schema is str:
        return text
    last_errors: list[str] = ["the response was empty"]
    for candidate in candidate_json(text):
        try:
            return adapter.validate_json(candidate)
        except pydantic.ValidationError as exc:
            last_errors = _format_errors(exc)
    raise ValidationFailure(last_errors, text)


def validate_value(value: Any, schema: Any) -> Any:
    """Validate an already-parsed Python value (dict, list, model instance...)."""
    pydantic = _pydantic()
    adapter = type_adapter(schema)
    try:
        return adapter.validate_python(value)
    except pydantic.ValidationError as exc:
        raise ValidationFailure(_format_errors(exc), repr(value)[:2000]) from exc


def feedback_message(errors: list[str], schema: Any) -> str:
    schema_json = json.dumps(json_schema(schema), sort_keys=True, default=str)
    bullet_list = "\n".join(f"- {error}" for error in errors[:20])
    return (
        "Your previous response could not be parsed into the required format.\n"
        f"Validation errors:\n{bullet_list}\n\n"
        "Reply again with only a JSON value that matches this JSON Schema, "
        f"with no surrounding prose:\n{schema_json}"
    )


__all__ = [
    "ValidationFailure",
    "candidate_json",
    "check_schema",
    "feedback_message",
    "json_schema",
    "schema_fingerprint",
    "type_adapter",
    "validate_text",
    "validate_value",
]

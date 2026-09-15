"""Input sanitizer: injection detection and PII redaction, before anything is cached or sent."""

from __future__ import annotations

import logging
import threading

from callm.config import InjectionConfig, PIIConfig
from callm.errors import PIIDetectedError, PromptInjectionError
from callm.pipeline import CallState, Handler, Invoke, Step
from callm.security.injection import ClassifierInjectionDetector, HeuristicInjectionDetector
from callm.security.pii import PIIRedactor
from callm.types import LLMRequest, LLMResponse, Message

logger = logging.getLogger("callm")

_heuristic = HeuristicInjectionDetector()
_redactors: dict[tuple[tuple[str, ...], bool, str], PIIRedactor] = {}
_redactors_lock = threading.Lock()


def _redactor(config: PIIConfig) -> PIIRedactor:
    key = (config.entities, config.ner, config.ner_model)
    with _redactors_lock:
        redactor = _redactors.get(key)
        if redactor is None:
            redactor = PIIRedactor(config.entities, ner=config.ner, ner_model=config.ner_model)
            _redactors[key] = redactor
        return redactor


def redact_request(request: LLMRequest, config: PIIConfig) -> tuple[LLMRequest, dict[str, int]]:
    """Mask PII in every text segment of the configured roles."""
    redactor = _redactor(config)
    registry: dict[tuple[str, str], str] = {}
    counts: dict[str, int] = {}

    def mask(text: str) -> str:
        result = redactor.redact(text, registry)
        for entity, count in result.counts.items():
            counts[entity] = counts.get(entity, 0) + count
        return result.text

    messages: list[Message] = [
        message.map_text(mask) if message.role in config.roles else message
        for message in request.messages
    ]
    if not counts:
        return request, counts
    return request.with_messages(messages), counts


def score_injection(request: LLMRequest, config: InjectionConfig) -> tuple[float, list[str]]:
    best = 0.0
    matches: list[str] = []
    classifier = (
        ClassifierInjectionDetector(config.classifier, config.classifier_label)
        if config.classifier
        else None
    )
    for message in request.messages:
        if message.role not in config.roles:
            continue
        for text in message.iter_texts():
            result = _heuristic.score(text)
            if classifier is not None:
                model_result = classifier.score(text)
                result.score = max(result.score, model_result.score)
                result.matches.extend(model_result.matches)
            best = max(best, result.score)
            for match in result.matches:
                if match not in matches:
                    matches.append(match)
    return best, matches


class SecurityMiddleware:
    name = "security"

    def handle(self, state: CallState, call_next: Handler) -> Step[LLMResponse]:
        injection = state.config.injection
        pii = state.config.pii
        request = state.request

        if injection is not None:
            score, matches = yield Invoke(
                sync=lambda: score_injection(request, injection),
                offload=injection.classifier is not None,
            )
            state.record.injection_score = score
            if score >= injection.threshold:
                state.record.injection_flagged = True
                if injection.action == "block":
                    raise PromptInjectionError(score, matches)
                logger.warning(
                    "callm: possible prompt injection (score=%.2f, signals=%s) in %s",
                    score,
                    ", ".join(matches) or "-",
                    state.record.function or "call",
                )

        if pii is not None:
            redacted, counts = yield Invoke(
                sync=lambda: redact_request(request, pii), offload=pii.ner
            )
            if counts:
                state.record.pii_redactions = counts
                if pii.action == "block":
                    raise PIIDetectedError(list(counts))
                state.request = redacted

        return (yield from call_next(state))


__all__ = ["SecurityMiddleware", "redact_request", "score_injection"]

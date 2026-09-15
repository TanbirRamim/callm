"""Prompt injection detection.

:class:`HeuristicInjectionDetector` is a zero-dependency scorer built from weighted
patterns seen in real-world injection and jailbreak attempts (instruction overrides,
system prompt extraction, role hijacking, fake chat-template delimiters, hidden unicode).
Pattern weights are combined as independent evidence: ``score = 1 - prod(1 - w)``.

It is a first line of defence, not a guarantee: paraphrased or novel attacks can evade
any pattern list. For stronger coverage also enable an ML classifier
(``InjectionConfig(classifier="protectai/deberta-v3-base-prompt-injection-v2")``) and keep
treating model output as untrusted.
"""

from __future__ import annotations

import re
import threading
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from callm.errors import MissingDependencyError


@dataclass(frozen=True)
class InjectionPattern:
    name: str
    pattern: re.Pattern[str]
    weight: float


def _p(name: str, regex: str, weight: float) -> InjectionPattern:
    return InjectionPattern(name, re.compile(regex, re.IGNORECASE | re.DOTALL), weight)


_PREVIOUS = r"(?:previous|prior|above|earlier|preceding|foregoing|system|developer)"
_INSTRUCTIONS = r"(?:instructions?|prompts?|rules|directives|guidelines|constraints)"
#: Not preceded by a negation ("please don't ignore the previous instructions").
_NOT_NEGATED = r"(?<!\bnot\s)(?<!n't\s)(?<!never\s)"

PATTERNS: tuple[InjectionPattern, ...] = (
    _p(
        "ignore_instructions",
        rf"{_NOT_NEGATED}\b(?:ignore|disregard|forget|override)\s+"
        rf"(?:(?:all|any|every|of|the|my|your|these|those)\s+)*"
        rf"{_PREVIOUS}\s+(?:\w+\s+)?{_INSTRUCTIONS}\b",
        0.85,
    ),
    _p(
        "ignore_everything",
        r"\b(?:ignore|disregard|forget)\s+(?:all|everything|anything)\s+"
        r"(?:you(?:'ve| have)?\s+(?:been\s+told|learned)|(?:said|written)\s+(?:above|before))",
        0.8,
    ),
    _p(
        "new_instructions",
        r"(?:^|\n)\s*(?:new|updated|real|actual|revised)\s+(?:system\s+)?instructions?\s*[:\-]",
        0.6,
    ),
    _p(
        "reveal_system_prompt",
        r"\b(?:reveal|show|print|display|output|repeat|recite|leak|dump|tell\s+me|give\s+me)"
        r"\b(?:\s+\w+){0,3}?\s+(?:your|the)\s+"
        r"(?:(?:full|entire|exact|original|hidden|secret|initial)\s+)*"
        r"(?:system\s+prompt|system\s+message|initial\s+instructions|hidden\s+instructions"
        r"|instructions\s+above|developer\s+message|prompt\s+above)",
        0.7,
    ),
    _p(
        "ask_system_prompt",
        r"\bwhat(?:'s|\s+is|\s+are|\s+were)\s+your\s+(?:(?:full|exact|hidden|secret|initial)\s+)*"
        r"(?:system\s+prompt|system\s+message|hidden\s+instructions|initial\s+instructions)",
        0.6,
    ),
    _p(
        "repeat_verbatim",
        r"\brepeat\b(?:\s+\w+){0,4}\s+(?:above|before\s+this|verbatim|word\s+for\s+word)\b",
        0.4,
    ),
    _p(
        "role_hijack",
        r"\byou\s+are\s+(?:now|no\s+longer)\b(?:\s+\w+){0,3}\s*"
        r"(?:DAN|jailbroken|unfiltered|unrestricted|uncensored|evil|free\s+from|not\s+bound"
        r"|without\s+(?:any\s+)?(?:rules|restrictions|limits))",
        0.7,
    ),
    _p(
        "dan",
        r"\b(?:DAN\s+mode|do\s+anything\s+now|jailbreak(?:ed)?\s+mode"
        r"|developer\s+mode\s+(?:enabled|on))\b",
        0.7,
    ),
    _p(
        "no_restrictions",
        r"\b(?:act|behave|respond|pretend|roleplay)\b(?:\s+\w+){0,6}\s+"
        r"(?:without|with\s+no|free\s+of|ignoring)\s+(?:any\s+)?"
        r"(?:restrictions|rules|filters|limitations|guidelines|censorship|safety)",
        0.55,
    ),
    _p(
        "bypass_safety",
        r"\b(?:bypass|disable|turn\s+off|circumvent|override|deactivate|ignore)\s+"
        r"(?:(?:your|its|all|any|of|the\s+model's)\s+)+(?:safety\s+|content\s+|ethical\s+)?"
        r"(?:filters?|guardrails?|restrictions|policies|safeguards|moderation|guidelines)"
        r"|\b(?:bypass|disable|turn\s+off|circumvent|deactivate)\s+(?:the\s+)?"
        r"(?:safety|ethical)\s+(?:filters?|guardrails?|restrictions|safeguards|guidelines)",
        0.6,
    ),
    _p(
        "do_not_follow",
        r"\b(?:do\s+not|don't|stop|no\s+longer)\s+(?:follow|obey|adhere\s+to|comply\s+with)\s+"
        r"(?:your|the|any)\s+(?:\w+\s+)?(?:rules|instructions|guidelines|programming|policies)",
        0.65,
    ),
    _p(
        "chat_template_tokens",
        r"<\|(?:im_start|im_end|system|user|assistant|endoftext|start_header_id|end_header_id"
        r"|eot_id)\|>|\[/?INST\]|<<\s*/?SYS\s*>>|<\s*/?\s*(?:system|sys)\s*>",
        0.6,
    ),
    _p(
        "fake_role_header",
        r"(?:^|\n)\s*(?:#{2,}\s*)?(?:system|assistant|developer)\s*(?:message|prompt)?\s*:\s*\S",
        0.35,
    ),
    _p(
        "end_of_prompt",
        r"\bend\s+of\s+(?:the\s+)?(?:system\s+)?(?:prompt|instructions|context|document)\b"
        r"|-{3,}\s*end\b",
        0.4,
    ),
    _p(
        "from_now_on",
        r"\bfrom\s+now\s+on\b(?:\s+\w+){0,4}\s+"
        r"(?:you\s+(?:will|must|are)|respond|answer|always|never)",
        0.3,
    ),
    _p(
        "exfiltration",
        r"\b(?:send|post|upload|exfiltrate|forward|transmit)\b(?:\s+\w+){0,5}\s+(?:to|at)\s+"
        r"(?:https?://|\S+@\S+\.\w+|this\s+(?:url|endpoint|webhook))",
        0.35,
    ),
    _p("markdown_image_exfil", r"!\[[^\]]*\]\(\s*https?://[^)\s]*[?&][^)\s]*=", 0.3),
    # High-signal non-English variants.
    _p(
        "ignore_instructions_de",
        r"\bignorier(?:e|en)?\b(?:\s+\w+){0,3}\s+(?:vorherigen|obigen|bisherigen|vorigen)\s+"
        r"(?:anweisungen|instruktionen|regeln)",
        0.85,
    ),
    _p(
        "ignore_instructions_es",
        r"\bignora(?:r)?\b(?:\s+\w+){0,3}\s+(?:instrucciones|reglas)\s+(?:anteriores|previas)",
        0.85,
    ),
    _p(
        "ignore_instructions_fr",
        r"\bignor(?:e|ez|er)\b(?:\s+\w+){0,3}\s+(?:instructions|consignes|règles)\s+"
        r"(?:précédentes|ci-dessus|antérieures)",
        0.85,
    ),
)

_INVISIBLE = re.compile("[​-‏‪-‮⁠-⁤⁦-⁩﻿\U000e0000-\U000e007f]")
_BASE64_BLOB = re.compile(r"(?:[A-Za-z0-9+/]{4}){60,}(?:[A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=)?")


@dataclass
class InjectionResult:
    score: float
    matches: list[str] = field(default_factory=list)

    def is_injection(self, threshold: float = 0.5) -> bool:
        return self.score >= threshold


def normalize(text: str) -> str:
    """NFKC-normalize, drop invisible characters and collapse horizontal whitespace."""
    text = unicodedata.normalize("NFKC", text)
    text = _INVISIBLE.sub("", text)
    return re.sub(r"[ \t\r\f\v]+", " ", text)


class HeuristicInjectionDetector:
    def __init__(self, patterns: Iterable[InjectionPattern] = PATTERNS) -> None:
        self.patterns = tuple(patterns)

    def score(self, text: str) -> InjectionResult:
        if not text:
            return InjectionResult(0.0)
        weights: list[tuple[str, float]] = []
        if len(_INVISIBLE.findall(text)) >= 3:
            weights.append(("hidden_unicode", 0.35))
        normalized = normalize(text)
        for pattern in self.patterns:
            if pattern.pattern.search(normalized):
                weights.append((pattern.name, pattern.weight))
        if _BASE64_BLOB.search(normalized):
            weights.append(("encoded_payload", 0.2))
        remaining = 1.0
        for _, weight in weights:
            remaining *= 1.0 - weight
        return InjectionResult(round(1.0 - remaining, 4), [name for name, _ in weights])


class ClassifierInjectionDetector:
    """Local Hugging Face text-classification model (requires ``transformers``)."""

    _pipelines: dict[str, Any] = {}
    _lock = threading.Lock()

    def __init__(self, model: str, label: str = "INJECTION") -> None:
        self.model = model
        self.label = label.upper()

    def _pipeline(self) -> Any:
        with self._lock:
            pipe = self._pipelines.get(self.model)
            if pipe is None:
                try:
                    from transformers import pipeline
                except ImportError as exc:
                    raise MissingDependencyError(
                        "Prompt injection classifier", "transformers", "security"
                    ) from exc
                pipe = pipeline(
                    "text-classification", model=self.model, truncation=True, max_length=512
                )
                self._pipelines[self.model] = pipe
            return pipe

    def score(self, text: str) -> InjectionResult:
        if not text.strip():
            return InjectionResult(0.0)
        outputs = self._pipeline()(text[:8000], top_k=None)
        if outputs and isinstance(outputs[0], list):
            outputs = outputs[0]
        for item in outputs:
            if str(item.get("label", "")).upper() == self.label:
                score = float(item.get("score", 0.0))
                return InjectionResult(score, [f"classifier:{self.model}"] if score >= 0.5 else [])
        return InjectionResult(0.0)


_default = HeuristicInjectionDetector()


def detect_injection(text: str) -> InjectionResult:
    """Score ``text`` with the default heuristic detector."""
    return _default.score(text)


__all__ = [
    "PATTERNS",
    "ClassifierInjectionDetector",
    "HeuristicInjectionDetector",
    "InjectionPattern",
    "InjectionResult",
    "detect_injection",
    "normalize",
]

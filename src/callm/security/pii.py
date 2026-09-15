"""PII detection and redaction.

Regex detectors cover structured identifiers: emails, phone numbers, US SSNs, payment
cards (Luhn-validated), IPv4 addresses and IBANs (checksum-validated). Person names need
NER: ``PIIRedactor(ner=True)`` uses spaCy's ``PERSON`` entities.

Placeholders are numbered per request and stable for repeated values, so
``"mail a@x.io, then a@x.io again"`` becomes ``"mail [EMAIL_1], then [EMAIL_1] again"``.
"""

from __future__ import annotations

import re
import threading
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

from callm.errors import MissingDependencyError

_EMAIL = re.compile(
    r"(?<![\w.+-])[A-Za-z0-9](?:[A-Za-z0-9._%+-]{0,62}[A-Za-z0-9_%+-])?"
    r"@(?:[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?\.)+[A-Za-z]{2,24}(?![\w-])"
)
_SSN = re.compile(r"(?<![\d-])(?!000|666|9\d\d)\d{3}-(?!00)\d{2}-(?!0000)\d{4}(?![\d-])")
_CARD = re.compile(r"(?<![\d-])(?:\d[ -]?){12,18}\d(?![\d-])")
#: International numbers written with a leading "+", or North American 3-3-4 numbers.
#: Bare digit groups without either shape are too ambiguous (timestamps, ids, measurements).
_PHONE = re.compile(
    r"(?<![\w+])\+\d{1,3}(?:[\s.-]?\(\d{1,4}\))?(?:[\s.-]?\d{1,5}){2,5}(?![\w-])"
    r"|(?<![\w+.:/-])(?:\(\d{3}\)\s?|\d{3}[-.\s])\d{3}[-.\s]\d{4}(?![\w.:/-])"
)
_IPV4 = re.compile(
    r"(?<![\d.])(?:(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)\.){3}"
    r"(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)(?![\d.])"
)
_IBAN = re.compile(
    r"(?<![A-Za-z0-9])[A-Z]{2}\d{2}(?:[ ]?[A-Z0-9]{4}){2,7}(?:[ ]?[A-Z0-9]{1,3})?(?![A-Za-z0-9])"
)
_DATE_LIKE = re.compile(r"^\d{4}[-./]\d{1,2}[-./]\d{1,2}$|^\d{1,2}[-./]\d{1,2}[-./]\d{2,4}$")
_DOTTED_QUAD = re.compile(r"^\d{1,3}(?:\.\d{1,3}){3}$")


def luhn_valid(number: str) -> bool:
    digits = [int(c) for c in number if c.isdigit()]
    if not 13 <= len(digits) <= 19:
        return False
    checksum = 0
    for index, digit in enumerate(reversed(digits)):
        if index % 2 == 1:
            digit *= 2
            if digit > 9:
                digit -= 9
        checksum += digit
    return checksum % 10 == 0


def card_valid(value: str) -> bool:
    """Luhn check plus a plausible issuer prefix and digit grouping."""
    if not luhn_valid(value):
        return False
    digits = "".join(c for c in value if c.isdigit())
    prefix = int(digits[:4])
    if digits[0] not in "3456" and not 2221 <= prefix <= 2720:
        return False
    groups = [len(g) for g in re.split(r"[ -]", value.strip()) if g]
    if len(groups) == 1:
        return True
    if groups in ([4, 6, 5], [4, 6, 4]):  # American Express / Diners
        return True
    return all(size == 4 for size in groups[:-1]) and 1 <= groups[-1] <= 4


def iban_valid(value: str) -> bool:
    compact = value.replace(" ", "")
    if not 15 <= len(compact) <= 34:
        return False
    rearranged = compact[4:] + compact[:4]
    try:
        numeric = "".join(str(int(ch, 36)) for ch in rearranged)
    except ValueError:
        return False
    return int(numeric) % 97 == 1


def _phone_valid(value: str) -> bool:
    stripped = value.strip()
    if _DATE_LIKE.match(stripped) or _DOTTED_QUAD.match(stripped):
        return False
    digits = sum(ch.isdigit() for ch in stripped)
    return 8 <= digits <= 15 if stripped.startswith("+") else digits == 10


@dataclass(frozen=True)
class PIIFinding:
    entity: str
    start: int
    end: int
    value: str = field(repr=False)


@dataclass
class RedactionResult:
    text: str
    findings: list[PIIFinding]

    @property
    def counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for finding in self.findings:
            counts[finding.entity] = counts.get(finding.entity, 0) + 1
        return counts


Detector = Callable[[str], Iterable[tuple[int, int]]]


def _regex_detector(
    pattern: re.Pattern[str], validate: Callable[[str], bool] | None = None
) -> Detector:
    def detect(text: str) -> Iterable[tuple[int, int]]:
        for match in pattern.finditer(text):
            if validate is None or validate(match.group(0)):
                yield match.start(), match.end()

    return detect


_VERSION_CONTEXT = re.compile(r"(?:\bversion|\bver\.?|\brelease|\bbuild|\bv)\s*$", re.IGNORECASE)


def _ip_detector(text: str) -> Iterable[tuple[int, int]]:
    for match in _IPV4.finditer(text):
        if _VERSION_CONTEXT.search(text[max(match.start() - 12, 0) : match.start()]):
            continue  # "version 2.10.3.4", "v1.2.3.4"
        yield match.start(), match.end()


_DETECTORS: dict[str, Detector] = {
    "email": _regex_detector(_EMAIL),
    "iban": _regex_detector(_IBAN, iban_valid),
    "credit_card": _regex_detector(_CARD, card_valid),
    "ssn": _regex_detector(_SSN),
    "ip_address": _ip_detector,
    "phone": _regex_detector(_PHONE, _phone_valid),
}
#: More specific detectors claim their spans first.
_ORDER = ("email", "iban", "credit_card", "ssn", "ip_address", "phone")

_spacy_models: dict[str, Any] = {}
_spacy_lock = threading.Lock()


def _load_spacy(model: str) -> Any:
    with _spacy_lock:
        if model in _spacy_models:
            return _spacy_models[model]
        try:
            import spacy
        except ImportError as exc:
            raise MissingDependencyError(
                "PII name detection (ner=True)", "spacy", "security"
            ) from exc
        try:
            nlp = spacy.load(model, disable=["parser", "lemmatizer", "textcat"])
        except OSError as exc:
            raise MissingDependencyError(
                f"PII name detection (run `python -m spacy download {model}`)",
                model,
                "security",
            ) from exc
        _spacy_models[model] = nlp
        return nlp


class PIIRedactor:
    """Find and mask PII in text."""

    def __init__(
        self,
        entities: Iterable[str] = ("email", "phone", "ssn", "credit_card", "ip_address", "iban"),
        *,
        ner: bool = False,
        ner_model: str = "en_core_web_sm",
        placeholder: str = "[{entity}_{index}]",
    ) -> None:
        self.entities = tuple(entities)
        self.ner = ner or "person" in self.entities
        self.ner_model = ner_model
        self.placeholder = placeholder

    def find(self, text: str) -> list[PIIFinding]:
        if not text:
            return []
        claimed: list[tuple[int, int]] = []
        findings: list[PIIFinding] = []

        def overlaps(start: int, end: int) -> bool:
            return any(start < c_end and end > c_start for c_start, c_end in claimed)

        for entity in _ORDER:
            if entity not in self.entities:
                continue
            for start, end in _DETECTORS[entity](text):
                if not overlaps(start, end):
                    claimed.append((start, end))
                    findings.append(PIIFinding(entity, start, end, text[start:end]))

        if self.ner:
            for ent in _load_spacy(self.ner_model)(text).ents:
                if ent.label_ == "PERSON" and not overlaps(ent.start_char, ent.end_char):
                    claimed.append((ent.start_char, ent.end_char))
                    findings.append(PIIFinding("person", ent.start_char, ent.end_char, ent.text))
        findings.sort(key=lambda f: f.start)
        return findings

    def redact(
        self, text: str, registry: dict[tuple[str, str], str] | None = None
    ) -> RedactionResult:
        """Mask PII in ``text``.

        Pass the same ``registry`` dict across several texts (e.g. all messages of one
        request) to keep placeholder numbering consistent between them.
        """
        registry = {} if registry is None else registry
        findings = self.find(text)
        if not findings:
            return RedactionResult(text, [])
        pieces: list[str] = []
        cursor = 0
        for finding in findings:
            key = (finding.entity, finding.value)
            placeholder = registry.get(key)
            if placeholder is None:
                index = 1 + sum(1 for entity, _ in registry if entity == finding.entity)
                placeholder = self.placeholder.format(entity=finding.entity.upper(), index=index)
                registry[key] = placeholder
            pieces.append(text[cursor : finding.start])
            pieces.append(placeholder)
            cursor = finding.end
        pieces.append(text[cursor:])
        return RedactionResult("".join(pieces), findings)


def redact_pii(text: str, entities: Iterable[str] | None = None, *, ner: bool = False) -> str:
    """Return ``text`` with PII replaced by placeholders such as ``[EMAIL_1]``."""
    redactor = PIIRedactor(entities, ner=ner) if entities is not None else PIIRedactor(ner=ner)
    return redactor.redact(text).text


__all__ = [
    "PIIFinding",
    "PIIRedactor",
    "RedactionResult",
    "card_valid",
    "iban_valid",
    "luhn_valid",
    "redact_pii",
]

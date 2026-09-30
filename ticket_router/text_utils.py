"""Small text helpers shared by the analyzers."""

from __future__ import annotations

import re

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")


def combine(subject: str, description: str) -> str:
    return f"{subject}. {description}"


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def count_phrases(text: str, phrases: list[str] | set[str]) -> int:
    """Count occurrences of single words or multi-word phrases on word boundaries."""
    lowered = text.lower()
    total = 0
    for phrase in phrases:
        total += len(re.findall(rf"\b{re.escape(phrase)}\b", lowered))
    return total

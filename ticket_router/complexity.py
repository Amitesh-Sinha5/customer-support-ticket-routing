"""Estimate how complex a ticket is to resolve.

The score combines interpretable signals so routing decisions can be explained:
length, technical depth, number of distinct issues, repeat-contact / prior
troubleshooting, and whether the ticket spans several categories.
"""

from __future__ import annotations

from . import config
from .keyword_categorizer import KeywordCategorizer
from .schemas import ComplexityResult
from .text_utils import combine, count_phrases, tokenize

TECHNICAL_TERMS = [
    "api", "error", "code", "server", "database", "integration", "sync", "timeout",
    "ssl", "certificate", "webhook", "configuration", "config", "version", "browser",
    "cache", "log", "logs", "stack trace", "exception", "endpoint", "token", "dns",
    "firewall", "sdk", "latency", "500", "502", "503", "404", "403", "ios", "android",
    "windows", "macos", "linux", "export", "import", "csv", "json", "crash", "crashes",
]

MULTI_ISSUE_MARKERS = [
    "also", "additionally", "another issue", "another problem", "on top of that",
    "as well", "second issue", "in addition", "furthermore", "not only",
]

PRIOR_ATTEMPT_MARKERS = [
    "already tried", "tried", "still", "again", "multiple times", "several times",
    "third time", "second time", "for weeks", "for days", "reinstalled", "restarted",
    "cleared", "contacted", "reached out", "no response", "followed the steps",
    "nothing works", "keeps happening", "nobody answered", "no one answered", "no reply",
    "still waiting", "haven't heard back", "last email",
]

WEIGHTS = {
    "length": 0.15,
    "technical_depth": 0.30,
    "multiple_issues": 0.20,
    "prior_attempts": 0.25,
    "cross_category": 0.10,
}


class ComplexityAnalyzer:
    def __init__(self, keyword_categorizer: KeywordCategorizer | None = None):
        self.keyword_categorizer = keyword_categorizer or KeywordCategorizer()

    def analyze(self, subject: str, description: str) -> ComplexityResult:
        text = combine(subject, description)
        words = len(tokenize(text))

        category_scores = self.keyword_categorizer.scores(subject, description)
        strong_categories = sum(1 for s in category_scores.values() if s >= 3)

        factors = {
            "length": min(words / 150, 1.0),
            "technical_depth": min(count_phrases(text, TECHNICAL_TERMS) / 5, 1.0),
            "multiple_issues": min(
                (count_phrases(text, MULTI_ISSUE_MARKERS) + max(text.count("?") - 1, 0)) / 2, 1.0
            ),
            "prior_attempts": min(count_phrases(text, PRIOR_ATTEMPT_MARKERS) / 3, 1.0),
            "cross_category": 1.0 if strong_categories >= 2 else 0.0,
        }
        score = sum(WEIGHTS[name] * value for name, value in factors.items())
        return ComplexityResult(
            score=round(score, 3),
            level=self._level(score),
            factors={k: round(v, 3) for k, v in factors.items()},
        )

    @staticmethod
    def _level(score: float) -> str:
        if score >= config.COMPLEXITY_HIGH_THRESHOLD:
            return "high"
        if score >= config.COMPLEXITY_MEDIUM_THRESHOLD:
            return "medium"
        return "low"

"""Predict resolution time and escalation risk from ticket features."""

from __future__ import annotations

import math

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

from . import config
from .schemas import (
    Category,
    ComplexityResult,
    EscalationPrediction,
    Priority,
    SentimentResult,
)
from .text_utils import combine, count_phrases, tokenize

# Phrases that historically precede a ticket being escalated to a supervisor.
ESCALATION_SIGNALS = [
    "manager", "supervisor", "escalate", "lawyer", "legal action", "legal", "sue",
    "chargeback", "dispute", "cancel my account", "cancel my subscription", "switch to",
    "competitor", "complaint", "report you", "bbb", "social media", "review", "last chance",
    "unacceptable",
]

CATEGORIES = list(Category)

FEATURE_NAMES = (
    [f"category_{c.value}" for c in CATEGORIES]
    + ["priority", "sentiment", "complexity", "log_words", "escalation_signals"]
)


def count_escalation_signals(subject: str, description: str) -> int:
    return count_phrases(combine(subject, description), ESCALATION_SIGNALS)


def build_features(category: Category, priority: Priority, sentiment_score: float,
                   complexity_score: float, word_count: int, escalation_signals: int) -> list[float]:
    one_hot = [1.0 if category == c else 0.0 for c in CATEGORIES]
    return one_hot + [
        float(priority.rank),
        sentiment_score,
        complexity_score,
        math.log1p(word_count),
        float(min(escalation_signals, 5)),
    ]


def features_for_ticket(subject: str, description: str, category: Category, priority: Priority,
                        sentiment: SentimentResult, complexity: ComplexityResult) -> list[float]:
    return build_features(
        category=category,
        priority=priority,
        sentiment_score=sentiment.score,
        complexity_score=complexity.score,
        word_count=len(tokenize(combine(subject, description))),
        escalation_signals=count_escalation_signals(subject, description),
    )


def new_resolution_model() -> GradientBoostingRegressor:
    # Trained on log(hours): resolution times are right-skewed.
    return GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=0.05,
                                     subsample=0.8, random_state=42)


def new_escalation_model() -> GradientBoostingClassifier:
    return GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.05,
                                      subsample=0.8, random_state=42)


class ResolutionTimePredictor:
    def __init__(self, model: GradientBoostingRegressor):
        self.model = model

    def predict_hours(self, features: list[float]) -> float:
        log_hours = self.model.predict(np.array([features]))[0]
        return round(float(math.expm1(log_hours)), 1)


class EscalationPredictor:
    def __init__(self, model: GradientBoostingClassifier, threshold: float = config.ESCALATION_THRESHOLD):
        self.model = model
        self.threshold = threshold

    def predict(self, features: list[float], priority: Priority, sentiment: SentimentResult,
                complexity: ComplexityResult, escalation_signals: int) -> EscalationPrediction:
        probability = float(self.model.predict_proba(np.array([features]))[0][1])
        return EscalationPrediction(
            probability=round(probability, 3),
            likely=probability >= self.threshold,
            reasons=self._reasons(priority, sentiment, complexity, escalation_signals),
        )

    @staticmethod
    def _reasons(priority: Priority, sentiment: SentimentResult, complexity: ComplexityResult,
                 escalation_signals: int) -> list[str]:
        """Human-readable risk factors, derived from the same features the model uses."""
        reasons = []
        if sentiment.label == "very_negative":
            reasons.append("Customer sentiment is very negative")
        elif sentiment.label == "negative":
            reasons.append("Customer sentiment is negative")
        if escalation_signals:
            reasons.append("Ticket mentions escalation language (manager, dispute, cancel, legal...)")
        if complexity.level == "high":
            reasons.append("Ticket complexity is high")
        if complexity.factors.get("prior_attempts", 0) >= 0.6:
            reasons.append("Customer reports repeated contact or failed troubleshooting")
        if priority in (Priority.HIGH, Priority.URGENT):
            reasons.append(f"Priority is {priority.value}")
        return reasons

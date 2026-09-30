"""Basic, rule-based ticket categorization using weighted keywords."""

from __future__ import annotations

from .schemas import Category
from .text_utils import count_phrases

# Keyword -> weight. Multi-word phrases are stronger signals than single words.
CATEGORY_KEYWORDS: dict[Category, dict[str, float]] = {
    Category.BILLING: {
        "bill": 1, "billing": 2, "billed": 2, "invoice": 2, "charge": 1.5, "charged": 2,
        "refund": 2, "payment": 2, "paid": 1, "pay": 1, "price": 1, "pricing": 1.5,
        "subscription": 1.5, "credit card": 2, "debit": 1, "receipt": 1.5, "overcharged": 3,
        "double charged": 3, "charged twice": 3, "plan": 0.5, "discount": 1, "coupon": 1.5,
        "transaction": 1.5, "fee": 1.5, "money back": 2, "chargeback": 3, "unauthorized": 2.5,
        "authorize": 1.5, "money": 1, "paid for": 2,
    },
    Category.TECHNICAL: {
        "error": 2, "bug": 2, "crash": 2, "crashes": 2, "crashing": 2, "not working": 1.5,
        "broken": 1.5, "fails": 1.5, "failed": 1, "failure": 1.5, "freeze": 2, "freezes": 2,
        "slow": 1, "loading": 1, "timeout": 2, "api": 2, "server": 1.5, "sync": 1.5,
        "integration": 2, "install": 1.5, "update": 0.5, "upload": 1, "export": 1,
        "browser": 1, "app": 0.5, "500": 1.5, "404": 1.5, "glitch": 2, "outage": 2,
        "down": 1, "database": 1.5, "configuration": 1.5, "ssl": 2, "webhook": 2,
        "blank": 1.5, "force quit": 2, "notifications": 1, "nothing happens": 1.5,
    },
    Category.ACCOUNT: {
        "account": 0.75, "password": 2.5, "login": 2, "log in": 2, "sign in": 2, "locked": 2,
        "locked out": 3, "username": 2, "email address": 1.5, "two-factor": 2.5, "2fa": 2.5,
        "verification code": 2, "reset": 1, "profile": 1.5, "deactivate": 2, "delete my account": 3,
        "suspended": 2, "permissions": 1.5, "team member": 1.5, "access": 1, "security": 1,
        "credentials": 2, "hacked": 2.5, "recovery email": 2.5, "login alert": 2,
    },
    Category.SHIPPING: {
        "shipping": 2, "shipment": 2, "shipped": 2, "delivery": 2, "delivered": 2,
        "package": 2, "parcel": 2, "tracking": 2.5, "tracking number": 3, "courier": 2,
        "arrive": 1, "arrived": 1.5, "damaged": 1.5, "wrong item": 3, "return label": 3,
        "address": 1, "order": 0.5, "warehouse": 1.5, "carrier": 2, "lost": 1, "dispatch": 2,
        "return": 1, "send back": 2, "driver": 1.5,
    },
    Category.GENERAL: {
        "question": 1, "information": 1, "info": 1, "feature": 0.5, "feedback": 2,
        "suggestion": 2, "hours": 1, "partnership": 2, "how do i": 1, "wondering": 1.5,
        "curious": 1.5, "recommend": 1, "general": 1, "inquiry": 1.5, "student discount": 1,
        "office": 1,
    },
}

SUBJECT_WEIGHT = 2.0  # subject lines are short and usually on-topic


class KeywordCategorizer:
    def __init__(self, keywords: dict[Category, dict[str, float]] | None = None):
        self.keywords = keywords or CATEGORY_KEYWORDS

    def scores(self, subject: str, description: str) -> dict[Category, float]:
        result: dict[Category, float] = {}
        for category, words in self.keywords.items():
            score = 0.0
            for phrase, weight in words.items():
                score += weight * (
                    SUBJECT_WEIGHT * count_phrases(subject, [phrase])
                    + count_phrases(description, [phrase])
                )
            result[category] = score
        return result

    def categorize(self, subject: str, description: str) -> tuple[Category, dict[Category, float]]:
        scores = self.scores(subject, description)
        best = max(scores, key=scores.get)
        if scores[best] == 0:
            return Category.GENERAL, scores
        return best, scores

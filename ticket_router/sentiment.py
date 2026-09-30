"""Lexicon-based sentiment analysis tuned for customer-support language.

Handles negation ("not happy"), intensifiers ("very", "extremely"), exclamation
marks and SHOUTING, which are all common in frustrated support tickets.
"""

from __future__ import annotations

import math
import re

from .schemas import SentimentResult

LEXICON: dict[str, float] = {
    # positive
    "thanks": 1.5, "thank": 1.5, "appreciate": 2.0, "appreciated": 2.0, "great": 2.0,
    "good": 1.5, "love": 2.5, "happy": 2.0, "pleased": 2.0, "excellent": 3.0,
    "helpful": 2.0, "awesome": 2.5, "amazing": 2.5, "wonderful": 2.5, "glad": 1.5,
    "nice": 1.5, "perfect": 2.5, "fantastic": 3.0, "satisfied": 2.0, "smooth": 1.0,
    "quick": 1.0, "easy": 1.0, "enjoy": 2.0, "resolved": 1.0, "fixed": 1.0,
    # negative
    "bad": -1.5, "poor": -2.0, "terrible": -3.0, "horrible": -3.0, "awful": -3.0,
    "worst": -3.5, "hate": -3.0, "angry": -3.0, "furious": -3.5, "frustrated": -2.5,
    "frustrating": -2.5, "annoyed": -2.0, "annoying": -2.0, "disappointed": -2.5,
    "disappointing": -2.5, "unacceptable": -3.0, "ridiculous": -3.0, "useless": -3.0,
    "upset": -2.5, "unhappy": -2.5, "problem": -1.0, "issue": -0.5, "broken": -2.0,
    "fail": -1.5, "fails": -1.5, "failed": -1.5, "failing": -1.5, "error": -1.0,
    "wrong": -1.5, "never": -1.0, "waste": -2.5, "wasted": -2.5, "scam": -3.5,
    "fraud": -3.0, "incompetent": -3.5, "pathetic": -3.5, "disgusted": -3.5,
    "outraged": -3.5, "worried": -1.5, "confused": -1.0, "stuck": -1.5, "delay": -1.0,
    "delayed": -1.5, "late": -1.0, "damaged": -2.0, "lost": -1.5, "crash": -1.5,
    "crashes": -1.5, "slow": -1.0, "unfortunately": -1.0, "sadly": -1.0, "sick": -2.0,
    "tired": -1.5, "nightmare": -3.0, "unusable": -3.0, "rude": -2.5, "ignored": -2.5,
}

NEGATIONS = {"not", "no", "never", "don't", "doesn't", "didn't", "isn't", "wasn't",
             "can't", "cannot", "won't", "haven't", "hasn't", "nothing", "neither"}

INTENSIFIERS = {"very": 0.3, "really": 0.3, "extremely": 0.5, "so": 0.2, "totally": 0.3,
                "completely": 0.4, "absolutely": 0.4, "incredibly": 0.5, "super": 0.3,
                "highly": 0.3, "seriously": 0.3}

_WORD_RE = re.compile(r"[A-Za-z]+(?:'[a-z]+)?")
ALPHA = 15.0  # normalisation constant, same idea as VADER


class SentimentAnalyzer:
    def analyze(self, text: str) -> SentimentResult:
        words = _WORD_RE.findall(text)
        total = 0.0
        for i, raw in enumerate(words):
            word = raw.lower()
            valence = LEXICON.get(word)
            if valence is None:
                continue
            window = [w.lower() for w in words[max(0, i - 3):i]]
            for prev in window:
                if prev in INTENSIFIERS:
                    valence *= 1 + INTENSIFIERS[prev]
            if any(prev in NEGATIONS for prev in window):
                valence *= -0.6
            if raw.isupper() and len(raw) > 2:
                valence *= 1.4
            total += valence

        if total != 0:
            exclamations = min(text.count("!"), 4)
            total += math.copysign(0.3 * exclamations, total)

        score = total / math.sqrt(total * total + ALPHA)
        return SentimentResult(score=round(score, 3), label=self._label(score))

    @staticmethod
    def _label(score: float) -> str:
        if score >= 0.3:
            return "positive"
        if score > -0.3:
            return "neutral"
        if score > -0.65:
            return "negative"
        return "very_negative"

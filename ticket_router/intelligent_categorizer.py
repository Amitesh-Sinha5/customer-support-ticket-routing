"""ML-based ticket categorization with a keyword fallback.

A TF-IDF + logistic-regression model trained on historical tickets captures
wording the keyword list does not. When the model is not confident, the
keyword categorizer's answer is used instead (if it found any keywords).
"""

from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

from . import config
from .keyword_categorizer import KeywordCategorizer
from .schemas import CategorizationResult, Category
from .text_utils import combine


def build_categorizer_pipeline() -> Pipeline:
    features = FeatureUnion([
        ("words", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True,
                                  stop_words="english")),
        ("chars", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3,
                                  sublinear_tf=True)),
    ])
    return Pipeline([
        ("features", features),
        ("clf", LogisticRegression(max_iter=2000, C=5.0)),
    ])


class IntelligentCategorizer:
    def __init__(self, model: Pipeline, keyword_categorizer: KeywordCategorizer | None = None,
                 threshold: float = config.ML_CONFIDENCE_THRESHOLD):
        self.model = model
        self.keyword_categorizer = keyword_categorizer or KeywordCategorizer()
        self.threshold = threshold

    def categorize(self, subject: str, description: str) -> CategorizationResult:
        keyword_category, keyword_scores = self.keyword_categorizer.categorize(subject, description)

        probabilities = self.model.predict_proba([combine(subject, description)])[0]
        best_index = int(probabilities.argmax())
        ml_category = Category(self.model.classes_[best_index])
        ml_confidence = float(probabilities[best_index])

        has_keywords = keyword_scores[keyword_category] > 0
        if ml_confidence >= self.threshold or not has_keywords:
            category, confidence, method = ml_category, ml_confidence, "ml"
        else:
            category, method = keyword_category, "keyword"
            confidence = keyword_scores[keyword_category] / (sum(keyword_scores.values()) or 1)

        return CategorizationResult(
            category=category,
            confidence=round(confidence, 3),
            method=method,
            keyword_category=keyword_category,
            keyword_scores={c.value: round(s, 2) for c, s in keyword_scores.items()},
        )

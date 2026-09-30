"""Train and persist the ML models from historical tickets."""

from __future__ import annotations

import csv
import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, mean_absolute_error, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from . import config
from .complexity import ComplexityAnalyzer
from .intelligent_categorizer import build_categorizer_pipeline
from .predictors import (
    features_for_ticket,
    new_escalation_model,
    new_resolution_model,
)
from .schemas import Category, Priority
from .sentiment import SentimentAnalyzer
from .text_utils import combine

logger = logging.getLogger(__name__)


@dataclass
class TrainedModels:
    categorizer: Pipeline
    resolution: object
    escalation: object


def load_history(path: Path = config.HISTORICAL_TICKETS_CSV) -> list[dict]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def train_models(history_path: Path = config.HISTORICAL_TICKETS_CSV) -> tuple[TrainedModels, dict]:
    rows = load_history(history_path)
    texts = [combine(r["subject"], r["description"]) for r in rows]
    labels = [r["category"] for r in rows]

    sentiment = SentimentAnalyzer()
    complexity = ComplexityAnalyzer()
    features = []
    for r in rows:
        features.append(features_for_ticket(
            r["subject"], r["description"], Category(r["category"]), Priority(r["priority"]),
            sentiment.analyze(combine(r["subject"], r["description"])),
            complexity.analyze(r["subject"], r["description"]),
        ))
    X = np.array(features)
    y_hours = np.array([math.log1p(float(r["resolution_hours"])) for r in rows])
    y_escalated = np.array([int(r["escalated"]) for r in rows])

    idx_train, idx_test = train_test_split(
        np.arange(len(rows)), test_size=0.2, random_state=42, stratify=labels
    )

    # Evaluate on a held-out split, then refit on all data for serving.
    categorizer = build_categorizer_pipeline()
    categorizer.fit([texts[i] for i in idx_train], [labels[i] for i in idx_train])
    cat_accuracy = accuracy_score([labels[i] for i in idx_test],
                                  categorizer.predict([texts[i] for i in idx_test]))

    resolution = new_resolution_model().fit(X[idx_train], y_hours[idx_train])
    mae_hours = mean_absolute_error(np.expm1(y_hours[idx_test]),
                                    np.expm1(resolution.predict(X[idx_test])))

    escalation = new_escalation_model().fit(X[idx_train], y_escalated[idx_train])
    esc_auc = roc_auc_score(y_escalated[idx_test], escalation.predict_proba(X[idx_test])[:, 1])

    metrics = {
        "training_rows": len(rows),
        "categorizer_accuracy": round(float(cat_accuracy), 3),
        "resolution_mae_hours": round(float(mae_hours), 2),
        "escalation_roc_auc": round(float(esc_auc), 3),
        "escalation_base_rate": round(float(y_escalated.mean()), 3),
    }

    models = TrainedModels(
        categorizer=build_categorizer_pipeline().fit(texts, labels),
        resolution=new_resolution_model().fit(X, y_hours),
        escalation=new_escalation_model().fit(X, y_escalated),
    )
    return models, metrics


def save_models(models: TrainedModels, metrics: dict) -> None:
    config.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(models.categorizer, config.CATEGORIZER_MODEL_PATH)
    joblib.dump(models.resolution, config.RESOLUTION_MODEL_PATH)
    joblib.dump(models.escalation, config.ESCALATION_MODEL_PATH)
    config.METRICS_PATH.write_text(json.dumps(metrics, indent=2))


def load_or_train() -> TrainedModels:
    paths = (config.CATEGORIZER_MODEL_PATH, config.RESOLUTION_MODEL_PATH, config.ESCALATION_MODEL_PATH)
    if all(p.exists() for p in paths):
        return TrainedModels(*(joblib.load(p) for p in paths))
    logger.info("No trained models found in %s; training from %s",
                config.ARTIFACTS_DIR, config.HISTORICAL_TICKETS_CSV)
    models, metrics = train_models()
    save_models(models, metrics)
    return models

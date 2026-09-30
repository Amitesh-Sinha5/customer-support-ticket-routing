"""Orchestrates every analysis step for an incoming ticket."""

from __future__ import annotations

import uuid

from . import config
from .complexity import ComplexityAnalyzer
from .intelligent_categorizer import IntelligentCategorizer
from .keyword_categorizer import KeywordCategorizer
from .predictors import (
    EscalationPredictor,
    ResolutionTimePredictor,
    count_escalation_signals,
    features_for_ticket,
)
from .queues import QueueAssigner
from .responses import DraftGenerator, ResponseSuggester, load_templates
from .routing import AgentRouter, load_agents
from .schemas import Ticket, TicketAnalysis, TicketCreate
from .sentiment import SentimentAnalyzer
from .text_utils import combine
from .training import TrainedModels, load_or_train


class TicketProcessor:
    """Runs the full pipeline:

    categorize -> assign queue -> sentiment -> complexity -> predict resolution
    time & escalation -> route to agent -> suggest responses -> draft reply
    """

    def __init__(self, models: TrainedModels, router: AgentRouter,
                 suggester: ResponseSuggester, drafter: DraftGenerator):
        keywords = KeywordCategorizer()
        self.categorizer = IntelligentCategorizer(models.categorizer, keywords)
        self.queue_assigner = QueueAssigner()
        self.sentiment = SentimentAnalyzer()
        self.complexity = ComplexityAnalyzer(keywords)
        self.resolution = ResolutionTimePredictor(models.resolution)
        self.escalation = EscalationPredictor(models.escalation)
        self.router = router
        self.suggester = suggester
        self.drafter = drafter

    @classmethod
    def from_defaults(cls, use_llm: bool | None = None) -> "TicketProcessor":
        return cls(
            models=load_or_train(),
            router=AgentRouter(load_agents(config.AGENTS_JSON)),
            suggester=ResponseSuggester(load_templates(config.RESPONSE_TEMPLATES_JSON)),
            drafter=DraftGenerator(use_llm=use_llm),
        )

    def process(self, payload: TicketCreate) -> Ticket:
        ticket_id = f"TCK-{uuid.uuid4().hex[:8].upper()}"
        subject, description, priority = payload.subject, payload.description, payload.priority

        categorization = self.categorizer.categorize(subject, description)
        category = categorization.category
        queue = self.queue_assigner.assign(category)

        sentiment = self.sentiment.analyze(combine(subject, description))
        complexity = self.complexity.analyze(subject, description)

        features = features_for_ticket(subject, description, category, priority, sentiment, complexity)
        hours = self.resolution.predict_hours(features)
        escalation = self.escalation.predict(
            features, priority, sentiment, complexity, count_escalation_signals(subject, description)
        )

        routing = self.router.route(category, queue, complexity, priority, escalation)

        suggestions = self.suggester.suggest(subject, description, category)
        draft = self.drafter.draft(ticket_id, subject, description, category, sentiment, hours,
                                   suggestions)

        return Ticket(
            id=ticket_id,
            subject=subject,
            description=description,
            priority=priority,
            analysis=TicketAnalysis(
                categorization=categorization,
                sentiment=sentiment,
                complexity=complexity,
                predicted_resolution_hours=hours,
                escalation=escalation,
                routing=routing,
                suggested_responses=suggestions,
                draft_reply=draft,
            ),
        )

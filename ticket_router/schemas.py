"""Domain models shared by every component and exposed through the API."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field


class Priority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"

    @property
    def rank(self) -> int:
        return list(Priority).index(self)


class Category(str, Enum):
    BILLING = "Billing"
    TECHNICAL = "Technical"
    ACCOUNT = "Account"
    SHIPPING = "Shipping"
    GENERAL = "General"


class TicketStatus(str, Enum):
    OPEN = "open"
    RESOLVED = "resolved"


class TicketCreate(BaseModel):
    """Incoming ticket payload: the three fields the system receives."""

    subject: str = Field(..., min_length=1, max_length=200)
    description: str = Field(..., min_length=1, max_length=5000)
    priority: Priority = Priority.MEDIUM


class CategorizationResult(BaseModel):
    category: Category
    confidence: float
    method: str  # "ml" or "keyword"
    keyword_category: Category
    keyword_scores: dict[str, float]


class SentimentResult(BaseModel):
    score: float  # compound score in [-1, 1]
    label: str  # positive | neutral | negative | very_negative


class ComplexityResult(BaseModel):
    score: float  # [0, 1]
    level: str  # low | medium | high
    factors: dict[str, float]


class EscalationPrediction(BaseModel):
    probability: float
    likely: bool
    reasons: list[str]


class RoutingDecision(BaseModel):
    queue: str
    agent_id: str | None
    agent_name: str | None
    required_level: int
    score: float | None
    reason: str


class SuggestedResponse(BaseModel):
    template_id: str
    title: str
    relevance: float
    body: str


class DraftReply(BaseModel):
    text: str
    source: str  # "llm" or "template"


class TicketAnalysis(BaseModel):
    categorization: CategorizationResult
    sentiment: SentimentResult
    complexity: ComplexityResult
    predicted_resolution_hours: float
    escalation: EscalationPrediction
    routing: RoutingDecision
    suggested_responses: list[SuggestedResponse]
    draft_reply: DraftReply


class Ticket(BaseModel):
    id: str
    subject: str
    description: str
    priority: Priority
    status: TicketStatus = TicketStatus.OPEN
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    analysis: TicketAnalysis


class Agent(BaseModel):
    id: str
    name: str
    team: str  # queue this agent belongs to
    level: int = Field(..., ge=1, le=3)  # 1 junior, 2 intermediate, 3 senior
    skills: dict[Category, float]  # proficiency per category in [0, 1]
    max_load: int = Field(..., ge=1)
    current_load: int = 0

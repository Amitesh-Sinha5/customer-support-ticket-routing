"""Route tickets to the most appropriate agent based on complexity and expertise."""

from __future__ import annotations

import json
import threading
from pathlib import Path

from .schemas import (
    Agent,
    Category,
    ComplexityResult,
    EscalationPrediction,
    Priority,
    RoutingDecision,
)

LEVEL_FOR_COMPLEXITY = {"low": 1, "medium": 2, "high": 3}
LEVEL_NAMES = {1: "junior", 2: "intermediate", 3: "senior"}

MIN_EXPERTISE = 0.5  # minimum proficiency in the ticket's category to be a candidate

# Scoring weights for candidate agents.
W_EXPERTISE = 0.5
W_LEVEL_FIT = 0.25
W_AVAILABILITY = 0.25


def load_agents(path: Path) -> list[Agent]:
    return [Agent(**raw) for raw in json.loads(path.read_text())]


def required_level(complexity: ComplexityResult, priority: Priority,
                   escalation: EscalationPrediction) -> int:
    level = LEVEL_FOR_COMPLEXITY[complexity.level]
    if priority == Priority.URGENT or escalation.likely:
        level += 1
    return min(level, 3)


class AgentRouter:
    def __init__(self, agents: list[Agent]):
        self.agents = {a.id: a for a in agents}
        self._lock = threading.Lock()

    def route(self, category: Category, queue: str, complexity: ComplexityResult,
              priority: Priority, escalation: EscalationPrediction) -> RoutingDecision:
        needed = required_level(complexity, priority, escalation)
        with self._lock:
            candidates = [
                a for a in self.agents.values()
                if a.current_load < a.max_load and a.skills.get(category, 0) >= MIN_EXPERTISE
            ]
            qualified = [a for a in candidates if a.level >= needed]
            pool, relaxed = (qualified, False) if qualified else (candidates, True)

            if not pool:
                return RoutingDecision(
                    queue=queue, agent_id=None, agent_name=None, required_level=needed, score=None,
                    reason=f"No available agent with {category.value} expertise; ticket waits in queue.",
                )

            best = max(pool, key=lambda a: self._score(a, category, needed))
            score = self._score(best, category, needed)
            best.current_load += 1

        reason = (
            f"{LEVEL_NAMES[best.level].capitalize()} agent, {category.value} proficiency "
            f"{best.skills[category]:.2f}, load {best.current_load}/{best.max_load}; "
            f"ticket needs {'an' if needed == 2 else 'a'} {LEVEL_NAMES[needed]} agent ({complexity.level} complexity"
            f"{', escalation risk' if escalation.likely else ''}"
            f"{', urgent' if priority == Priority.URGENT else ''})."
        )
        if relaxed:
            reason += " No agent at the required level was free, so the best available was chosen."
        return RoutingDecision(queue=queue, agent_id=best.id, agent_name=best.name,
                               required_level=needed, score=round(score, 3), reason=reason)

    def release(self, agent_id: str | None) -> None:
        if agent_id is None:
            return
        with self._lock:
            agent = self.agents.get(agent_id)
            if agent and agent.current_load > 0:
                agent.current_load -= 1

    def list_agents(self) -> list[Agent]:
        with self._lock:
            return [a.model_copy() for a in self.agents.values()]

    @staticmethod
    def _score(agent: Agent, category: Category, needed: int) -> float:
        expertise = agent.skills.get(category, 0)
        # Prefer the least-senior qualified agent so seniors stay free for hard tickets.
        level_fit = max(0.0, 1 - 0.3 * abs(agent.level - needed))
        availability = 1 - agent.current_load / agent.max_load
        return W_EXPERTISE * expertise + W_LEVEL_FIT * level_fit + W_AVAILABILITY * availability

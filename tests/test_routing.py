from ticket_router.routing import AgentRouter, required_level
from ticket_router.schemas import (
    Agent,
    Category,
    ComplexityResult,
    EscalationPrediction,
    Priority,
)

LOW = ComplexityResult(score=0.1, level="low", factors={})
HIGH = ComplexityResult(score=0.8, level="high", factors={})
CALM = EscalationPrediction(probability=0.1, likely=False, reasons=[])
RISKY = EscalationPrediction(probability=0.9, likely=True, reasons=[])


def make_router() -> AgentRouter:
    return AgentRouter([
        Agent(id="J", name="Junior", team="Billing", level=1, max_load=2, skills={Category.BILLING: 0.7}),
        Agent(id="S", name="Senior", team="Billing", level=3, max_load=2, skills={Category.BILLING: 0.95}),
        Agent(id="T", name="Tech", team="Technical Support", level=3, max_load=2,
              skills={Category.TECHNICAL: 0.9, Category.BILLING: 0.3}),
    ])


def test_required_level_rises_with_urgency_and_risk():
    assert required_level(LOW, Priority.LOW, CALM) == 1
    assert required_level(LOW, Priority.URGENT, CALM) == 2
    assert required_level(HIGH, Priority.LOW, RISKY) == 3


def test_simple_ticket_goes_to_junior():
    decision = make_router().route(Category.BILLING, "Billing", LOW, Priority.LOW, CALM)
    assert decision.agent_id == "J"


def test_complex_ticket_goes_to_senior():
    decision = make_router().route(Category.BILLING, "Billing", HIGH, Priority.HIGH, CALM)
    assert decision.agent_id == "S"
    assert decision.required_level == 3


def test_agent_without_expertise_is_never_chosen():
    router = make_router()
    for _ in range(4):
        decision = router.route(Category.BILLING, "Billing", LOW, Priority.LOW, CALM)
        assert decision.agent_id in {"J", "S"}


def test_load_is_tracked_and_released():
    router = make_router()
    decisions = [router.route(Category.BILLING, "Billing", LOW, Priority.LOW, CALM) for _ in range(4)]
    assert {d.agent_id for d in decisions} == {"J", "S"}

    full = router.route(Category.BILLING, "Billing", LOW, Priority.LOW, CALM)
    assert full.agent_id is None  # everyone with Billing expertise is at max load

    router.release("J")
    assert router.route(Category.BILLING, "Billing", LOW, Priority.LOW, CALM).agent_id == "J"


def test_falls_back_to_less_senior_agent_when_seniors_are_busy():
    router = make_router()
    router.agents["S"].current_load = 2
    decision = router.route(Category.BILLING, "Billing", HIGH, Priority.URGENT, RISKY)
    assert decision.agent_id == "J"
    assert "best available" in decision.reason

from types import SimpleNamespace

from ticket_router.responses import DraftGenerator
from ticket_router.schemas import Category, TicketCreate

ANGRY = TicketCreate(
    subject="Charged twice AGAIN",
    description="This is the third time I've contacted you. You charged my card twice and nobody answered. "
                "This is completely unacceptable! I want to speak to a manager or I will file a chargeback.",
    priority="urgent",
)
CALM = TicketCreate(subject="Business hours", description="What are your support hours on weekends?",
                    priority="low")


def test_full_pipeline_produces_complete_analysis(processor):
    ticket = processor.process(ANGRY)
    a = ticket.analysis
    assert ticket.id.startswith("TCK-")
    assert a.categorization.category == Category.BILLING
    assert a.routing.queue == "Billing"
    assert a.sentiment.label in {"negative", "very_negative"}
    assert a.predicted_resolution_hours > 0
    assert a.routing.agent_id is not None
    assert len(a.suggested_responses) == 3
    assert a.suggested_responses[0].template_id.startswith("BIL-")
    assert ticket.id in a.draft_reply.text
    assert a.draft_reply.source == "template"


def test_angry_ticket_has_higher_escalation_risk_than_calm_one(processor):
    angry = processor.process(ANGRY).analysis.escalation
    calm = processor.process(CALM).analysis.escalation
    assert angry.probability > calm.probability
    assert angry.likely and not calm.likely
    assert angry.reasons


def test_categorization_on_unseen_wording(processor):
    cases = {
        ("Money taken without permission", "There is a debit I did not authorize."): Category.BILLING,
        ("Screen goes white", "Clicking settings makes everything go blank and I have to force quit."): Category.TECHNICAL,
        ("Hacked?", "Someone changed my recovery email and I got a login alert."): Category.ACCOUNT,
        ("Courier left my box at wrong house", "The delivery driver dropped it at a neighbour."): Category.SHIPPING,
        ("Office location", "Just curious where your office is."): Category.GENERAL,
    }
    for (subject, description), expected in cases.items():
        result = processor.process(TicketCreate(subject=subject, description=description)).analysis
        assert result.categorization.category == expected, subject


class FakeClient:
    """Stands in for anthropic.Anthropic so the LLM path is tested without network calls."""

    def __init__(self, stop_reason="end_turn", text='{"reply": "Hello from Claude"}'):
        response = SimpleNamespace(stop_reason=stop_reason, content=[SimpleNamespace(type="text", text=text)])
        self.calls = []

        def create(**kwargs):
            self.calls.append(kwargs)
            return response

        self.beta = SimpleNamespace(messages=SimpleNamespace(create=create))


def _drafter_with(client) -> DraftGenerator:
    drafter = DraftGenerator(use_llm=False)
    drafter._client = client
    return drafter


def test_llm_draft_is_used_when_available(processor):
    client = FakeClient()
    processor.drafter = _drafter_with(client)
    draft = processor.process(ANGRY).analysis.draft_reply
    assert draft.source == "llm"
    assert draft.text == "Hello from Claude"
    assert "<prewritten_responses>" in client.calls[0]["messages"][0]["content"]


def test_llm_refusal_falls_back_to_template(processor):
    processor.drafter = _drafter_with(FakeClient(stop_reason="refusal", text=""))
    assert processor.process(ANGRY).analysis.draft_reply.source == "template"


def test_malformed_llm_output_falls_back_to_template(processor):
    processor.drafter = _drafter_with(FakeClient(text="not json"))
    assert processor.process(ANGRY).analysis.draft_reply.source == "template"

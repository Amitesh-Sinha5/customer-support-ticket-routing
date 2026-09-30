from ticket_router.chat import (
    BotMessage,
    ChatSession,
    DraftedReply,
    State,
    TicketCreated,
    derive_subject,
    parse_priority,
)
from ticket_router.schemas import Priority


def test_parse_priority():
    assert parse_priority("1") == Priority.LOW
    assert parse_priority("4") == Priority.URGENT
    assert parse_priority("it's pretty urgent") == Priority.URGENT
    assert parse_priority("High") == Priority.HIGH
    assert parse_priority("banana") is None


def test_derive_subject():
    assert derive_subject("My app crashes. It started yesterday.") == "My app crashes."
    long = derive_subject("word " * 40)
    assert len(long) <= 63 and long.endswith("...")


def test_full_conversation_creates_ticket(processor):
    chat = ChatSession(processor)
    assert isinstance(chat.greeting()[0], BotMessage)

    chat.handle("hello")
    assert chat.state == State.AWAITING_ISSUE

    chat.handle("my parcel is late")  # too short: bot asks for more detail
    assert chat.state == State.AWAITING_DETAILS

    chat.handle("It should have arrived 5 days ago and tracking has not moved since Monday.")
    assert chat.state == State.AWAITING_PRIORITY

    assert chat.state == State.AWAITING_PRIORITY and chat.handle("banana")[0].text.startswith("Please reply")

    events = chat.handle("3")
    ticket = next(e.ticket for e in events if isinstance(e, TicketCreated))
    assert ticket.subject == "my parcel is late"
    assert ticket.priority == Priority.HIGH
    assert ticket.analysis.routing.queue == "Shipping & Delivery"
    assert any(isinstance(e, DraftedReply) for e in events)
    assert chat.state == State.AWAITING_MORE

    chat.handle("no thanks")
    assert chat.done
    assert len(chat.store.list()) == 1


def test_detailed_first_message_skips_follow_up_and_new_issue_flow(processor):
    chat = ChatSession(processor)
    chat.handle("I was charged twice for my subscription this month and need a refund")
    assert chat.state == State.AWAITING_PRIORITY
    chat.handle("urgent")
    chat.handle("yes")
    assert chat.state == State.AWAITING_ISSUE
    chat.handle("The app crashes every time I upload a photo from my phone")
    chat.handle("2")
    queues = [t.analysis.routing.queue for t in chat.store.list()]
    assert queues == ["Billing", "Technical Support"]


def test_export_contains_conversation_and_analysis(processor, tmp_path):
    from ticket_router.chat_export import export_session

    chat = ChatSession(processor)
    chat.greeting()
    chat.handle("I was charged twice for my subscription this month and need a refund")
    events = chat.handle("3")
    ticket = next(e.ticket for e in events if isinstance(e, TicketCreated))

    path = export_session(chat, tmp_path)
    text = path.read_text()
    assert path.suffix == ".md"
    assert "**Customer**" in text and "> I was charged twice" in text
    assert "Hi, I'm Ava" in text
    assert f"| Reference | {ticket.id} |" in text
    assert "## Internal ticket analysis" in text
    assert "Escalation risk" in text

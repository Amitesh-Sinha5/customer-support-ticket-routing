"""Export a chat session as a readable Markdown transcript."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from . import config
from .chat import BOT_NAME, BotMessage, ChatSession, DraftedReply, TicketCreated
from .responses import format_hours
from .schemas import Ticket

EXPORTS_DIR = config.PROJECT_ROOT / "exports"


def _quote(text: str) -> str:
    return "\n".join(f"> {line}" if line else ">" for line in text.splitlines())


def _cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def _ticket_card(ticket: Ticket) -> str:
    a = ticket.analysis
    rows = [
        ("Reference", ticket.id),
        ("Subject", ticket.subject),
        ("Priority", ticket.priority.value.title()),
        ("Team", a.routing.queue),
        ("Assigned to", a.routing.agent_name or "Next available agent"),
        ("Expected fix", f"within about {format_hours(a.predicted_resolution_hours)}"),
        ("Status", ticket.status.value.title()),
    ]
    return "| Field | Value |\n|---|---|\n" + "\n".join(f"| {k} | {_cell(v)} |" for k, v in rows)


def _analysis(ticket: Ticket) -> str:
    a = ticket.analysis
    c, s, x, e, r = a.categorization, a.sentiment, a.complexity, a.escalation, a.routing
    factors = ", ".join(f"{k.replace('_', ' ')} {v:.2f}" for k, v in x.factors.items() if v > 0) or "none"
    lines = [
        f"### {ticket.id} - {ticket.subject}",
        "",
        f"- **Customer message:** {ticket.description}",
        f"- **Category:** {c.category.value} ({c.method}, confidence {c.confidence:.2f}; "
        f"keyword guess {c.keyword_category.value})",
        f"- **Sentiment:** {s.label.replace('_', ' ')} ({s.score:+.2f})",
        f"- **Complexity:** {x.level} ({x.score:.2f}; {factors})",
        f"- **Predicted resolution:** ~{a.predicted_resolution_hours} hours",
        f"- **Escalation risk:** {e.probability:.0%}{' - likely' if e.likely else ''}"
        + (f" ({'; '.join(e.reasons)})" if e.reasons else ""),
        f"- **Routing:** {r.queue} → {r.agent_name or 'unassigned'}. {r.reason}",
        "- **Suggested responses:** " + ", ".join(
            f"{t.template_id} {t.title} ({t.relevance:.2f})" for t in a.suggested_responses),
        f"- **Draft source:** {'Claude' if a.draft_reply.source == 'llm' else 'template library'}",
    ]
    return "\n".join(lines)


def transcript_to_markdown(session: ChatSession, exported_at: datetime | None = None) -> str:
    exported_at = exported_at or datetime.now()
    tickets = session.store.list()
    out = [
        "# Customer Support Chat Transcript",
        "",
        f"- **Session started:** {session.started_at:%Y-%m-%d %H:%M}",
        f"- **Exported:** {exported_at:%Y-%m-%d %H:%M}",
        f"- **Messages:** {sum(1 for t in session.transcript if t.speaker == 'customer')} from customer",
        f"- **Tickets opened:** {', '.join(t.id for t in tickets) or 'none'}",
        "",
        "## Conversation",
        "",
    ]
    for entry in session.transcript:
        stamp = f"{entry.time:%H:%M}"
        content = entry.content
        if entry.speaker == "customer":
            out += [f"**Customer** · {stamp}", "", _quote(content), ""]
        elif isinstance(content, BotMessage):
            out += [f"**{BOT_NAME} (Support)** · {stamp}", "", _quote(content.text), ""]
        elif isinstance(content, DraftedReply):
            out += [f"**{BOT_NAME} (Support)** · {stamp}", "", _quote(content.text), ""]
        elif isinstance(content, TicketCreated):
            out += [f"**🎫 Ticket created** · {stamp}", "", _ticket_card(content.ticket), ""]

    if tickets:
        out += ["---", "", "## Internal ticket analysis", "",
                "_For support staff only - not shown to the customer._", ""]
        for ticket in tickets:
            out += [_analysis(ticket), ""]
    return "\n".join(out).rstrip() + "\n"


def export_session(session: ChatSession, directory: Path = EXPORTS_DIR) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    now = datetime.now()
    path = directory / f"chat-{now:%Y%m%d-%H%M%S}.md"
    path.write_text(transcript_to_markdown(session, now), encoding="utf-8")
    return path

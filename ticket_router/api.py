"""REST API for receiving and inspecting support tickets.

Run with:  uvicorn ticket_router.api:app --reload
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse

from .chat import ChatSession, DraftedReply, Event, TicketCreated
from .chat_sessions import ChatSessionRegistry
from .pipeline import TicketProcessor
from .queues import QUEUES
from .schemas import (
    Agent,
    ChatEvent,
    ChatMessage,
    ChatTurn,
    Ticket,
    TicketCreate,
    TicketStatus,
)
from .store import TicketStore

STATIC_DIR = Path(__file__).resolve().parent / "static"

DESCRIPTION = """
Receives support tickets, categorises them, scores sentiment and complexity, predicts resolution
time and escalation risk, routes each ticket to the best-suited agent and drafts a first reply.

The [chat page](/) uses the **Chat** endpoints below. Tickets are held in memory and reset when
the service restarts.
"""


def _chat_event(event: Event) -> ChatEvent:
    if isinstance(event, TicketCreated):
        return ChatEvent(type="ticket", ticket=event.ticket)
    if isinstance(event, DraftedReply):
        return ChatEvent(type="draft", text=event.text, source=event.source)
    return ChatEvent(type="message", text=event.text)


def _chat_turn(session_id: str, session: ChatSession, events: list[Event]) -> ChatTurn:
    return ChatTurn(session_id=session_id, state=session.state.name.lower(),
                    events=[_chat_event(e) for e in events])


def create_app(processor: TicketProcessor | None = None) -> FastAPI:
    state: dict = {}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        state["processor"] = processor or TicketProcessor.from_defaults()
        state["store"] = TicketStore()
        state["chats"] = ChatSessionRegistry(state["processor"])
        yield

    app = FastAPI(
        title="Automated Customer Support Ticket Routing & Response",
        description=DESCRIPTION,
        version="1.0.0",
        lifespan=lifespan,
        # Hide the long schema list and open "Try it out" by default.
        swagger_ui_parameters={"defaultModelsExpandDepth": -1, "tryItOutEnabled": True},
    )

    @app.get("/", include_in_schema=False)
    def home() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/health", tags=["System"])
    def health() -> dict:
        """Service status and whether Claude is drafting the replies."""
        return {"status": "ok", "llm_drafting": state["processor"].drafter.use_llm}

    @app.post("/chat", response_model=ChatTurn, status_code=201, tags=["Chat"])
    def start_chat() -> ChatTurn:
        """Open a chat session with the support assistant and get its greeting."""
        session_id, session = state["chats"].create()
        return _chat_turn(session_id, session, session.greeting())

    @app.post("/chat/{session_id}/messages", response_model=ChatTurn, tags=["Chat"])
    def send_chat_message(session_id: str, payload: ChatMessage) -> ChatTurn:
        """Send a customer message; returns the assistant's replies and any ticket it opened."""
        session = state["chats"].get(session_id)
        if session is None:
            raise HTTPException(404, "Chat session not found or expired")
        events = session.handle(payload.text)
        for event in events:
            if isinstance(event, TicketCreated):
                state["store"].add(event.ticket)
        return _chat_turn(session_id, session, events)

    @app.post("/tickets", response_model=Ticket, status_code=201, tags=["Tickets"])
    def create_ticket(payload: TicketCreate) -> Ticket:
        """Submit a ticket and get the full analysis, routing decision and drafted reply."""
        ticket = state["processor"].process(payload)
        state["store"].add(ticket)
        return ticket

    @app.get("/tickets", response_model=list[Ticket], tags=["Tickets"])
    def list_tickets(queue: str | None = Query(None, description=f"One of: {', '.join(QUEUES)}"),
                     status: TicketStatus | None = None) -> list[Ticket]:
        """List tickets, optionally filtered by queue and status."""
        if queue and queue not in QUEUES:
            raise HTTPException(400, f"Unknown queue '{queue}'. Valid queues: {QUEUES}")
        return state["store"].list(queue=queue, status=status)

    @app.get("/tickets/{ticket_id}", response_model=Ticket, tags=["Tickets"])
    def get_ticket(ticket_id: str) -> Ticket:
        """Get one ticket with its analysis."""
        ticket = state["store"].get(ticket_id)
        if ticket is None:
            raise HTTPException(404, "Ticket not found")
        return ticket

    @app.post("/tickets/{ticket_id}/resolve", response_model=Ticket, tags=["Tickets"])
    def resolve_ticket(ticket_id: str) -> Ticket:
        """Mark a ticket resolved and free its assigned agent."""
        store: TicketStore = state["store"]
        existing = store.get(ticket_id)
        if existing is None:
            raise HTTPException(404, "Ticket not found")
        if existing.status == TicketStatus.RESOLVED:
            return existing
        ticket = store.mark_resolved(ticket_id)
        state["processor"].router.release(ticket.analysis.routing.agent_id)
        return ticket

    @app.get("/queues", tags=["Team"])
    def queues() -> dict[str, dict]:
        """Open-ticket count and ticket ids per support queue."""
        open_tickets = state["store"].list(status=TicketStatus.OPEN)
        return {
            q: {
                "open_tickets": len(ids := [t.id for t in open_tickets if t.analysis.routing.queue == q]),
                "ticket_ids": ids,
            }
            for q in QUEUES
        }

    @app.get("/agents", response_model=list[Agent], tags=["Team"])
    def agents() -> list[Agent]:
        """Support agents with their skills and current workload."""
        return state["processor"].router.list_agents()

    return app


app = create_app()

"""REST API for receiving and inspecting support tickets.

Run with:  uvicorn ticket_router.api:app --reload
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query

from .pipeline import TicketProcessor
from .queues import QUEUES
from .schemas import Agent, Ticket, TicketCreate, TicketStatus
from .store import TicketStore


def create_app(processor: TicketProcessor | None = None) -> FastAPI:
    state: dict = {}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        state["processor"] = processor or TicketProcessor.from_defaults()
        state["store"] = TicketStore()
        yield

    app = FastAPI(title="Automated Customer Support Ticket Routing & Response", lifespan=lifespan)

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "llm_drafting": state["processor"].drafter.use_llm}

    @app.post("/tickets", response_model=Ticket, status_code=201)
    def create_ticket(payload: TicketCreate) -> Ticket:
        ticket = state["processor"].process(payload)
        state["store"].add(ticket)
        return ticket

    @app.get("/tickets", response_model=list[Ticket])
    def list_tickets(queue: str | None = Query(None, description=f"One of: {', '.join(QUEUES)}"),
                     status: TicketStatus | None = None) -> list[Ticket]:
        if queue and queue not in QUEUES:
            raise HTTPException(400, f"Unknown queue '{queue}'. Valid queues: {QUEUES}")
        return state["store"].list(queue=queue, status=status)

    @app.get("/tickets/{ticket_id}", response_model=Ticket)
    def get_ticket(ticket_id: str) -> Ticket:
        ticket = state["store"].get(ticket_id)
        if ticket is None:
            raise HTTPException(404, "Ticket not found")
        return ticket

    @app.post("/tickets/{ticket_id}/resolve", response_model=Ticket)
    def resolve_ticket(ticket_id: str) -> Ticket:
        store: TicketStore = state["store"]
        existing = store.get(ticket_id)
        if existing is None:
            raise HTTPException(404, "Ticket not found")
        if existing.status == TicketStatus.RESOLVED:
            return existing
        ticket = store.mark_resolved(ticket_id)
        state["processor"].router.release(ticket.analysis.routing.agent_id)
        return ticket

    @app.get("/queues")
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

    @app.get("/agents", response_model=list[Agent])
    def agents() -> list[Agent]:
        return state["processor"].router.list_agents()

    return app


app = create_app()

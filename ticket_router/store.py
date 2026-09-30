"""Thread-safe in-memory ticket storage."""

from __future__ import annotations

import threading

from .schemas import Ticket, TicketStatus


class TicketStore:
    def __init__(self):
        self._tickets: dict[str, Ticket] = {}
        self._lock = threading.Lock()

    def add(self, ticket: Ticket) -> None:
        with self._lock:
            self._tickets[ticket.id] = ticket

    def get(self, ticket_id: str) -> Ticket | None:
        with self._lock:
            return self._tickets.get(ticket_id)

    def list(self, queue: str | None = None, status: TicketStatus | None = None) -> list[Ticket]:
        with self._lock:
            tickets = list(self._tickets.values())
        if queue:
            tickets = [t for t in tickets if t.analysis.routing.queue == queue]
        if status:
            tickets = [t for t in tickets if t.status == status]
        return sorted(tickets, key=lambda t: t.created_at)

    def mark_resolved(self, ticket_id: str) -> Ticket | None:
        with self._lock:
            ticket = self._tickets.get(ticket_id)
            if ticket is None:
                return None
            ticket.status = TicketStatus.RESOLVED
            return ticket

"""In-memory registry of the chat sessions opened from the web page."""

from __future__ import annotations

import threading
import uuid
from collections import OrderedDict

from .chat import ChatSession
from .pipeline import TicketProcessor

# Oldest sessions are dropped beyond this so abandoned chats cannot grow memory without bound.
MAX_SESSIONS = 200


class ChatSessionRegistry:
    def __init__(self, processor: TicketProcessor, max_sessions: int = MAX_SESSIONS):
        self._processor = processor
        self._max_sessions = max_sessions
        self._sessions: OrderedDict[str, ChatSession] = OrderedDict()
        self._lock = threading.Lock()

    def create(self) -> tuple[str, ChatSession]:
        session_id = uuid.uuid4().hex
        session = ChatSession(self._processor)
        with self._lock:
            self._sessions[session_id] = session
            while len(self._sessions) > self._max_sessions:
                self._sessions.popitem(last=False)
        return session_id, session

    def get(self, session_id: str) -> ChatSession | None:
        with self._lock:
            return self._sessions.get(session_id)

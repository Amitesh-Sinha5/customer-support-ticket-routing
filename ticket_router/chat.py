"""Conversation logic for the customer chat.

The chat collects the three ticket fields conversationally (the customer's
message becomes the description, a subject is derived from it, and the
customer picks a priority), runs the ticket through the TicketProcessor, and
replies with the drafted response. It has no UI code, so it can be driven by
the terminal client or by tests.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto

from .pipeline import TicketProcessor
from .schemas import Priority, Ticket, TicketCreate
from .store import TicketStore

BOT_NAME = "Ava"
MIN_DETAIL_WORDS = 6
MAX_SUBJECT_CHARS = 60

PRIORITY_CHOICES = {
    Priority.LOW: ("1", "low", "not urgent", "whenever"),
    Priority.MEDIUM: ("2", "medium", "normal", "moderate"),
    Priority.HIGH: ("3", "high", "important", "soon"),
    Priority.URGENT: ("4", "urgent", "critical", "asap", "emergency"),
}

GOODBYES = {"no", "nope", "nah", "no thanks", "no thank you", "that's all", "thats all",
            "that's it", "nothing", "nothing else", "bye", "goodbye", "all good", "i'm good"}
GREETINGS = {"hi", "hello", "hey", "hi there", "hello there", "good morning", "good afternoon",
             "good evening", "hiya", "yo"}
YESES = {"yes", "yeah", "yep", "sure", "yes please", "ok", "okay", "i do", "one more"}


@dataclass
class BotMessage:
    text: str


@dataclass
class TicketCreated:
    ticket: Ticket


@dataclass
class DraftedReply:
    text: str
    source: str  # "llm" or "template"


Event = BotMessage | TicketCreated | DraftedReply


@dataclass
class TranscriptEntry:
    speaker: str  # "customer" or "bot"
    content: str | Event
    time: datetime = field(default_factory=datetime.now)


class State(Enum):
    AWAITING_ISSUE = auto()
    AWAITING_DETAILS = auto()
    AWAITING_PRIORITY = auto()
    AWAITING_MORE = auto()
    DONE = auto()


def derive_subject(description: str) -> str:
    first = re.split(r"(?<=[.!?])\s+|\n", description.strip(), maxsplit=1)[0].strip()
    if len(first) > MAX_SUBJECT_CHARS:
        first = first[:MAX_SUBJECT_CHARS].rsplit(" ", 1)[0] + "..."
    return first or "Support request"


def parse_priority(text: str) -> Priority | None:
    lowered = text.strip().lower()
    for priority, options in PRIORITY_CHOICES.items():
        if lowered in options:
            return priority
    for priority, options in PRIORITY_CHOICES.items():
        if any(re.search(rf"\b{re.escape(o)}\b", lowered) for o in options if not o.isdigit()):
            return priority
    return None


def _normalise(text: str) -> str:
    return re.sub(r"[^\w\s']", "", text.lower()).strip()


class ChatSession:
    def __init__(self, processor: TicketProcessor, store: TicketStore | None = None):
        self.processor = processor
        self.store = store or TicketStore()
        self.state = State.AWAITING_ISSUE
        self._description = ""
        self._subject: str | None = None
        self.started_at = datetime.now()
        self.transcript: list[TranscriptEntry] = []

    @property
    def done(self) -> bool:
        return self.state == State.DONE

    def greeting(self) -> list[Event]:
        return self._record_bot([BotMessage(
            f"Hi, I'm {BOT_NAME} from Customer Support.\n"
            "Tell me what's going on and I'll get it to the right team."
        )])

    def handle(self, text: str) -> list[Event]:
        text = text.strip()
        if text:
            self.transcript.append(TranscriptEntry("customer", text))
        return self._record_bot(self._respond(text))

    def start_new_issue(self) -> list[Event]:
        self.state, self._description, self._subject = State.AWAITING_ISSUE, "", None
        return self._record_bot([BotMessage("Sure, let's start a new request. What can I help you with?")])

    def _record_bot(self, events: list[Event]) -> list[Event]:
        self.transcript.extend(TranscriptEntry("bot", e) for e in events)
        return events

    def _respond(self, text: str) -> list[Event]:
        if not text:
            return [BotMessage("Sorry, I didn't catch that. Could you type your message again?")]

        if self.state == State.AWAITING_ISSUE:
            return self._on_issue(text)
        if self.state == State.AWAITING_DETAILS:
            # The short first message makes a good subject; the details complete the description.
            self._subject = derive_subject(self._description)
            self._description = f"{self._description.rstrip('.')}. {text}"
            return self._ask_priority()
        if self.state == State.AWAITING_PRIORITY:
            return self._on_priority(text)
        if self.state == State.AWAITING_MORE:
            return self._on_more(text)
        return [BotMessage("This chat has ended. Start a new session to open another ticket.")]

    # ------------------------------------------------------------------ steps

    def _on_issue(self, text: str) -> list[Event]:
        if _normalise(text) in GREETINGS:
            return [BotMessage("Hello! What can I help you with today?")]
        self._description = text
        if len(text.split()) < MIN_DETAIL_WORDS:
            self.state = State.AWAITING_DETAILS
            return [BotMessage(
                "Could you tell me a bit more? For example, what happened, when it started, "
                "and any error messages or order numbers."
            )]
        return self._ask_priority()

    def _ask_priority(self) -> list[Event]:
        self.state = State.AWAITING_PRIORITY
        return [BotMessage(
            "Thanks for the details. How urgent is this for you?\n"
            "  1  Low      - whenever you get a chance\n"
            "  2  Medium   - I'd like it sorted soon\n"
            "  3  High     - it's affecting my work\n"
            "  4  Urgent   - I'm completely blocked"
        )]

    def _on_priority(self, text: str) -> list[Event]:
        priority = parse_priority(text)
        if priority is None:
            return [BotMessage("Please reply with a number from 1 to 4 (or low / medium / high / urgent).")]

        ticket = self.processor.process(TicketCreate(
            subject=self._subject or derive_subject(self._description),
            description=self._description[:5000],
            priority=priority,
        ))
        self.store.add(ticket)
        self.state, self._description, self._subject = State.AWAITING_MORE, "", None
        draft = ticket.analysis.draft_reply
        return [
            BotMessage("I've logged your request. Here's your ticket:"),
            TicketCreated(ticket),
            DraftedReply(draft.text, draft.source),
            BotMessage("Is there anything else I can help you with?"),
        ]

    def _on_more(self, text: str) -> list[Event]:
        normalised = _normalise(text)
        if normalised in GOODBYES:
            self.state = State.DONE
            count = len(self.store.list())
            return [BotMessage(
                f"Thanks for chatting with us. You have {count} open ticket{'s' if count != 1 else ''}, "
                "and our team will keep you updated by email. Have a great day!"
            )]
        if normalised in YESES:
            self.state = State.AWAITING_ISSUE
            return [BotMessage("Of course. What else can I help you with?")]
        return self._on_issue(text)

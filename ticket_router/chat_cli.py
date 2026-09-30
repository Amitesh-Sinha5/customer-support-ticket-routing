"""Terminal chat client: chat with support as a customer.

Usage:  python scripts/chat.py [--agent-view] [--no-llm]
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from datetime import datetime

from rich import box
from rich.align import Align
from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .chat import BOT_NAME, BotMessage, ChatSession, DraftedReply, Event, TicketCreated
from . import config
from .chat_export import export_session
from .pipeline import TicketProcessor
from .responses import format_hours
from .schemas import Ticket

MIN_TYPING_SECONDS = 0.7
PROMPT = "You › "

HELP_TEXT = (
    "[bold]/new[/]      start a new request\n"
    "[bold]/tickets[/]  list tickets opened in this chat\n"
    "[bold]/details[/]  toggle the internal agent view (routing, sentiment, predictions)\n"
    "[bold]/export[/]   save this whole chat as a readable Markdown file\n"
    "[bold]/clear[/]    clear the screen\n"
    "[bold]/help[/]     show this help\n"
    "[bold]/quit[/]     leave the chat"
)

SENTIMENT_STYLE = {"positive": "green", "neutral": "white", "negative": "yellow", "very_negative": "red"}
LEVEL_STYLE = {"low": "green", "medium": "yellow", "high": "red"}
PRIORITY_STYLE = {"low": "green", "medium": "cyan", "high": "yellow", "urgent": "bold red"}


class ChatUI:
    def __init__(self, session: ChatSession, console: Console, agent_view: bool):
        self.session = session
        self.console = console
        self.agent_view = agent_view

    # --------------------------------------------------------------- layout

    @property
    def bubble_width(self) -> int:
        return max(40, min(76, int(self.console.width * 0.75)))

    @staticmethod
    def _now() -> str:
        return datetime.now().strftime("%H:%M")

    def header(self) -> None:
        title = Text("Customer Support", style="bold white")
        subtitle = Text("We usually reply instantly · type /help for commands", style="dim")
        self.console.print(Panel(Group(Align.center(title), Align.center(subtitle)),
                                 box=box.HEAVY, border_style="cyan", padding=(0, 2)))
        self.console.print(Align.center(Text(f"● {BOT_NAME} is online", style="green")))
        self.console.print()

    def bot_bubble(self, text: str, caption: str | None = None) -> None:
        self.console.print(Panel(
            Text(text), title=f"[bold cyan]{BOT_NAME}[/] [dim]· Support[/]", title_align="left",
            subtitle=f"[dim]{caption + ' · ' if caption else ''}{self._now()}[/]", subtitle_align="left",
            border_style="cyan", box=box.ROUNDED, width=self.bubble_width, padding=(0, 1),
        ))

    def customer_bubble(self, text: str) -> None:
        self.console.print(Align.right(Panel(
            Text(text), title="[bold green]You[/]", title_align="right",
            subtitle=f"[dim]{self._now()} ✓✓[/]", subtitle_align="right",
            border_style="green", box=box.ROUNDED, width=min(self.bubble_width, max(len(text) + 4, 16)),
            padding=(0, 1),
        )))

    def ticket_card(self, ticket: Ticket) -> None:
        a = ticket.analysis
        grid = Table.grid(padding=(0, 2))
        grid.add_column(style="dim", justify="right")
        grid.add_column()
        grid.add_row("Reference", f"[bold]{ticket.id}[/]")
        grid.add_row("Subject", ticket.subject)
        grid.add_row("Priority", f"[{PRIORITY_STYLE[ticket.priority.value]}]{ticket.priority.value.title()}[/]")
        grid.add_row("Team", a.routing.queue)
        grid.add_row("Assigned to", a.routing.agent_name or "Next available agent")
        grid.add_row("Expected fix", f"within about {format_hours(a.predicted_resolution_hours)}")
        grid.add_row("Status", "[green]● Open[/]")
        self.console.print(Panel(grid, title="[bold magenta]🎫 Support ticket[/]", title_align="left",
                                 border_style="magenta", box=box.ROUNDED, width=self.bubble_width,
                                 padding=(0, 1)))

    def agent_panel(self, ticket: Ticket) -> None:
        a = ticket.analysis
        c, s, x, e, r = a.categorization, a.sentiment, a.complexity, a.escalation, a.routing
        table = Table(box=box.SIMPLE_HEAD, show_header=False, padding=(0, 1), expand=True)
        table.add_column(style="bold yellow", no_wrap=True)
        table.add_column()
        table.add_row("Category", f"{c.category.value}  [dim]({c.method}, confidence {c.confidence:.2f}; "
                                  f"keyword guess {c.keyword_category.value})[/]")
        table.add_row("Sentiment", f"[{SENTIMENT_STYLE[s.label]}]{s.label.replace('_', ' ')}[/] [dim]({s.score:+.2f})[/]")
        top_factors = ", ".join(f"{k.replace('_', ' ')} {v:.1f}" for k, v in x.factors.items() if v > 0)
        table.add_row("Complexity", f"[{LEVEL_STYLE[x.level]}]{x.level}[/] [dim]({x.score:.2f}"
                                    f"{'; ' + top_factors if top_factors else ''})[/]")
        esc_style = "bold red" if e.likely else "green"
        table.add_row("Escalation", f"[{esc_style}]{e.probability:.0%}{' - likely' if e.likely else ''}[/]"
                                    + (f"\n[dim]{'; '.join(e.reasons)}[/]" if e.reasons else ""))
        table.add_row("Resolution", f"~{a.predicted_resolution_hours} h")
        table.add_row("Routing", f"{r.queue} → {r.agent_name or 'unassigned'}\n[dim]{r.reason}[/]")
        table.add_row("Suggested", "\n".join(f"{t.template_id}  {t.title} [dim]({t.relevance:.2f})[/]"
                                             for t in a.suggested_responses))
        table.add_row("Draft", "Claude" if a.draft_reply.source == "llm" else "Template library")
        self.console.print(Panel(table, title="[bold yellow]Agent view · internal[/]", title_align="left",
                                 border_style="yellow", box=box.ROUNDED, width=self.bubble_width + 8))

    def tickets_table(self) -> None:
        tickets = self.session.store.list()
        if not tickets:
            self.bot_bubble("You haven't opened any tickets in this chat yet.")
            return
        table = Table(title="Your tickets", box=box.ROUNDED, border_style="magenta")
        for col in ("Reference", "Subject", "Priority", "Team", "Assigned to", "Status"):
            table.add_column(col)
        for t in tickets:
            table.add_row(t.id, t.subject, t.priority.value.title(), t.analysis.routing.queue,
                          t.analysis.routing.agent_name or "-", t.status.value.title())
        self.console.print(table)

    # ------------------------------------------------------------------ flow

    def render(self, events: list[Event]) -> None:
        last_ticket = None
        for event in events:
            if isinstance(event, BotMessage):
                self.bot_bubble(event.text)
            elif isinstance(event, TicketCreated):
                last_ticket = event.ticket
                self.ticket_card(event.ticket)
            elif isinstance(event, DraftedReply):
                caption = ("drafted by AI" if event.source == "llm" else "from response library") \
                    if self.agent_view else None
                self.bot_bubble(event.text, caption)
                if self.agent_view and last_ticket:
                    self.agent_panel(last_ticket)

    def typing(self, action):
        start = time.monotonic()
        with self.console.status(f"[cyan]{BOT_NAME} is typing…[/]", spinner="dots"):
            result = action()
            time.sleep(max(0.0, MIN_TYPING_SECONDS - (time.monotonic() - start)))
        return result

    def read_input(self) -> str:
        text = self.console.input(f"[bold green]{PROMPT}[/]")
        if self.console.is_terminal:
            # Replace the raw input line(s) with a chat bubble.
            lines = max(1, math.ceil((len(PROMPT) + len(text)) / max(self.console.width, 1)))
            sys.stdout.write("\x1b[1A\x1b[2K" * lines)
            sys.stdout.flush()
        return text

    def run(self) -> None:
        self.header()
        self.render(self.typing(self.session.greeting))
        while not self.session.done:
            try:
                text = self.read_input()
            except (EOFError, KeyboardInterrupt):
                self.console.print()
                break
            command = text.strip().lower()
            if not command:
                continue
            if command.startswith("/"):
                if not self.command(command):
                    break
                continue
            self.customer_bubble(text.strip())
            self.render(self.typing(lambda: self.session.handle(text)))
        self.console.print(Align.center(Text("Chat ended · thank you for contacting support", style="dim")))

    def command(self, command: str) -> bool:
        """Handle a slash command. Returns False when the chat should end."""
        if command in ("/quit", "/exit", "/q"):
            return False
        if command == "/help":
            self.console.print(Panel(HELP_TEXT, title="Commands", border_style="blue",
                                     width=self.bubble_width))
        elif command == "/new":
            self.render(self.typing(self.session.start_new_issue))
        elif command == "/tickets":
            self.tickets_table()
        elif command == "/export":
            path = export_session(self.session)
            self.console.print(Align.center(Text(f"Chat exported to {path.relative_to(config.PROJECT_ROOT)}", style="magenta")))
        elif command == "/details":
            self.agent_view = not self.agent_view
            state = "on" if self.agent_view else "off"
            self.console.print(Align.center(Text(f"Agent view {state}", style="yellow")))
            tickets = self.session.store.list()
            if self.agent_view and tickets:
                self.agent_panel(tickets[-1])
        elif command == "/clear":
            self.console.clear()
            self.header()
        else:
            self.console.print(Text(f"Unknown command {command}. Type /help for the list.", style="red"))
        return True


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Chat with customer support in the terminal.")
    parser.add_argument("--agent-view", action="store_true",
                        help="show the internal analysis (routing, sentiment, predictions) for each ticket")
    parser.add_argument("--no-llm", action="store_true", help="always draft replies from templates")
    args = parser.parse_args(argv)

    console = Console()
    console.clear()
    with console.status("[cyan]Connecting you to support…[/]", spinner="dots"):
        processor = TicketProcessor.from_defaults(use_llm=False if args.no_llm else None)
    ChatUI(ChatSession(processor), console, agent_view=args.agent_view).run()


if __name__ == "__main__":
    main()

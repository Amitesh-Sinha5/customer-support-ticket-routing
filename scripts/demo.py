"""Run a handful of sample tickets through the full pipeline and print the results.

Usage:  python scripts/demo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ticket_router.pipeline import TicketProcessor  # noqa: E402
from ticket_router.schemas import TicketCreate  # noqa: E402

SAMPLES = [
    TicketCreate(
        subject="Charged 3 times this month!!",
        description="I see three separate payments of $49 on my Visa for the same subscription. "
                    "This is the second time I've written in and nobody answered. This is unacceptable, "
                    "fix it or I will file a chargeback with my bank.",
        priority="high",
    ),
    TicketCreate(
        subject="Dashboard won't load after update",
        description="Since version 4.3.0 the dashboard shows a blank page. The browser console shows "
                    "a 502 from the /api/reports endpoint. I already cleared the cache and tried Firefox "
                    "and Chrome. Also the CSV export times out.",
        priority="urgent",
    ),
    TicketCreate(
        subject="Can't get into my account",
        description="I forgot my password and the reset email isn't coming through.",
        priority="medium",
    ),
    TicketCreate(
        subject="Where's my parcel?",
        description="Ordered a week ago, courier site hasn't changed status since Monday.",
        priority="low",
    ),
    TicketCreate(
        subject="Loving the new release",
        description="Great work on the latest version! Any plans to support offline mode?",
        priority="low",
    ),
]


def main() -> None:
    processor = TicketProcessor.from_defaults()
    for payload in SAMPLES:
        t = processor.process(payload)
        a = t.analysis
        print("=" * 78)
        print(f"{t.id}  [{t.priority.value}]  {t.subject}")
        print(f"  Category   : {a.categorization.category.value} ({a.categorization.method}, "
              f"conf {a.categorization.confidence}) | keyword guess: {a.categorization.keyword_category.value}")
        print(f"  Queue      : {a.routing.queue}")
        print(f"  Sentiment  : {a.sentiment.label} ({a.sentiment.score})")
        print(f"  Complexity : {a.complexity.level} ({a.complexity.score})")
        print(f"  Resolution : ~{a.predicted_resolution_hours} h")
        print(f"  Escalation : {a.escalation.probability:.0%} likely={a.escalation.likely} "
              f"{a.escalation.reasons}")
        print(f"  Routed to  : {a.routing.agent_name or 'unassigned'} - {a.routing.reason}")
        print(f"  Suggestions: {[s.template_id + ' ' + s.title for s in a.suggested_responses]}")
        print(f"  Draft ({a.draft_reply.source}):")
        print("    " + a.draft_reply.text.replace("\n", "\n    "))


if __name__ == "__main__":
    main()

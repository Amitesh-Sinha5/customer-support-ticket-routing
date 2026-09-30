"""Map ticket categories to support queues."""

from __future__ import annotations

from .schemas import Category

CATEGORY_TO_QUEUE: dict[Category, str] = {
    Category.BILLING: "Billing",
    Category.TECHNICAL: "Technical Support",
    Category.ACCOUNT: "Account Management",
    Category.SHIPPING: "Shipping & Delivery",
    Category.GENERAL: "General Support",
}

QUEUES: list[str] = list(CATEGORY_TO_QUEUE.values())


class QueueAssigner:
    def assign(self, category: Category) -> str:
        return CATEGORY_TO_QUEUE[category]

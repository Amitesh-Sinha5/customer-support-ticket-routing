import pytest

from ticket_router import config
from ticket_router.pipeline import TicketProcessor
from ticket_router.responses import DraftGenerator, ResponseSuggester, load_templates
from ticket_router.routing import AgentRouter, load_agents
from ticket_router.training import load_or_train


@pytest.fixture(scope="session")
def models():
    return load_or_train()


@pytest.fixture
def processor(models):
    """A fresh processor per test so agent workloads don't leak between tests."""
    return TicketProcessor(
        models=models,
        router=AgentRouter(load_agents(config.AGENTS_JSON)),
        suggester=ResponseSuggester(load_templates(config.RESPONSE_TEMPLATES_JSON)),
        drafter=DraftGenerator(use_llm=False),
    )

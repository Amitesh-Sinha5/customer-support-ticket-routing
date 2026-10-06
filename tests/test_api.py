import pytest
from fastapi.testclient import TestClient

from ticket_router.api import create_app


@pytest.fixture
def client(processor):
    with TestClient(create_app(processor)) as c:
        yield c


def submit(client, subject="Cannot log in", description="My password reset email never arrives.",
           priority="high"):
    return client.post("/tickets", json={"subject": subject, "description": description, "priority": priority})


def test_create_and_fetch_ticket(client):
    response = submit(client)
    assert response.status_code == 201
    ticket = response.json()
    assert ticket["analysis"]["routing"]["queue"] == "Account Management"
    assert client.get(f"/tickets/{ticket['id']}").json()["id"] == ticket["id"]


def test_validation_errors(client):
    assert client.post("/tickets", json={"description": "no subject"}).status_code == 422
    assert submit(client, priority="whenever").status_code == 422
    assert submit(client, subject="").status_code == 422


def test_filter_by_queue_and_queue_summary(client):
    submit(client)
    submit(client, subject="Package lost", description="My parcel never arrived and tracking is stuck.")
    shipping = client.get("/tickets", params={"queue": "Shipping & Delivery"}).json()
    assert len(shipping) == 1
    queues = client.get("/queues").json()
    assert queues["Shipping & Delivery"]["open_tickets"] == 1
    assert queues["Account Management"]["open_tickets"] == 1
    assert client.get("/tickets", params={"queue": "Nope"}).status_code == 400


def test_resolve_releases_agent(client):
    ticket = submit(client).json()
    agent_id = ticket["analysis"]["routing"]["agent_id"]
    load = lambda: next(a for a in client.get("/agents").json() if a["id"] == agent_id)["current_load"]

    assert load() == 1
    resolved = client.post(f"/tickets/{ticket['id']}/resolve").json()
    assert resolved["status"] == "resolved"
    assert load() == 0
    client.post(f"/tickets/{ticket['id']}/resolve")  # resolving twice must not go negative
    assert load() == 0


def test_root_redirects_to_docs(client):
    response = client.get("/", follow_redirects=False)
    assert response.status_code == 307
    assert response.headers["location"] == "/docs"


def test_unknown_ticket_returns_404(client):
    assert client.get("/tickets/TCK-NOPE").status_code == 404
    assert client.post("/tickets/TCK-NOPE/resolve").status_code == 404

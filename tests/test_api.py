from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from api.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:  # the with-block runs startup, which loads the data
        yield c


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["nurses"] == 200


def test_sample_request_returns_ranked_matches_with_explanations(client):
    body = client.get("/requests/R001/matches").json()
    assert len(body["matches"]) <= 5
    assert [m["rank"] for m in body["matches"]] == list(range(1, len(body["matches"]) + 1))
    scores = [m["score"] for m in body["matches"]]
    assert scores == sorted(scores, reverse=True)
    assert all(m["explanation"] for m in body["matches"])


def test_post_match_with_a_custom_request(client):
    payload = {
        "required_skill": "elderly_care",
        "language": "Hindi",
        "city": "Delhi",
        "lat": 28.61,
        "lon": 77.21,
        "slots": ["Mon-day", "Tue-day", "Wed-day"],
    }
    response = client.post("/match?top_k=3", json=payload)
    assert response.status_code == 200
    assert len(response.json()["matches"]) <= 3


def test_unknown_sample_request_is_a_404(client):
    assert client.get("/requests/R999/matches").status_code == 404


@pytest.mark.parametrize(
    "change",
    [
        {"required_skill": "juggling"},
        {"language": "Klingon"},
        {"slots": ["Funday-day"]},
        {"slots": []},
        {"lat": 123.0},
    ],
)
def test_bad_input_is_rejected_with_422(client, change):
    payload = {
        "required_skill": "elderly_care",
        "language": "Hindi",
        "lat": 28.61,
        "lon": 77.21,
        "slots": ["Mon-day"],
    } | change
    assert client.post("/match", json=payload).status_code == 422


def test_top_k_limits_are_enforced(client):
    assert client.get("/requests/R001/matches?top_k=0").status_code == 422
    assert client.get("/requests/R001/matches?top_k=50").status_code == 422

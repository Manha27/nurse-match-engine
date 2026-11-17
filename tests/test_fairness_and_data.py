"""Dataset sanity checks and the fairness evaluation on the seeded dataset."""

from __future__ import annotations

import pytest

from matcher.config import ALL_SLOTS, CITIES, LANGUAGES, SKILLS
from matcher.data_gen import generate_nurses, generate_requests
from matcher.fairness import evaluate, gini


@pytest.fixture(scope="module")
def dataset():
    nurses = generate_nurses(200, seed=42)
    requests = generate_requests(nurses, 100, seed=43)
    return nurses, requests


# ---- data generator ---------------------------------------------------------


def test_generator_produces_the_requested_counts(dataset):
    nurses, requests = dataset
    assert len(nurses) == 200
    assert len(requests) == 100
    assert len({n.nurse_id for n in nurses}) == 200


def test_generator_is_repeatable():
    a = generate_nurses(50, seed=7)
    b = generate_nurses(50, seed=7)
    assert a == b


def test_generated_values_are_valid(dataset):
    nurses, requests = dataset
    for n in nurses:
        assert n.skills and n.skills <= set(SKILLS)
        assert n.languages and n.languages <= set(LANGUAGES)
        assert n.city in CITIES
        assert n.available_slots and n.available_slots <= set(ALL_SLOTS)
        assert 0 <= n.years_experience <= 25
        assert (n.review_count == 0 and n.rating == 0.0) or 3.0 <= n.rating <= 5.0
    nurse_ids = {n.nurse_id for n in nurses}
    for r in requests:
        assert r.required_skill in SKILLS
        assert r.language in LANGUAGES
        assert r.city in CITIES
        assert r.slots <= set(ALL_SLOTS)
        assert set(r.previous_nurse_ids) <= nurse_ids


# ---- fairness ---------------------------------------------------------------


def test_gini_known_values():
    assert gini([1, 1, 1, 1]) == pytest.approx(0.0)
    assert gini([0, 0, 0, 4]) == pytest.approx(0.75)
    assert gini([0, 0, 0]) == 0.0


def test_no_nurse_is_always_picked_first(dataset):
    nurses, requests = dataset
    report = evaluate(nurses, requests)
    assert report.always_first == []


def test_no_single_nurse_takes_a_large_share_of_first_picks(dataset):
    nurses, requests = dataset
    report = evaluate(nurses, requests)
    assert report.max_top1_share <= 0.10


def test_workload_balancing_spreads_work_without_hurting_quality_much(dataset):
    nurses, requests = dataset
    plain = evaluate(nurses, requests)
    balanced = evaluate(nurses, requests, balance_weight=0.10)
    assert balanced.max_top1_count <= plain.max_top1_count
    assert balanced.top10_share < plain.top10_share
    assert balanced.gini_top1 < plain.gini_top1
    assert balanced.mean_top1_score >= plain.mean_top1_score - 0.02

"""Learned weights, on simulated feedback.

The feedback comes from invented hidden preferences, so these tests check that
the method recovers what it should, not how it would do on real data.
"""

from __future__ import annotations

import pytest

from matcher.data_gen import generate_nurses, generate_requests
from matcher.learning import (
    DEFAULT_WEIGHTS,
    FACTORS,
    HIDDEN_WEIGHTS,
    PastMatch,
    expected_satisfaction,
    fit_weights,
    satisfaction_probability,
    simulate_history,
    weight_error,
)


@pytest.fixture(scope="module")
def nurses():
    return generate_nurses(200, seed=42)


@pytest.fixture(scope="module")
def held_out(nurses):
    return generate_requests(nurses, 100, seed=43)


@pytest.fixture(scope="module")
def history(nurses):
    return simulate_history(nurses, 2000, seed=1000)


def test_history_is_repeatable(nurses):
    a = simulate_history(nurses, 80, seed=5)
    b = simulate_history(nurses, 80, seed=5)
    assert a == b


def test_history_contains_lower_ranked_picks_and_both_outcomes(history):
    # Without lower-ranked picks the model could never see what separates a good nurse from a so-so one.
    assert {p.shown_rank for p in history} >= {1, 2, 3}
    assert {p.went_well for p in history} == {0, 1}


def test_better_matches_go_well_more_often():
    great = {"distance": 1.0, "rating": 1.0, "experience": 1.0, "continuity": 1.0, "workload": 1.0}
    poor = {"distance": 0.0, "rating": 0.0, "experience": 0.0, "continuity": 0.0, "workload": 1.0}
    assert satisfaction_probability(great) > 0.9
    assert satisfaction_probability(poor) < 0.05


def test_weight_error_known_values():
    assert weight_error(DEFAULT_WEIGHTS, DEFAULT_WEIGHTS) == pytest.approx(0.0)
    # (0.30, 0.25, 0.15, 0.30) against (0.10, 0.40, 0.20, 0.30): 0.20 + 0.15 + 0.05 + 0 = 0.40
    assert weight_error(DEFAULT_WEIGHTS, HIDDEN_WEIGHTS) == pytest.approx(0.40)


def test_fitted_weights_are_valid_even_from_a_small_history(history):
    for size in (50, 300, len(history)):
        w = fit_weights(history[:size])
        values = [w.as_dict()[f] for f in FACTORS]
        assert all(v >= 0 for v in values)
        assert sum(values) == pytest.approx(1.0)
        assert w.workload == 0.0


def test_fit_recovers_the_hidden_weights_from_enough_matches(history):
    learned = fit_weights(history)
    assert weight_error(learned, HIDDEN_WEIGHTS) < 0.25
    # Closer to the truth than the hand-set weights were.
    assert weight_error(learned, HIDDEN_WEIGHTS) < weight_error(DEFAULT_WEIGHTS, HIDDEN_WEIGHTS)


def test_learned_weights_rank_better_than_hand_set_on_unseen_requests(nurses, held_out, history):
    learned = fit_weights(history)
    default_score = expected_satisfaction(nurses, held_out, DEFAULT_WEIGHTS)
    learned_score = expected_satisfaction(nurses, held_out, learned)
    best_possible = expected_satisfaction(nurses, held_out, HIDDEN_WEIGHTS)
    assert learned_score > default_score
    # Ranking by the hidden weights is the best any weights can do, so nothing can beat it.
    assert learned_score <= best_possible + 1e-9


def test_fit_refuses_feedback_with_only_one_kind_of_outcome():
    factors = {f: 0.5 for f in FACTORS}
    all_bad = [PastMatch(1, factors, 0) for _ in range(20)]
    with pytest.raises(ValueError, match="good and bad outcomes"):
        fit_weights(all_bad)

"""Hand-checked cases.

Each case builds a handful of nurses, and the expected outcome was worked out on
paper first. Where a number matters, the arithmetic is written in the test.

Default weights: distance 0.30, rating 0.25, experience 0.15, continuity 0.30.
Distance factor = 1 - km/25, never below 0. Distance is only scored unless a
service radius is passed in. Rating is shrunk towards 4.2 with a prior worth 10
reviews, then mapped from 3.0-5.0 onto 0-1. Experience = min(years, 10)/10.
"""

from __future__ import annotations

import pytest

from matcher import Weights, eligible_nurses, match
from matcher.rules import check_hard_rules

from .helpers import WEEKDAYS_DAY, WEEKDAYS_NIGHT, make_nurse, make_request


def ids(result):
    return [m.nurse.nurse_id for m in result.matches]


# ---- hard rules -------------------------------------------------------------


def test_nurse_without_the_skill_is_never_recommended():
    perfect_but_wrong_skill = make_nurse("N1", skills=("baby_care",), rating=5.0, years=15)
    plain_but_right_skill = make_nurse("N2", km=10, rating=4.0, years=1)
    result = match(make_request(skill="elderly_care"), [perfect_but_wrong_skill, plain_but_right_skill])
    assert ids(result) == ["N2"]


def test_nurse_who_does_not_speak_the_language_is_never_recommended():
    nurses = [make_nurse("N1", languages=("English",)), make_nurse("N2", languages=("Hindi", "Marathi"))]
    result = match(make_request(language="Marathi"), nurses)
    assert ids(result) == ["N2"]


def test_nurse_missing_even_one_slot_is_excluded():
    # Request is Mon-Fri days. N1 is free Mon-Thu only.
    mon_to_thu = frozenset(f"{d}-day" for d in ("Mon", "Tue", "Wed", "Thu"))
    nurse = make_nurse("N1", slots=mon_to_thu)
    request = make_request()
    failures = check_hard_rules(nurse, request, distance_km=0, max_distance_km=25)
    assert [f.rule for f in failures] == ["availability"]
    assert "Fri-day" in failures[0].message
    assert eligible_nurses(request, [nurse]) == []


def test_day_only_nurse_cannot_take_a_night_request():
    day_nurse = make_nurse("N1", slots=WEEKDAYS_DAY)
    night_nurse = make_nurse("N2", slots=WEEKDAYS_NIGHT)
    result = match(make_request(slots=WEEKDAYS_NIGHT), [day_nurse, night_nurse])
    assert ids(result) == ["N2"]


def test_distance_alone_never_excludes_a_nurse():
    # Distance is a scored factor. Without a service radius, a far nurse is still eligible.
    near, far = make_nurse("N1", km=24), make_nurse("N2", km=26)
    assert ids(match(make_request(), [near, far])) == ["N1", "N2"]


def test_service_radius_removes_nurses_beyond_it():
    near, far = make_nurse("N1", km=24), make_nurse("N2", km=26)
    assert ids(match(make_request(), [near, far], max_distance_km=25)) == ["N1"]
    assert ids(match(make_request(), [near, far], max_distance_km=40)) == ["N1", "N2"]


def test_nurse_in_another_city_ranks_last_by_default_and_is_dropped_by_a_radius():
    local = make_nurse("N1", km=3)
    mumbai = make_nurse("N2", city="Mumbai", lat=19.076, lon=72.8777)
    result = match(make_request(), [mumbai, local])
    assert ids(result) == ["N1", "N2"]
    assert result.matches[1].factors["distance"] == 0.0
    assert ids(match(make_request(), [mumbai, local], max_distance_km=25)) == ["N1"]


def test_a_far_nurse_is_flagged_in_the_explanation():
    result = match(make_request(), [make_nurse("N1", km=60)])
    assert "60.0 km away" in result.matches[0].explanation


def test_funnel_has_a_distance_step_only_when_a_radius_is_set():
    nurses = [make_nurse("N1", km=3), make_nurse("N2", km=40)]
    without = match(make_request(), nurses).funnel
    with_radius = match(make_request(), nurses, max_distance_km=25).funnel
    assert [step for step, _ in without][-1] == "Free for every requested slot"
    assert with_radius[-1] == ("Within 25 km", 1)


@pytest.mark.parametrize("bad_radius", [0, -5])
def test_a_non_positive_radius_is_rejected(bad_radius):
    with pytest.raises(ValueError, match="greater than zero"):
        match(make_request(), [make_nurse("N1")], max_distance_km=bad_radius)


def test_weights_must_be_non_negative_and_not_all_zero():
    with pytest.raises(ValueError, match="cannot be negative"):
        Weights(distance=-0.1)
    with pytest.raises(ValueError, match="above zero"):
        Weights(distance=0, rating=0, experience=0, continuity=0)


def test_a_nurse_failing_several_rules_reports_all_of_them():
    nurse = make_nurse("N1", skills=("baby_care",), languages=("English",))
    failures = check_hard_rules(nurse, make_request(), distance_km=0, max_distance_km=25)
    assert {f.rule for f in failures} == {"skill", "language"}


# ---- soft factors -----------------------------------------------------------


def test_exact_score_for_one_nurse():
    # At the door: distance factor 1.0 -> 0.30 * 1.0                    = 0.30
    # Rating 4.5 from 90 reviews: (90*4.5 + 10*4.2)/100 = 4.47
    #   -> (4.47 - 3.0)/2 = 0.735                  -> 0.25 * 0.735      = 0.18375
    # 5 years: 0.5                                  -> 0.15 * 0.5        = 0.075
    # No previous care: 0                           -> 0.30 * 0          = 0
    nurse = make_nurse("N1", km=0, years=5, rating=4.5, reviews=90)
    top = match(make_request(), [nurse]).matches[0]
    assert top.score == pytest.approx(0.55875, abs=1e-6)
    assert top.distance_km == pytest.approx(0.0, abs=1e-6)


def test_contributions_add_up_to_the_score():
    nurses = [make_nurse("N1", km=3), make_nurse("N2", km=9, rating=4.9, years=12)]
    for m in match(make_request(), nurses).matches:
        assert sum(m.contributions.values()) == pytest.approx(m.score)


def test_closer_nurse_wins_when_everything_else_is_equal():
    result = match(make_request(), [make_nurse("N1", km=12), make_nurse("N2", km=4)])
    assert ids(result) == ["N2", "N1"]


def test_higher_rating_wins_when_everything_else_is_equal():
    result = match(make_request(), [make_nurse("N1", rating=4.2), make_nurse("N2", rating=4.8)])
    assert ids(result) == ["N2", "N1"]


def test_a_perfect_rating_from_two_reviews_does_not_beat_a_solid_record():
    # N1: 5.0 from 2 reviews -> (2*5.0 + 42)/12 = 4.333 -> factor 0.667
    # N2: 4.7 from 80 reviews -> (80*4.7 + 42)/90 = 4.644 -> factor 0.822
    thin = make_nurse("N1", rating=5.0, reviews=2)
    solid = make_nurse("N2", rating=4.7, reviews=80)
    assert ids(match(make_request(), [thin, solid])) == ["N2", "N1"]


def test_nurse_with_no_reviews_is_scored_on_the_typical_rating():
    # The stored rating is ignored when there are no reviews: factor (4.2 - 3.0)/2 = 0.6
    new_nurse = make_nurse("N1", rating=0.0, reviews=0)
    top = match(make_request(), [new_nurse]).matches[0]
    assert top.factors["rating"] == pytest.approx(0.6)


def test_experience_stops_counting_after_ten_years():
    result = match(make_request(), [make_nurse("N1", years=10), make_nurse("N2", years=20)])
    assert result.matches[0].score == pytest.approx(result.matches[1].score)


# ---- continuity -------------------------------------------------------------


def test_previous_nurse_beats_a_slightly_better_nurse():
    # N1 (previous): 5 km -> 0.8*0.30 = 0.240; 4.0 from 90 -> 4.02 -> 0.51*0.25 = 0.1275;
    #                2 years -> 0.2*0.15 = 0.030; continuity 0.30        => 0.6975
    # N2:            1 km -> 0.96*0.30 = 0.288; 4.9 from 90 -> 4.83 -> 0.915*0.25 = 0.22875;
    #                10 years -> 0.15                                    => 0.66675
    previous = make_nurse("N1", km=5, rating=4.0, years=2)
    better = make_nurse("N2", km=1, rating=4.9, years=10)

    with_history = match(make_request(previous=["N1"]), [previous, better])
    assert ids(with_history) == ["N1", "N2"]
    assert with_history.matches[0].score == pytest.approx(0.6975, abs=1e-4)
    assert with_history.matches[1].score == pytest.approx(0.66675, abs=1e-4)

    # Same people, no history: the better nurse wins, so continuity is what made the difference.
    without_history = match(make_request(), [previous, better])
    assert ids(without_history) == ["N2", "N1"]


def test_previous_nurse_who_breaks_a_rule_is_not_recommended_but_is_mentioned():
    previous = make_nurse("N1", languages=("English",), slots=WEEKDAYS_NIGHT)
    other = make_nurse("N2", km=5)
    result = match(make_request(language="Hindi", previous=["N1"]), [previous, other])
    assert ids(result) == ["N2"]
    assert any("Nurse N1" in n and "cared for this patient before" in n for n in result.notes)
    assert any("does not speak Hindi" in n for n in result.notes)


# ---- edge cases -------------------------------------------------------------


def test_fewer_than_five_eligible_returns_what_exists_and_says_so():
    nurses = [make_nurse(f"N{i}", km=i) for i in range(1, 4)]
    result = match(make_request(), nurses)
    assert len(result.matches) == 3
    assert result.eligible_count == 3
    assert any("Only 3" in n for n in result.notes)


def test_a_single_eligible_nurse_gets_a_grammatical_note():
    result = match(make_request(), [make_nurse("N1")])
    assert result.notes == ["Only 1 nurse meets every requirement."]


def test_no_eligible_nurse_gives_an_empty_list_and_a_funnel_that_shows_where_they_went():
    nurses = [make_nurse("N1", skills=("baby_care",)), make_nurse("N2", skills=("physio_support",))]
    result = match(make_request(skill="elderly_care"), nurses)
    assert result.matches == []
    assert result.notes
    assert result.funnel[0] == ("All nurses", 2)
    assert result.funnel[1][1] == 0  # nobody offers the skill


def test_more_than_five_eligible_returns_exactly_five():
    nurses = [make_nurse(f"N{i}", km=i) for i in range(1, 9)]
    assert len(match(make_request(), nurses).matches) == 5


def test_identical_nurses_are_ordered_by_id_so_results_are_repeatable():
    result = match(make_request(), [make_nurse("N2"), make_nurse("N1")])
    assert ids(result) == ["N1", "N2"]


# ---- explanations -----------------------------------------------------------


def test_explanation_names_continuity_and_the_gap_to_the_nurse_above():
    previous = make_nurse("N1", km=5, rating=4.0, years=2)
    better = make_nurse("N2", km=1, rating=4.9, years=10)
    result = match(make_request(previous=["N1"]), [previous, better])
    first, second = result.matches
    assert "cared for this patient before" in first.explanation
    # N2 loses on continuity (0.30) by more than it gains anywhere else.
    assert "Ranks below #1 mainly on continuity" in second.explanation


# ---- workload balancing -----------------------------------------------------


def test_workload_term_moves_a_busy_nurse_down_but_only_when_switched_on():
    nurses = [make_nurse("N1"), make_nurse("N2")]
    busy = {"N1": 3}
    assert ids(match(make_request(), nurses, load=busy)) == ["N1", "N2"]  # off: tie, id order
    balanced = Weights(workload=0.10)
    assert ids(match(make_request(), nurses, weights=balanced, load=busy)) == ["N2", "N1"]

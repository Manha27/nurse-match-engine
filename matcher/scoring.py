"""Soft factors. Each one maps to a number between 0 and 1, then the weights are applied."""

from __future__ import annotations

from dataclasses import dataclass

from .config import (
    EXPERIENCE_CAP_YEARS,
    PRIOR_RATING,
    PRIOR_WEIGHT,
    RATING_CEIL,
    RATING_FLOOR,
    WORKLOAD_CAP,
)
from .models import CareRequest, Nurse


@dataclass(frozen=True)
class Weights:
    distance: float = 0.30
    rating: float = 0.25
    experience: float = 0.15
    continuity: float = 0.30
    # Off by default. The fairness evaluation turns it on to spread work around.
    workload: float = 0.0

    def as_dict(self) -> dict[str, float]:
        return {
            "distance": self.distance,
            "rating": self.rating,
            "experience": self.experience,
            "continuity": self.continuity,
            "workload": self.workload,
        }

    @property
    def total(self) -> float:
        return sum(self.as_dict().values())


DEFAULT_WEIGHTS = Weights()


def distance_factor(distance_km: float, max_distance_km: float) -> float:
    """1.0 at the patient's door, falling in a straight line to 0.0 at the limit."""
    return max(0.0, 1.0 - distance_km / max_distance_km)


def shrunk_rating(rating: float, review_count: int) -> float:
    """Pull a rating towards a typical value when there are few reviews behind it."""
    return (review_count * rating + PRIOR_WEIGHT * PRIOR_RATING) / (review_count + PRIOR_WEIGHT)


def rating_factor(rating: float, review_count: int) -> float:
    r = shrunk_rating(rating, review_count)
    return min(1.0, max(0.0, (r - RATING_FLOOR) / (RATING_CEIL - RATING_FLOOR)))


def experience_factor(years: int) -> float:
    return min(years, EXPERIENCE_CAP_YEARS) / EXPERIENCE_CAP_YEARS


def continuity_factor(nurse: Nurse, request: CareRequest) -> float:
    return 1.0 if nurse.nurse_id in request.previous_nurse_ids else 0.0


def workload_factor(current_load: int) -> float:
    """1.0 for a nurse with nothing assigned, 0.0 at WORKLOAD_CAP or more."""
    return 1.0 - min(current_load, WORKLOAD_CAP) / WORKLOAD_CAP


def score_factors(
    nurse: Nurse,
    request: CareRequest,
    distance_km: float,
    max_distance_km: float,
    current_load: int = 0,
) -> dict[str, float]:
    return {
        "distance": distance_factor(distance_km, max_distance_km),
        "rating": rating_factor(nurse.rating, nurse.review_count),
        "experience": experience_factor(nurse.years_experience),
        "continuity": continuity_factor(nurse, request),
        "workload": workload_factor(current_load),
    }


def weighted_contributions(factors: dict[str, float], weights: Weights) -> dict[str, float]:
    """Each factor's share of the final 0-1 score. They add up to the score."""
    total = weights.total
    w = weights.as_dict()
    return {name: w[name] * factors[name] / total for name in w}

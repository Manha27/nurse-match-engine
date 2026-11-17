"""Turn a score breakdown into a sentence a coordinator can read out on a call.

No language model is involved. The text is built from the same numbers that
produced the score, so the explanation cannot disagree with the ranking.
"""

from __future__ import annotations

from .models import Nurse
from .scoring import Weights

STRONG = 0.7  # a factor at or above this is worth mentioning as a strength
WEAK = 0.3  # a factor at or below this is mentioned as a drawback

FACTOR_LABEL = {
    "distance": "distance",
    "rating": "rating",
    "experience": "experience",
    "continuity": "continuity with the patient",
    "workload": "current workload",
}


def _strength(name: str, nurse: Nurse, distance_km: float, load: int) -> str:
    if name == "distance":
        return f"lives {distance_km:.1f} km away"
    if name == "rating":
        return f"rated {nurse.rating:.1f} from {nurse.review_count} reviews"
    if name == "experience":
        return f"has {nurse.years_experience} years of experience"
    if name == "continuity":
        return "has cared for this patient before"
    return "has a light workload right now"


def _weakness(name: str, nurse: Nurse, distance_km: float, load: int) -> str:
    if name == "distance":
        return f"is {distance_km:.1f} km away"
    if name == "rating":
        if nurse.review_count < 10:
            return f"has only {nurse.review_count} reviews so far"
        return f"has a modest rating ({nurse.rating:.1f})"
    if name == "experience":
        years = nurse.years_experience
        return f"has {years} year{'s' if years != 1 else ''} of experience"
    return f"already has {load} recent assignments"


def explain(
    rank: int,
    nurse: Nurse,
    distance_km: float,
    factors: dict[str, float],
    contributions: dict[str, float],
    weights: Weights,
    load: int = 0,
    above_contributions: dict[str, float] | None = None,
) -> str:
    active = [n for n, w in weights.as_dict().items() if w > 0]

    strengths = sorted(
        (n for n in active if factors[n] >= STRONG), key=lambda n: contributions[n], reverse=True
    )
    # Not having cared for a patient before is the normal case, not a drawback.
    weaknesses = [n for n in active if factors[n] <= WEAK and n != "continuity"]

    parts = []
    if strengths:
        phrases = [_strength(n, nurse, distance_km, load) for n in strengths]
        sentence = ", ".join(phrases[:-1]) + (" and " if len(phrases) > 1 else "") + phrases[-1]
        parts.append(sentence[0].upper() + sentence[1:] + ".")
    else:
        parts.append("Meets every required rule, with no standout strength.")

    if weaknesses:
        phrases = [_weakness(n, nurse, distance_km, load) for n in weaknesses]
        parts.append("Held back because the nurse " + " and ".join(phrases) + ".")

    if rank > 1 and above_contributions is not None:
        gaps = {n: above_contributions[n] - contributions[n] for n in active}
        biggest = max(gaps, key=gaps.get)
        if gaps[biggest] > 0.005:
            parts.append(
                f"Ranks below #{rank - 1} mainly on {FACTOR_LABEL[biggest]} "
                f"({-gaps[biggest]:+.2f} of score)."
            )
        else:
            parts.append(f"Very close to #{rank - 1}.")

    return " ".join(parts)

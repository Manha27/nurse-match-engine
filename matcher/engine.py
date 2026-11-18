"""The matching pipeline: filter with hard rules, score the survivors, rank, explain."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

from .config import label
from .explain import explain
from .geo import haversine_km
from .models import CareRequest, Nurse
from .rules import RULE_ORDER, check_hard_rules
from .scoring import DEFAULT_WEIGHTS, Weights, score_factors, weighted_contributions


@dataclass
class Match:
    rank: int
    nurse: Nurse
    score: float
    distance_km: float
    factors: dict[str, float]
    contributions: dict[str, float]
    explanation: str


@dataclass
class MatchResult:
    request: CareRequest
    matches: list[Match]
    eligible_count: int
    # (step, nurses remaining) after each hard rule, in RULE_ORDER.
    funnel: list[tuple[str, int]]
    notes: list[str] = field(default_factory=list)


def eligible_nurses(
    request: CareRequest, nurses: Sequence[Nurse], max_distance_km: float | None = None
) -> list[tuple[Nurse, float]]:
    """Nurses who pass every hard rule, each with their distance to the patient."""
    out = []
    for nurse in nurses:
        d = haversine_km(request.lat, request.lon, nurse.lat, nurse.lon)
        if not check_hard_rules(nurse, request, d, max_distance_km):
            out.append((nurse, d))
    return out


def match(
    request: CareRequest,
    nurses: Sequence[Nurse],
    weights: Weights = DEFAULT_WEIGHTS,
    top_k: int = 5,
    max_distance_km: float | None = None,
    load: Mapping[str, int] | None = None,
) -> MatchResult:
    if max_distance_km is not None and max_distance_km <= 0:
        raise ValueError("max_distance_km must be greater than zero (or None for no limit).")
    load = load or {}
    by_id = {n.nurse_id: n for n in nurses}

    failed_rules: dict[str, set[str]] = {}
    failure_text: dict[str, str] = {}
    distances: dict[str, float] = {}
    for nurse in nurses:
        d = haversine_km(request.lat, request.lon, nurse.lat, nurse.lon)
        distances[nurse.nurse_id] = d
        failures = check_hard_rules(nurse, request, d, max_distance_km)
        failed_rules[nurse.nurse_id] = {f.rule for f in failures}
        failure_text[nurse.nurse_id] = " and ".join(f.message for f in failures)

    funnel = [("All nurses", len(nurses))]
    steps = {
        "skill": f"Offer {label(request.required_skill)}",
        "language": f"Speak {request.language}",
        "availability": "Free for every requested slot",
    }
    if max_distance_km is not None:
        steps["distance"] = f"Within {max_distance_km:.0f} km"
    rules_in_use = [rule for rule in RULE_ORDER if rule in steps]
    for i, rule in enumerate(rules_in_use):
        checked = set(rules_in_use[: i + 1])
        remaining = sum(1 for n in nurses if not (failed_rules[n.nurse_id] & checked))
        funnel.append((steps[rule], remaining))

    candidates = []
    for nurse in nurses:
        if failed_rules[nurse.nurse_id]:
            continue
        d = distances[nurse.nurse_id]
        factors = score_factors(nurse, request, d, load.get(nurse.nurse_id, 0))
        contributions = weighted_contributions(factors, weights)
        candidates.append((sum(contributions.values()), d, nurse, factors, contributions))

    # Highest score first. Exact ties go to the closer nurse, then to the lower id,
    # so results are repeatable.
    candidates.sort(key=lambda c: (-c[0], c[1], c[2].nurse_id))

    matches: list[Match] = []
    for i, (score, d, nurse, factors, contributions) in enumerate(candidates[:top_k]):
        above = matches[-1].contributions if matches else None
        matches.append(
            Match(
                rank=i + 1,
                nurse=nurse,
                score=score,
                distance_km=d,
                factors=factors,
                contributions=contributions,
                explanation=explain(
                    i + 1, nurse, d, factors, contributions, weights,
                    load=load.get(nurse.nurse_id, 0), above_contributions=above,
                ),
            )
        )

    notes = []
    if not candidates:
        notes.append("No nurse meets every requirement. See the funnel to find which rule removed the most.")
    elif len(candidates) < top_k:
        count = len(candidates)
        notes.append(
            f"Only {count} nurse meets every requirement." if count == 1
            else f"Only {count} nurses meet every requirement."
        )
    for prev_id in request.previous_nurse_ids:
        prev = by_id.get(prev_id)
        if prev is not None and failed_rules[prev_id]:
            notes.append(
                f"{prev.name} ({prev_id}) has cared for this patient before but "
                f"{failure_text[prev_id]}."
            )

    return MatchResult(
        request=request,
        matches=matches,
        eligible_count=len(candidates),
        funnel=funnel,
        notes=notes,
    )

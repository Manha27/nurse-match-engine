"""Hard rules. A nurse who breaks any of these is never recommended."""

from __future__ import annotations

from typing import NamedTuple

from .config import label, slot_sort_key
from .models import CareRequest, Nurse

# Order used for the "how many nurses are left" funnel shown in the demo.
RULE_ORDER = ("skill", "language", "availability", "distance")


class RuleFailure(NamedTuple):
    rule: str
    message: str


def check_hard_rules(
    nurse: Nurse, request: CareRequest, distance_km: float, max_distance_km: float
) -> list[RuleFailure]:
    """Return every rule the nurse fails. An empty list means eligible."""
    failures = []

    if request.required_skill not in nurse.skills:
        failures.append(RuleFailure("skill", f"does not offer {label(request.required_skill)}"))

    if request.language not in nurse.languages:
        failures.append(RuleFailure("language", f"does not speak {request.language}"))

    missing = request.slots - nurse.available_slots
    if missing:
        slots = ", ".join(sorted(missing, key=slot_sort_key))
        failures.append(RuleFailure("availability", f"is not free for {slots}"))

    if distance_km > max_distance_km:
        failures.append(
            RuleFailure("distance", f"lives {distance_km:.0f} km away (limit {max_distance_km:.0f} km)")
        )

    return failures

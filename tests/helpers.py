"""Small builders so each test case reads as a short story."""

from __future__ import annotations

from matcher.config import DAYS
from matcher.models import CareRequest, Nurse

WEEKDAYS_DAY = frozenset(f"{d}-day" for d in DAYS[:5])
WEEKDAYS_NIGHT = frozenset(f"{d}-night" for d in DAYS[:5])

PUNE_LAT, PUNE_LON = 18.5204, 73.8567
KM_PER_DEGREE_LAT = 111.195  # great-circle length of one degree of latitude


def north_of_patient(km: float) -> tuple[float, float]:
    """A point exactly `km` due north of the default patient (distance is then known by hand)."""
    return PUNE_LAT + km / KM_PER_DEGREE_LAT, PUNE_LON


def make_nurse(
    nurse_id: str = "N1",
    km: float = 0.0,
    skills=("elderly_care",),
    languages=("Hindi",),
    years: int = 5,
    rating: float = 4.5,
    reviews: int = 90,
    slots=WEEKDAYS_DAY,
    city: str = "Pune",
    lat: float | None = None,
    lon: float | None = None,
) -> Nurse:
    default_lat, default_lon = north_of_patient(km)
    return Nurse(
        nurse_id=nurse_id,
        name=f"Nurse {nurse_id}",
        skills=frozenset(skills),
        languages=frozenset(languages),
        city=city,
        lat=default_lat if lat is None else lat,
        lon=default_lon if lon is None else lon,
        years_experience=years,
        rating=rating,
        review_count=reviews,
        available_slots=frozenset(slots),
    )


def make_request(
    skill: str = "elderly_care",
    language: str = "Hindi",
    slots=WEEKDAYS_DAY,
    previous=(),
) -> CareRequest:
    return CareRequest(
        request_id="R-test",
        patient_id="P-test",
        patient_name="Test Patient",
        required_skill=skill,
        language=language,
        city="Pune",
        lat=PUNE_LAT,
        lon=PUNE_LON,
        slots=frozenset(slots),
        previous_nurse_ids=tuple(previous),
    )

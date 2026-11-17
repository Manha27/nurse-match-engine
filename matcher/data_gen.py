"""Synthetic data: 200 nurses and 100 care requests.

Everything here is invented. Names are random combinations of common Indian
first and last names, locations are scattered around three city centres, and no
real person or record is involved.

    python -m matcher.data_gen --seed 42
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .config import CITIES, DAYS, SHIFTS, SKILLS
from .dataset import save_dataset
from .models import CareRequest, Nurse

FIRST_NAMES = [
    "Anita", "Sunita", "Meera", "Kavita", "Priya", "Neha", "Pooja", "Rekha", "Shweta", "Asha",
    "Lata", "Jyoti", "Savita", "Deepa", "Manisha", "Sneha", "Radha", "Geeta", "Usha", "Nisha",
    "Farida", "Salma", "Harpreet", "Jaspreet", "Mary", "Rosy", "Aarti", "Komal", "Vaishali", "Swati",
    "Rahul", "Amit", "Sanjay", "Vijay", "Rohit", "Imran", "Arjun", "Mohan", "Ravi", "Deepak",
]
LAST_NAMES = [
    "Patil", "Shinde", "Jadhav", "Kulkarni", "Deshmukh", "More", "Pawar", "Gaikwad", "Sharma", "Verma",
    "Gupta", "Singh", "Kaur", "Yadav", "Mishra", "Khan", "Ansari", "D'Souza", "Fernandes", "Pereira",
    "Shah", "Mehta", "Joshi", "Nair", "Iyer", "Chauhan", "Thakur", "Bhosale", "Naik", "Rao",
]

# Share of nurses living in each city, and of requests coming from each city.
CITY_SHARE = {"Pune": 0.30, "Mumbai": 0.35, "Delhi": 0.35}

# Probability that a nurse in a city speaks each language (independent draws).
NURSE_LANGUAGES = {
    "Pune": {"Marathi": 0.85, "Hindi": 0.80, "English": 0.60, "Gujarati": 0.05, "Punjabi": 0.02, "Urdu": 0.05},
    "Mumbai": {"Marathi": 0.60, "Hindi": 0.85, "English": 0.65, "Gujarati": 0.30, "Punjabi": 0.02, "Urdu": 0.10},
    "Delhi": {"Hindi": 0.95, "English": 0.65, "Punjabi": 0.35, "Urdu": 0.15, "Marathi": 0.01, "Gujarati": 0.02},
}
# The one language a family asks for, per city.
PATIENT_LANGUAGE = {
    "Pune": {"Marathi": 0.40, "Hindi": 0.30, "English": 0.27, "Gujarati": 0.03},
    "Mumbai": {"Hindi": 0.38, "Marathi": 0.25, "English": 0.22, "Gujarati": 0.12, "Urdu": 0.03},
    "Delhi": {"Hindi": 0.62, "English": 0.20, "Punjabi": 0.14, "Urdu": 0.04},
}

SKILL_WEIGHT = {
    "elderly_care": 0.35,
    "dementia_care": 0.15,
    "baby_care": 0.18,
    "post_surgery_care": 0.20,
    "physio_support": 0.12,
}
REQUEST_SKILL_WEIGHT = {
    "elderly_care": 0.35,
    "dementia_care": 0.20,
    "baby_care": 0.15,
    "post_surgery_care": 0.20,
    "physio_support": 0.10,
}

# Weekly patterns families ask for.
REQUEST_PATTERNS = {
    "weekdays": ([0, 1, 2, 3, 4], 0.40),
    "all_week": ([0, 1, 2, 3, 4, 5, 6], 0.20),
    "mon_wed_fri": ([0, 2, 4], 0.15),
    "tue_thu_sat": ([1, 3, 5], 0.15),
    "weekends": ([5, 6], 0.10),
}
REQUEST_SHIFT_PROB = {"day": 0.65, "night": 0.35}


def _jitter(rng: np.random.Generator, city: str) -> tuple[float, float]:
    """A point scattered around the city centre (roughly 6 km standard deviation)."""
    lat0, lon0 = CITIES[city]
    dlat = float(np.clip(rng.normal(0, 0.06), -0.15, 0.15))
    dlon = float(np.clip(rng.normal(0, 0.06), -0.15, 0.15))
    return round(lat0 + dlat, 5), round(lon0 + dlon, 5)


def _pick(rng: np.random.Generator, weights: dict[str, float]) -> str:
    keys = list(weights)
    p = np.array([weights[k] for k in keys], dtype=float)
    return str(rng.choice(keys, p=p / p.sum()))


def _nurse_skills(rng: np.random.Generator) -> frozenset[str]:
    primary = _pick(rng, SKILL_WEIGHT)
    skills = {primary}
    # Dementia care usually comes with general elderly care.
    if primary == "dementia_care" and rng.random() < 0.7:
        skills.add("elderly_care")
    for extra in rng.permutation(SKILLS)[: rng.integers(0, 3)]:
        skills.add(str(extra))
    return frozenset(skills)


def _nurse_slots(rng: np.random.Generator) -> frozenset[str]:
    roll = rng.random()
    if roll < 0.35:  # full-time, any day
        days = set(range(7))
    elif roll < 0.70:  # weekdays, sometimes a weekend day too
        days = set(range(5))
        for d in (5, 6):
            if rng.random() < 0.40:
                days.add(d)
    else:  # part-time, a random handful of days
        k = int(rng.integers(3, 7))
        days = set(int(x) for x in rng.choice(7, size=k, replace=False))

    shift_roll = rng.random()
    shifts = ["day"] if shift_roll < 0.25 else ["night"] if shift_roll < 0.40 else list(SHIFTS)
    return frozenset(f"{DAYS[d]}-{s}" for d in days for s in shifts)


def generate_nurses(n: int = 200, seed: int = 42) -> list[Nurse]:
    rng = np.random.default_rng(seed)
    nurses = []
    for i in range(n):
        city = _pick(rng, CITY_SHARE)
        lat, lon = _jitter(rng, city)

        langs = {l for l, p in NURSE_LANGUAGES[city].items() if rng.random() < p}
        if not langs:
            langs = {"Hindi"}

        years = int(np.clip(rng.gamma(2.2, 3.0), 0, 25))
        reviews = int(rng.poisson(5 + years * 5))
        if reviews == 0:
            rating = 0.0
        else:
            rating = float(np.clip(rng.normal(4.35 + 0.01 * years, 0.35), 3.0, 5.0))
            rating = round(rating, 1)

        nurses.append(
            Nurse(
                nurse_id=f"N{i + 1:03d}",
                name=f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)}",
                skills=_nurse_skills(rng),
                languages=frozenset(langs),
                city=city,
                lat=lat,
                lon=lon,
                years_experience=years,
                rating=rating,
                review_count=reviews,
                available_slots=_nurse_slots(rng),
            )
        )
    return nurses


def generate_requests(nurses: list[Nurse], n: int = 100, seed: int = 43) -> list[CareRequest]:
    rng = np.random.default_rng(seed)
    pattern_names = list(REQUEST_PATTERNS)
    pattern_p = np.array([REQUEST_PATTERNS[k][1] for k in pattern_names])

    requests = []
    for i in range(n):
        city = _pick(rng, CITY_SHARE)
        lat, lon = _jitter(rng, city)
        skill = _pick(rng, REQUEST_SKILL_WEIGHT)
        language = _pick(rng, PATIENT_LANGUAGE[city])

        pattern = str(rng.choice(pattern_names, p=pattern_p))
        shift = _pick(rng, REQUEST_SHIFT_PROB)
        slots = frozenset(f"{DAYS[d]}-{shift}" for d in REQUEST_PATTERNS[pattern][0])

        # About a third of patients have been looked after by someone before.
        previous: tuple[str, ...] = ()
        if rng.random() < 0.35:
            pool = [x for x in nurses if x.city == city and skill in x.skills]
            if pool:
                previous = (pool[int(rng.integers(len(pool)))].nurse_id,)

        requests.append(
            CareRequest(
                request_id=f"R{i + 1:03d}",
                patient_id=f"P{i + 1:03d}",
                patient_name=f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)}",
                required_skill=skill,
                language=language,
                city=city,
                lat=lat,
                lon=lon,
                slots=slots,
                previous_nurse_ids=previous,
            )
        )
    return requests


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic nurses and care requests.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--nurses", type=int, default=200)
    parser.add_argument("--requests", type=int, default=100)
    parser.add_argument("--out", type=Path, default=Path("data"))
    args = parser.parse_args()

    nurses = generate_nurses(args.nurses, seed=args.seed)
    requests = generate_requests(nurses, args.requests, seed=args.seed + 1)
    save_dataset(args.out, nurses, requests)
    print(f"Wrote {len(nurses)} nurses and {len(requests)} requests to {args.out}/")


if __name__ == "__main__":
    main()

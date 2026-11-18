"""Constants shared by the generator, the engine, the API and the demo."""

SKILLS = (
    "elderly_care",
    "dementia_care",
    "baby_care",
    "post_surgery_care",
    "physio_support",
)

LANGUAGES = ("English", "Hindi", "Marathi", "Gujarati", "Punjabi", "Urdu")

# City centres (lat, lon). Home locations are generated around these.
CITIES = {
    "Pune": (18.5204, 73.8567),
    "Mumbai": (19.0760, 72.8777),
    "Delhi": (28.6139, 77.2090),
}

DAYS = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")
SHIFTS = ("day", "night")

# A slot is one shift on one weekday, written "Mon-day" or "Sat-night".
ALL_SLOTS = tuple(f"{d}-{s}" for d in DAYS for s in SHIFTS)

# The distance score falls in a straight line from 1.0 at the patient's door to 0.0 at
# this distance, and stays at 0.0 beyond it.
DISTANCE_SCALE_KM = 25.0

# Optional service radius: nurses further away are left out entirely. It is off by
# default (distance is only scored). The demo's switch for it starts from this value.
SUGGESTED_RADIUS_KM = 25.0

# Rating handling: a rating is shrunk towards PRIOR_RATING until the nurse has
# collected enough reviews. PRIOR_WEIGHT is "how many reviews the prior is worth".
PRIOR_RATING = 4.2
PRIOR_WEIGHT = 10
RATING_FLOOR = 3.0
RATING_CEIL = 5.0

# Experience stops adding to the score after this many years.
EXPERIENCE_CAP_YEARS = 10

# Used only when the workload factor is switched on (see fairness.py).
WORKLOAD_CAP = 5


def slot_sort_key(slot: str) -> tuple[int, int]:
    day, shift = slot.split("-")
    return DAYS.index(day), SHIFTS.index(shift)


def label(name: str) -> str:
    """'dementia_care' -> 'dementia care'"""
    return name.replace("_", " ")

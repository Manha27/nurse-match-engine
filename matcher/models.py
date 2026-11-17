from __future__ import annotations

from dataclasses import dataclass, field

from .config import slot_sort_key


@dataclass(frozen=True)
class Nurse:
    nurse_id: str
    name: str
    skills: frozenset[str]
    languages: frozenset[str]
    city: str
    lat: float
    lon: float
    years_experience: int
    rating: float  # 0.0 when review_count is 0 (ignored in that case)
    review_count: int
    available_slots: frozenset[str]

    def to_dict(self) -> dict:
        return {
            "nurse_id": self.nurse_id,
            "name": self.name,
            "skills": sorted(self.skills),
            "languages": sorted(self.languages),
            "city": self.city,
            "lat": self.lat,
            "lon": self.lon,
            "years_experience": self.years_experience,
            "rating": self.rating,
            "review_count": self.review_count,
            "available_slots": sorted(self.available_slots, key=slot_sort_key),
        }

    @classmethod
    def from_dict(cls, d: dict) -> Nurse:
        return cls(
            nurse_id=d["nurse_id"],
            name=d["name"],
            skills=frozenset(d["skills"]),
            languages=frozenset(d["languages"]),
            city=d["city"],
            lat=d["lat"],
            lon=d["lon"],
            years_experience=d["years_experience"],
            rating=d["rating"],
            review_count=d["review_count"],
            available_slots=frozenset(d["available_slots"]),
        )


@dataclass(frozen=True)
class CareRequest:
    request_id: str
    patient_id: str
    patient_name: str
    required_skill: str
    language: str
    city: str
    lat: float
    lon: float
    slots: frozenset[str]
    # Nurses who have looked after this patient before.
    previous_nurse_ids: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict:
        return {
            "request_id": self.request_id,
            "patient_id": self.patient_id,
            "patient_name": self.patient_name,
            "required_skill": self.required_skill,
            "language": self.language,
            "city": self.city,
            "lat": self.lat,
            "lon": self.lon,
            "slots": sorted(self.slots, key=slot_sort_key),
            "previous_nurse_ids": list(self.previous_nurse_ids),
        }

    @classmethod
    def from_dict(cls, d: dict) -> CareRequest:
        return cls(
            request_id=d["request_id"],
            patient_id=d["patient_id"],
            patient_name=d.get("patient_name", ""),
            required_skill=d["required_skill"],
            language=d["language"],
            city=d.get("city", ""),
            lat=d["lat"],
            lon=d["lon"],
            slots=frozenset(d["slots"]),
            previous_nurse_ids=tuple(d.get("previous_nurse_ids", ())),
        )

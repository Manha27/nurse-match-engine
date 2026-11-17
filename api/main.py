"""HTTP API.

    uvicorn api.main:app --reload

Interactive docs are served at /docs.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException, Query, Request
from pydantic import BaseModel, Field, field_validator

from matcher import CareRequest, MatchResult, match
from matcher.config import ALL_SLOTS, CITIES, LANGUAGES, SKILLS
from matcher.dataset import load_dataset

SkillName = Literal[*SKILLS]
LanguageName = Literal[*LANGUAGES]
CityName = Literal[*CITIES]


class CareRequestIn(BaseModel):
    request_id: str = "adhoc"
    patient_id: str = "adhoc"
    patient_name: str = ""
    required_skill: SkillName
    language: LanguageName
    city: CityName | None = None
    lat: float = Field(ge=-90, le=90)
    lon: float = Field(ge=-180, le=180)
    slots: list[str] = Field(min_length=1, description='e.g. ["Mon-day", "Tue-day"]')
    previous_nurse_ids: list[str] = Field(default_factory=list)

    model_config = {
        "json_schema_extra": {
            "example": {
                "required_skill": "dementia_care",
                "language": "Marathi",
                "city": "Pune",
                "lat": 18.53,
                "lon": 73.85,
                "slots": ["Mon-day", "Tue-day", "Wed-day", "Thu-day", "Fri-day"],
                "previous_nurse_ids": [],
            }
        }
    }

    @field_validator("slots")
    @classmethod
    def slots_must_exist(cls, slots: list[str]) -> list[str]:
        bad = [s for s in slots if s not in ALL_SLOTS]
        if bad:
            raise ValueError(f"unknown slot(s) {bad}; use the form 'Mon-day' or 'Sat-night'")
        return slots

    def to_domain(self) -> CareRequest:
        return CareRequest(
            request_id=self.request_id,
            patient_id=self.patient_id,
            patient_name=self.patient_name,
            required_skill=self.required_skill,
            language=self.language,
            city=self.city or "",
            lat=self.lat,
            lon=self.lon,
            slots=frozenset(self.slots),
            previous_nurse_ids=tuple(self.previous_nurse_ids),
        )


class NurseOut(BaseModel):
    nurse_id: str
    name: str
    city: str
    lat: float
    lon: float
    skills: list[str]
    languages: list[str]
    years_experience: int
    rating: float
    review_count: int


class MatchOut(BaseModel):
    rank: int
    score: float = Field(description="0 to 1, higher is better")
    distance_km: float
    explanation: str
    factors: dict[str, float] = Field(description="each factor on a 0-1 scale")
    contributions: dict[str, float] = Field(description="each factor's share of the score")
    nurse: NurseOut


class FunnelStep(BaseModel):
    step: str
    nurses_remaining: int


class MatchResponse(BaseModel):
    request_id: str
    eligible_count: int
    funnel: list[FunnelStep]
    notes: list[str]
    matches: list[MatchOut]


class RequestSummary(BaseModel):
    request_id: str
    patient_name: str
    required_skill: str
    language: str
    city: str


def to_response(result: MatchResult) -> MatchResponse:
    return MatchResponse(
        request_id=result.request.request_id,
        eligible_count=result.eligible_count,
        funnel=[FunnelStep(step=s, nurses_remaining=n) for s, n in result.funnel],
        notes=result.notes,
        matches=[
            MatchOut(
                rank=m.rank,
                score=round(m.score, 4),
                distance_km=round(m.distance_km, 2),
                explanation=m.explanation,
                factors={k: round(v, 4) for k, v in m.factors.items() if k != "workload"},
                contributions={k: round(v, 4) for k, v in m.contributions.items() if k != "workload"},
                nurse=NurseOut(
                    nurse_id=m.nurse.nurse_id,
                    name=m.nurse.name,
                    city=m.nurse.city,
                    lat=m.nurse.lat,
                    lon=m.nurse.lon,
                    skills=sorted(m.nurse.skills),
                    languages=sorted(m.nurse.languages),
                    years_experience=m.nurse.years_experience,
                    rating=m.nurse.rating,
                    review_count=m.nurse.review_count,
                ),
            )
            for m in result.matches
        ],
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.nurses, app.state.requests = load_dataset()
    yield


app = FastAPI(
    title="Nurse match engine",
    description="Top nurses for a home-care request, with the reason behind each rank.",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
def health(request: Request) -> dict:
    return {
        "status": "ok",
        "nurses": len(request.app.state.nurses),
        "sample_requests": len(request.app.state.requests),
    }


@app.get("/requests", response_model=list[RequestSummary])
def list_requests(request: Request) -> list[RequestSummary]:
    """The sample care requests, so you have something to try."""
    return [
        RequestSummary(
            request_id=r.request_id,
            patient_name=r.patient_name,
            required_skill=r.required_skill,
            language=r.language,
            city=r.city,
        )
        for r in request.app.state.requests
    ]


@app.get("/requests/{request_id}/matches", response_model=MatchResponse)
def matches_for_sample_request(
    request_id: str, request: Request, top_k: int = Query(5, ge=1, le=20)
) -> MatchResponse:
    """Top nurses for one of the sample requests."""
    care_request = next((r for r in request.app.state.requests if r.request_id == request_id), None)
    if care_request is None:
        raise HTTPException(status_code=404, detail=f"No sample request with id {request_id!r}")
    return to_response(match(care_request, request.app.state.nurses, top_k=top_k))


@app.post("/match", response_model=MatchResponse)
def match_request(
    body: CareRequestIn, request: Request, top_k: int = Query(5, ge=1, le=20)
) -> MatchResponse:
    """Top nurses for a care request sent in the body."""
    return to_response(match(body.to_domain(), request.app.state.nurses, top_k=top_k))

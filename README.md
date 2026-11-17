# nurse-match-engine

Given a home-care request, this ranks the nurses who could take it and says why each one is where it is.

It was built for the TheCarePal AI/ML internship assignment (AI-05, nurse to patient matching). All data is synthetic.

The matching works in two steps. Hard rules (skill, language, availability, and a travel limit) remove nurses who cannot do the job at all. The remaining nurses are scored on distance, rating, experience and whether they have looked after this patient before, then ranked. Each recommendation comes with a sentence built from the same numbers that produced the score.

## Run it

Python 3.11 or newer.

```bash
git clone https://github.com/Manha27/nurse-match-engine.git
cd nurse-match-engine
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# the repo ships with generated data; this recreates it (same seed, same output)
python -m matcher.data_gen --seed 42

pytest                                   # 39 tests
streamlit run app/streamlit_app.py       # demo, opens in the browser
uvicorn api.main:app --reload            # API, docs at http://localhost:8000/docs
python -m matcher.fairness               # fairness table, printed to the terminal
```

The demo is a single narrow column so it should read fine on a phone. Pick a sample request, change the care type, language, days or shift if you like, adjust the scoring weights, and read the top 5. The second tab shows the fairness check.

## API

`POST /match` takes a care request and returns the top nurses (`top_k` defaults to 5, maximum 20).

```bash
curl -X POST "http://localhost:8000/match?top_k=3" \
  -H "Content-Type: application/json" \
  -d '{
    "required_skill": "dementia_care",
    "language": "Marathi",
    "city": "Pune",
    "lat": 18.53,
    "lon": 73.85,
    "slots": ["Mon-day", "Tue-day", "Wed-day", "Thu-day", "Fri-day"]
  }'
```

Trimmed response:

```json
{
  "eligible_count": 6,
  "funnel": [
    {"step": "All nurses", "nurses_remaining": 200},
    {"step": "Offer dementia care", "nurses_remaining": 72},
    {"step": "Speak Marathi", "nurses_remaining": 37},
    {"step": "Free for every requested slot", "nurses_remaining": 18},
    {"step": "Within 25 km", "nurses_remaining": 6}
  ],
  "matches": [
    {
      "rank": 2,
      "nurse": {"nurse_id": "N100", "name": "Radha Deshmukh", "rating": 4.9, "years_experience": 7},
      "score": 0.4315,
      "distance_km": 16.13,
      "explanation": "Rated 4.9 from 40 reviews and has 7 years of experience. Ranks below #1 mainly on distance (-0.15 of score)."
    }
  ]
}
```

Other endpoints: `GET /requests` lists the 100 sample requests, `GET /requests/{id}/matches` ranks one of them, `GET /health`. Bad input (unknown skill, slot like `Funday-day`, latitude out of range) gets a 422 with the reason.

## How it works

**Data** (`matcher/data_gen.py`). 200 nurses and 100 requests across Pune, Mumbai and Delhi, from a fixed seed. Nurses have 1 to 4 skills, spoken languages that depend on the city, a home location, years of experience, a rating with a review count, and weekly availability. Requests ask for one skill and one language, a weekly pattern (weekdays, all week, Mon/Wed/Fri and so on) on a day or night shift, and about a third name a nurse who has looked after the patient before.

**Hard rules** (`matcher/rules.py`). A nurse must offer the skill, speak the language, be free for every requested slot, and live within 25 km. Break one and the nurse is out, no matter how well they score elsewhere. The 25 km limit is my addition to the three rules in the brief, and it can be changed (see DECISIONS.md).

**Scoring** (`matcher/scoring.py`). Each factor becomes a number from 0 to 1, then the weights are applied.

| Factor | Weight | 0 to 1 scale |
|---|---|---|
| Distance | 0.30 | 1 at the door, falling in a straight line to 0 at 25 km |
| Rating | 0.25 | Shrunk towards 4.2 when there are few reviews, then 3.0 to 5.0 mapped to 0 to 1 |
| Experience | 0.15 | Years of experience, counted up to 10 |
| Same nurse as before | 0.30 | 1 if the nurse has cared for this patient, else 0 |

**Explanations** (`matcher/explain.py`). Plain templates over the score breakdown, with no language model. Strong factors are named, weak ones are called out, and every nurse below #1 gets a line saying which factor cost them the most compared with the nurse above.

**Fairness** (`matcher/fairness.py`). See the next section.

## Evaluation

**Hand-checked cases** (`tests/test_hand_checked.py`). 22 small scenarios where I worked out the answer on paper first, with the arithmetic in the test: each hard rule on its own, a nurse failing several rules, one exact score computed by hand, a five-star rating from two reviews losing to a solid record, continuity beating a slightly better nurse (and losing when the history is removed), fewer than five eligible nurses, none eligible, and ties.

**Fairness** (`python -m matcher.fairness`, also in the demo). Each of the 100 requests is ranked and we count who comes first. Raw numbers from the seeded data:

| Measure | Score only | With workload balancing (0.10) |
|---|---|---|
| Requests with no eligible nurse | 2 | 2 |
| Gini of first picks (random-pick baseline 0.76) | 0.83 | 0.78 |
| Most first picks for one nurse | 8 (8%) | 5 (5%) |
| Share of first picks held by the 10 busiest nurses | 43% | 34% |
| Nurses never in anyone's top 5 | 84 | 77 |
| Nurses first for every request they qualify for | 0 | 0 |
| Average match score of the first pick | 0.559 | 0.556 |
| Average distance of the first pick (km) | 7.3 | 6.8 |

What this says: no nurse is always picked first, and the busiest nurse gets 8% of first picks. Ranking on score alone is somewhat more concentrated than picking a random eligible nurse (0.83 against 0.76). Adding a small penalty for nurses who were already picked first brings it close to random (0.78) and the average quality of the first pick barely moves (0.559 to 0.556). The penalty is off by default because the brief asks for ranking on skills, language, distance and feedback, and it is a policy choice for the operations team.

84 of the 200 nurses never reach anyone's top 5. For 48 of them that is because they do not qualify for any of the 100 requests (skill, language, availability or location never lines up). The other 36 qualified for at least one request but never ranked high enough, which is the part worth watching. DECISIONS.md has more on how to read these numbers.

## Cost to run

| Part | Cost |
|---|---|
| Python, NumPy, pandas, pytest | Free, open source |
| FastAPI, uvicorn, httpx | Free, open source |
| Streamlit | Free, open source |
| Language model or paid API | None used |
| Hosting | None needed to run it locally. I did not deploy it. |

No API keys or secrets are needed anywhere. `.env.example` lists the two optional settings.

## Assumptions

- Availability is a weekly repeating pattern of slots (a day plus a day or night shift). There are no dates, leave days or one-off visits.
- A request needs one skill and one language, and the nurse must cover all of its slots.
- Distance is straight-line, not road distance.
- Nurses have no cap on how many patients they can take. Capacity would be a hard rule in a real system.
- A nurse with no reviews has their rating treated as a typical 4.2.
- About 2% of generated requests have no eligible nurse. I left that in on purpose so the empty case gets exercised.

## What I would build next

1. Learn the weights from feedback: log which recommendation was accepted and how the visit went, then fit a model on that instead of hand-picked weights. The brief lists this as a bonus, so I am saying how it would work rather than claiming it is built.
2. Map view of the top matches around the patient.
3. Road or travel-time distance (OSRM or similar), since 5 km in Mumbai and 5 km in Pune are different journeys.
4. Capacity as a hard rule, plus leave and one-off availability pulled from the nurse app.
5. A thin wrapper so the CRM can call the engine, for example as a Cloud Function in the existing Firebase setup.

## AI tools used

Claude (Anthropic) was used to plan the design, draft the code, tests and documentation, and check the arithmetic in the test cases. I reviewed and understand each part, and the choices and their reasons are written down in `docs/DECISIONS.md`.

## Layout

```
matcher/       the engine: data_gen, rules, scoring, explain, engine, fairness
api/main.py    FastAPI app
app/           Streamlit demo
tests/         hand-checked cases, data and fairness checks, API tests
data/          generated nurses.json and requests.json
docs/          DECISIONS.md and the fairness report
```

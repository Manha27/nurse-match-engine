# nurse-match-engine

Given a home-care request, this ranks the nurses who could take it and says why each one is where it is.

It was built for the TheCarePal AI/ML internship assignment (AI-05, nurse to patient matching). All data is synthetic.

The matching works in two steps. Hard rules (skill, language and availability) remove nurses who cannot do the job at all. The remaining nurses are scored on distance, rating, experience and whether they have looked after this patient before, then ranked. Each recommendation comes with a sentence built from the same numbers that produced the score. An optional travel radius can also be switched on, which is off by default.

## Run it

Python 3.11 or newer.

```bash
git clone https://github.com/Manha27/nurse-match-engine.git
cd nurse-match-engine
python -m venv .venv
source .venv/bin/activate            # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt

# the repo ships with generated data; this recreates it (same seed, same output)
python -m matcher.data_gen --seed 42

pytest                                   # 55 tests
streamlit run app/streamlit_app.py       # demo, opens in the browser
uvicorn api.main:app --reload            # API, docs at http://localhost:8000/docs
python -m matcher.fairness               # fairness table, printed to the terminal
python -m matcher.learning               # learned weights from simulated feedback
```

## The demo

`streamlit run app/streamlit_app.py` opens an app with a sidebar menu (also available as tabs at the top right). It is sized to fit the browser window: the page itself never scrolls, and each panel scrolls on its own if it needs to.

- **Find a Nurse.** The left panel holds the care request (sample request, care needed, language, patient's city, days, shift, previous nurse) and the scoring controls. Press "Find Top Matches" to apply them. Four summary cards sit across the top. Below them are the top 5 nurses as cards with availability, distance, experience, skills, languages, the match score and a plain-language reason. "View Details" picks the nurse shown in "Exact Score Breakdown". On the right are the map (use "Expand map" for a large pop-up), the chart of how many nurses each hard rule removed, and the score breakdown.
- **Fairness Analytics.** The fairness numbers and a chart of first picks with and without workload balancing.
- **Learned Weights.** How the weights were learned from simulated feedback and whether it helps.

Things worth trying: R001 (the previous nurse ranks first), R004 (a Pune patient whose #4 and #5 matches live in Mumbai, then switch on the radius), R080 (only 3 eligible nurses).

## API

`POST /match` takes a care request and returns the top nurses (`top_k` defaults to 5, maximum 20). `max_distance_km` is optional and leaves out nurses beyond that distance.

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
  "eligible_count": 18,
  "funnel": [
    {"step": "All nurses", "nurses_remaining": 200},
    {"step": "Offer dementia care", "nurses_remaining": 72},
    {"step": "Speak Marathi", "nurses_remaining": 37},
    {"step": "Free for every requested slot", "nurses_remaining": 18}
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

Other endpoints: `GET /requests` lists the 100 sample requests, `GET /requests/{id}/matches` ranks one of them, `GET /health`. Bad input (unknown skill, a slot like `Funday-day`, latitude out of range, a radius of zero) gets a 422 with the reason. The `city` field is for display only: matching uses `lat` and `lon`.

## How it works

**Data** (`matcher/data_gen.py`). 200 nurses and 100 requests across Pune, Mumbai and Delhi, from a fixed seed. Nurses have 1 to 4 skills, spoken languages that depend on the city, a home location, years of experience, a rating with a review count, and weekly availability. Requests ask for one skill and one language, a weekly pattern (weekdays, all week, Mon/Wed/Fri and so on) on a day or night shift, and about a third name a nurse who has looked after the patient before.

**Hard rules** (`matcher/rules.py`). A nurse must offer the skill, speak the language and be free for every requested slot. Break one and the nurse is out, no matter how well they score elsewhere. Distance is not a hard rule: it is scored. If the operations team wants a service area, the optional radius (`max_distance_km`) adds a fourth rule.

**Scoring** (`matcher/scoring.py`). Each factor becomes a number from 0 to 1, then the weights are applied.

| Factor | Weight | 0 to 1 scale |
|---|---|---|
| Distance | 0.30 | 1 at the door, falling in a straight line to 0 at 25 km, and 0 beyond |
| Rating | 0.25 | Shrunk towards 4.2 when there are few reviews, then 3.0 to 5.0 mapped to 0 to 1 |
| Experience | 0.15 | Years of experience, counted up to 10 |
| Same nurse as before | 0.30 | 1 if the nurse has cared for this patient, else 0 |

Weights are checked on creation: none can be negative and they cannot all be zero. Scores are compared at full precision, and exact ties go to the nearer nurse and then the lower id.

**Explanations** (`matcher/explain.py`). Plain templates over the score breakdown, with no language model. Strong factors are named, weak ones are called out, and every nurse below #1 gets a line saying which factor cost them the most compared with the nurse above.

**Fairness** (`matcher/fairness.py`) and **learned weights** (`matcher/learning.py`) are covered below.

## Evaluation

**Hand-checked cases** (`tests/test_hand_checked.py`). 29 tests built from small scenarios where I worked out the answer on paper first, with the arithmetic in the test: each hard rule on its own, a nurse failing several rules, one exact score computed by hand, a five-star rating from two reviews losing to a solid record, continuity beating a slightly better nurse (and losing when the history is removed), the optional radius, weight validation, fewer than five eligible nurses, none eligible, and ties.

**Fairness** (`python -m matcher.fairness`, also in the demo). Each of the 100 requests is ranked and we count who comes first. Raw numbers from the seeded data:

| Measure | Score only | With workload balancing (0.10) |
|---|---|---|
| Requests with no eligible nurse | 0 | 0 |
| Gini of first picks (random-pick baseline 0.76) | 0.83 | 0.78 |
| Most first picks for one nurse | 8 (8%) | 5 (5%) |
| Share of first picks held by the 10 busiest nurses | 43% | 35% |
| Nurses never in anyone's top 5 | 81 | 74 |
| Nurses first for every request they qualify for | 0 | 0 |
| Average match score of the first pick | 0.555 | 0.552 |
| Average distance of the first pick (km) | 9.6 | 10.2 |

What this says: no nurse is always picked first, and the busiest nurse gets 8% of first picks. Ranking on score alone is somewhat more concentrated than picking a random eligible nurse (0.83 against 0.76). Adding a small penalty for nurses who were already picked first brings it close to random (0.78). The cost is small and visible: the average score of the first pick drops from 0.555 to 0.552 and the first pick lives 0.6 km further away on average. The penalty is off by default because it is a policy choice for the operations team.

81 of the 200 nurses never reach anyone's top 5. For 29 of them that is because they do not qualify for any of the 100 requests (skill, language or availability never lines up). The other 52 qualified for at least one request but never ranked high enough, which is the part worth watching. DECISIONS.md has more on how to read these numbers.

**Distance and the travel radius.** Because distance is scored and not a rule, a nurse in another city can appear when few local nurses qualify. In the sample data 29 of the 497 top-5 slots (6%) go to a nurse based in a different city than the patient, and R004 is an example. The explanation says so ("Held back because the nurse is 131.6 km away") and the demo marks those cards "Based in Mumbai". With the radius switched on at 25 km, 12 requests have fewer than 5 eligible nurses, and two of them (R030 and R092) have none. Which behaviour the operations team wants is their call, so both are available.

## Learned weights (bonus)

The weights in the scoring table are my guess. `matcher/learning.py` shows how they would be learned instead.

There is no real feedback and I may not use any, so the feedback is simulated. A hidden set of family preferences (distance 0.10, rating 0.40, experience 0.20, continuity 0.30, which the engine never sees) decides whether a past match "went well", with some luck added. The steps:

1. Generate past matches by running the normal engine. Usually the top nurse is assigned, but 40% of the time one of ranks 2 to 5 is, so the data shows what separates good nurses from so-so ones.
2. Draw an outcome for each match from the hidden preferences.
3. Fit a logistic regression of outcome on the four factor values. The coefficients, scaled to add up to 1, become the weights.
4. Rank 100 requests the model never saw, with each set of weights, and see how likely the first pick is to go well under the hidden preferences.

Results (`docs/learning_report.md`, all repeatable):

| Factor | Hand-set | Hidden preference | Learned from 2,000 matches |
|---|---|---|---|
| Distance | 0.30 | 0.10 | 0.08 |
| Rating | 0.25 | 0.40 | 0.46 |
| Experience | 0.15 | 0.20 | 0.18 |
| Continuity | 0.30 | 0.30 | 0.28 |

| Weights used | Chance the first pick goes well |
|---|---|
| Hand-set | 60.7% |
| Learned | 64.6% |
| Best possible (the hidden preferences) | 64.7% |

The learning curve, averaged over 3 simulated histories:

| Past matches | Distance from hidden weights | Chance first pick goes well |
|---|---|---|
| 50 | 0.42 | 62.1% |
| 100 | 0.51 | 61.8% |
| 250 | 0.42 | 62.7% |
| 500 | 0.25 | 63.0% |
| 1,000 | 0.12 | 64.3% |
| 2,000 | 0.10 | 64.6% |

The first pick improves from the start, but the learned weights themselves are not close to the hidden ones until around 1,000 matches. My hand-set weights are 0.40 away from the hidden ones, so below 500 matches the learned weights are no closer than my guess. With small histories the fit is also noisy: in 5 of the 18 fits (all at 500 matches or fewer) one coefficient came out negative and was set to zero. Only 3 histories were averaged, so read the curve as a guide.

Limits to keep in mind. The feedback comes from a known linear rule, so this shows the method recovers what it should, not how it would do on real feedback, which is noisier and has its own biases. The gain is modest, about 3.9 points. In the demo, tick "Use weights learned from simulated feedback" under "Scoring" to rank with them.

## Cost to run

| Part | Cost |
|---|---|
| Python, NumPy, pandas, scikit-learn, pytest | Free, open source |
| FastAPI, uvicorn, httpx | Free, open source |
| Streamlit, pydeck | Free, open source |
| Map background tiles | Loaded from a free public tile service in the browser, no key or account |
| Language model or paid API | None used |
| Hosting | None needed to run it locally. I did not deploy it. |

No API keys or secrets are needed anywhere. `.env.example` lists the two optional settings.

## Assumptions

- Availability is a weekly repeating pattern of slots (a day plus a day or night shift). There are no dates, leave days or one-off visits.
- A request needs one skill and one language, and the nurse must cover all of its slots.
- Distance is straight-line, not road distance. Beyond 25 km every nurse gets 0 for distance, so a nurse 40 km away and one 130 km away look the same on that factor.
- Nurses have no cap on how many patients they can take. Capacity would be a hard rule in a real system.
- A nurse with no reviews has their rating treated as a typical 4.2.
- Every sample request has at least 3 eligible nurses (the smallest pool is R080), so the "no eligible nurse" case only shows up in the demo when you narrow a request or switch on the 25 km radius (R030 and R092 then have nobody).

## What I would build next

1. Feed the learning step with real feedback: log which recommendation was accepted and how the visit went. The method is built and tested on simulated feedback, but nothing here has seen real data.
2. Road or travel-time distance (OSRM or similar), since 5 km in Mumbai and 5 km in Pune are different journeys, and a distance score that keeps falling past 25 km.
3. Capacity as a hard rule, plus leave and one-off availability pulled from the nurse app.
4. A service area per city, so the radius is a setting the operations team owns rather than a number in the request.
5. A thin wrapper so the CRM can call the engine, for example as a Cloud Function in the existing Firebase setup.

## AI tools used

Claude (Anthropic) was used to plan the design, draft the code, tests and documentation, and check the arithmetic in the test cases. I reviewed and understand each part, and the choices and their reasons are written down in `docs/DECISIONS.md`.

## Layout

```
matcher/       the engine: data_gen, rules, scoring, explain, engine, fairness, learning
api/main.py    FastAPI app
app/           Streamlit demo
tests/         hand-checked cases, data and fairness checks, learning checks, API tests
data/          generated nurses.json and requests.json
docs/          DECISIONS.md, the fairness report and the learning report
.streamlit/    demo theme
```

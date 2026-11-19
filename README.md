# Nurse Match Engine

A small matching system for home-care requests. It filters out nurses who do not meet the request, ranks the remaining nurses, and explains each result.

**Live demo:** https://nurse-patient-match.streamlit.app/

All nurses, patients, locations, ratings and requests in this project are synthetic. The app is a demonstration, not a tool for making real care or hiring decisions.

## How matching works

The engine has two stages:

1. **Eligibility:** a nurse must offer the required care skill, speak the requested language and be available for every requested day and shift. A travel radius can be enabled as an additional rule.
2. **Ranking:** eligible nurses are scored using distance, rating, experience and whether they have cared for the patient before. The weighted contributions add up to the displayed score. The explanation is generated from those same values; it does not use a language model.

Distance is straight-line distance calculated from coordinates, not driving time. It affects the score by default but does not exclude a nurse unless the optional radius is enabled.

## Run locally

Python 3.11 or newer is required.

```bash
git clone https://github.com/Manha27/nurse-match-engine.git
cd nurse-match-engine
python -m venv .venv

# Windows PowerShell
.venv\Scripts\Activate.ps1

# macOS / Linux
# source .venv/bin/activate

pip install -r requirements.txt
pytest -q
streamlit run app/streamlit_app.py
```

The repository includes the generated dataset, so the app can be run without generating data again. To recreate it with the same seed:

```bash
python -m matcher.data_gen --seed 42
```

The generator writes 200 nurses and 100 requests to `data/`.

## Demo pages

### Find a Nurse

Choose a sample request or adjust its care skill, language, city, days, shift and previous nurse. The scoring options allow the weights to be changed and an optional travel radius to be applied. The results show the eligible count, ranked nurses, match scores, map, eligibility funnel and a score breakdown.

The score breakdown uses:

| Factor | Default weight | Meaning |
|---|---:|---|
| Distance | 0.30 | Closer nurses receive a higher value. The value falls to zero at 25 km. |
| Rating | 0.25 | Review ratings, adjusted towards 4.2 when there are few reviews. |
| Experience | 0.15 | Years of experience, capped at 10. |
| Continuity | 0.30 | 1 when the nurse has cared for the patient before; otherwise 0. |

Weights are normalised before scoring, so their relative proportions matter. The score is a ranking aid, not a probability of good care.

### Fairness Analytics

This page studies how often each nurse appears first or in the top five across the 100 sample requests. It compares normal ranking with an optional workload penalty that reduces the score of nurses who have already been picked first in earlier requests in the simulation.

The Gini value measures how concentrated first picks are: 0 means an even distribution; larger values mean more concentration. The page also compares the result with a random eligible-nurse baseline and reports how much the match score changes when balancing is used.

This is **allocation fairness within the synthetic nurse pool**. It is not a demographic fairness or bias audit: the dataset has no real people or protected-group labels, and the analysis does not establish fairness in a real service. First picks are recommendations, not confirmed assignments.

### Learned Weights

This page is a research demonstration of how scoring weights could be estimated from past outcomes. It creates simulated matches, assigns simulated good/bad outcomes using invented hidden preferences, and fits a logistic regression model to those examples. The learned weights are then evaluated on sample requests not used for fitting.

The feedback is generated again by the program; it is not collected from the current device or from clicks in the app. Searching for nurses does not create feedback or update a personal history. The report is deterministic for the supplied seeds and is cached during an app session.

The current experiment learns from 2,000 simulated matches. On the held-out requests, the recorded run gave a 60.7% expected success rate with the hand-set weights, 64.6% with learned weights and 64.7% with the simulator's hidden weights. These are results under the simulator's invented rule, not real-world performance estimates.

## Useful sample requests for testing

| Request | Patient | What it demonstrates |
|---|---|---|
| `R001` | Radha Pereira | Previous nurse N065 is eligible and ranks first, showing continuity. |
| `R004` | Deepa Joshi | Without a radius, nurses from another city can appear when local options are limited. With a 25 km radius, only three remain. |
| `R018` | Shweta Chauhan | Only four nurses qualify; the previous nurse is not eligible. |
| `R030` | Geeta Singh | No nurse remains when a 25 km radius is applied. |
| `R044` | Rohit Pereira | A 25 km radius leaves only two nurses. |
| `R061` | Savita Bhosale | Dementia-care and Punjabi-language requirements, with five eligible nurses. |
| `R080` | Rohit Shinde | Smallest standard eligible pool: three nurses. |
| `R092` | Amit Mehta | Another no-match case with a 25 km radius. |

For the cleanest first run, start with `R001`, then try `R004` with and without the radius, `R080`, and `R030`. The remaining requests show additional edge cases.

## API

The FastAPI service uses the same matching engine as the Streamlit app. Start it in a second terminal:

```bash
uvicorn api.main:app --reload
```

Open `http://localhost:8000/docs` for the interactive API documentation.

- `GET /health` checks that the service is running.
- `GET /requests` lists the sample requests.
- `GET /requests/{request_id}/matches` ranks a sample request.
- `POST /match` ranks a custom request. `top_k` defaults to 5 (maximum 20); `max_distance_km` is optional.

Example request:

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

Invalid skills, languages, slots, coordinates and non-positive travel radii are rejected by request validation.

## Evaluation

The project includes 55 automated tests. They cover the hard rules, multiple simultaneous rule failures, exact score calculations, rating adjustment for small review counts, continuity, optional radius, ties, empty and short result lists, API validation, fairness calculations and the simulated learning process.

Run them with:

```bash
pytest -q
```

The detailed design choices and reproducible reports are in `docs/DECISIONS.md`, `docs/fairness_report.md` and `docs/learning_report.md`.

## Assumptions and limits

- All data is generated. No patient, nurse or visit records are used.
- A request contains one care skill, one language and a repeating weekly availability pattern.
- Availability is represented by day and day/night shift. Dates, leave and one-off visits are not modelled.
- Distance is straight-line distance, not road distance or travel time.
- The engine does not model nurse capacity, current caseload or travel between consecutive visits.
- Ratings are adjusted towards a 4.2 prior when review counts are low; a nurse with no reviews receives the prior value.
- The default weights are hand-set choices. The learned weights use simulated outcomes only.
- The fairness page measures recommendation distribution, not actual assignments or demographic fairness.
- This project is for demonstration and academic evaluation. It should not be used to select a nurse for a real patient without clinical, operational and human review.

## AI tool use

Claude Code was used for front-end design assistance. The matching logic, backend implementation, data generation, tests and evaluation were developed by me.

## Project structure

```text
app/                 Streamlit demo
api/                 FastAPI endpoints
matcher/             Data models, rules, scoring, matching, fairness and learning
data/                 Synthetic nurse and request data
tests/                Automated tests
docs/                 Design decisions and evaluation reports
.streamlit/           Streamlit theme and settings
```

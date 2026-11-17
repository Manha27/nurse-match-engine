# Design decisions

Each entry says what I chose, what else I considered, and why. Numbers quoted here come from the seeded dataset (seed 42).

## 1. Hard rules and soft scores are separate

Skill, language, availability and the travel limit are yes/no filters. Distance, rating, experience and continuity are scored.

The alternative was one big score with a heavy penalty for missing a skill. I did not do that because a penalty can always be outweighed, and a family that needs Marathi should never be sent a nurse who does not speak it, however good her rating is. Separating the two also makes the explanation honest: eligibility is a fact, ranking is a judgement.

## 2. The 25 km limit is a hard rule, and it is my addition

The brief lists skill, availability and language as hard rules and distance as scored. I kept distance scored, but also added a cap, because without one a nurse in Mumbai can appear in a Pune family's list when nobody closer qualifies. That is not a usable recommendation. The cap is one constant (`MAX_DISTANCE_KM`) and the demo has a slider for it. I flagged it as an assumption in the README.

## 3. Straight-line distance

Haversine on latitude and longitude. Road distance would be better but needs a routing service, which adds cost, an external dependency and failure modes, for a ranking where relative order mostly survives. Listed under "next" in the README.

## 4. Linear distance decay

Factor is `1 - km/25`. I looked at exponential decay but could not justify its steepness from any data, whereas a straight line is easy to say out loud: 5 km is 0.8, 12.5 km is 0.5.

## 5. Ratings are shrunk towards a typical value

`(reviews x rating + 10 x 4.2) / (reviews + 10)`. A nurse with one five-star review should not beat a nurse with 80 reviews averaging 4.7. The prior (4.2, worth 10 reviews) is a judgement call, deliberately modest so a handful of reviews moves the number but a handful of reviews cannot make it perfect. There is a test for exactly this case. A nurse with zero reviews is scored on the typical value, not on a 0.

## 6. Experience stops counting at 10 years

A 20-year nurse is not twice as suitable as a 10-year nurse for most home care. Capping removes the effect of long tenure dominating the other factors.

## 7. Continuity weighs as much as distance (0.30)

For dementia and elderly care, a familiar face matters. The weight is high enough to beat a somewhat better nurse (test: a previous nurse at 5 km with a 4.0 rating beats a 1 km, 4.9-rated nurse, 0.6975 to 0.66675) but not high enough to override quality entirely: a previous nurse who is far away and poorly rated loses to a great nurse next door. If the previous nurse fails a hard rule, they are left out and the result carries a note saying why, so the coordinator knows to call the family.

## 8. A weighted sum, not a trained model

There is no real feedback data, and the brief says not to use any real records. A model trained on invented labels would just learn my own assumptions with extra steps, and it could not be explained. A weighted sum can be read, tuned in the demo and defended. Learning weights from real feedback is the first item under "next".

## 9. Explanations are templates, not a language model

The explanation is built from the same factor values that produced the score, so it cannot say something the numbers do not support, it costs nothing, and it gives the same text every time. A language model would sound more natural but could invent reasons.

The comparison line ("Ranks below #1 mainly on distance") is the part I care about most, since "why is she third" is the question a coordinator actually gets asked.

## 10. How fairness is measured

What I count: how often each nurse is ranked first and in the top 5, over all 100 requests.

- **Always first.** A nurse who is #1 for every request they qualify for (at least 5). The result is zero.
- **Concentration.** Share of first picks held by the busiest nurse and by the busiest 10. Busiest nurse: 8%. Busiest 10: 43%.
- **Gini of first picks.** 0.83, but the number needs context. With 100 requests and 200 nurses, even a perfectly even spread leaves half the nurses with zero picks, so a Gini of 0 is impossible. I compare against a baseline: picking a random eligible nurse for each request gives 0.76. So score ranking is a little more concentrated than random (0.83), which is expected when better nurses really do rank higher.

"Nurses never in a top 5" is reported but is not a pass/fail measure. Of the 84 such nurses, 48 do not qualify for any of the 100 requests, so no ranking could have picked them. The remaining 36 qualified at least once and never ranked high enough, and that group is the one to look at.

## 11. Workload balancing is available but off by default

The fix for concentration is a small penalty for nurses already picked first (`workload` weight, 0.10 in the evaluation). In the simulation it takes the busiest nurse from 8 first picks to 5, the busiest 10's share from 43% to 34%, and the Gini from 0.83 to 0.78, while the average score of the first pick moves from 0.559 to 0.556.

It is off by default because it is a policy decision about trading match quality for spread, and it needs a real notion of load (current patients, not just this simulation's count). The simulation is also pessimistic: it assumes every first pick is accepted and nobody ever finishes a case. So treat the improvement as an indication, not a forecast.

## 12. Availability is weekly slots

A slot is a day plus a shift, such as `Mon-day` or `Sat-night`. A nurse is available if the request's slots are a subset of hers. This is simple to explain and to test. It cannot represent leave, one-off visits or partial shifts.

## 13. Tie-breaking

Equal scores go to the nearer nurse, then the lower id. Results are the same on every run, which matters for tests and for a coordinator seeing a stable list.

## 14. Generated data

The generator is seeded, so everything is repeatable. Language mix depends on the city, dementia nurses usually also do elderly care, ratings drift up slightly with experience, and request patterns are the common ones (weekdays, all week, three days, weekends). About 2% of requests end up with no eligible nurse. I tuned the availability mix once to stop that being 5%, and left a few in on purpose so the "no match" path is tested and shown in the demo. This is invented data and shows nothing about real nurse supply.

## 15. API and demo

`POST /match` accepts a request body, because that is how the CRM would call it. `GET /requests/{id}/matches` exists so the sample data is easy to try. Input is validated with Pydantic, so unknown skills, malformed slots and out-of-range coordinates return a 422 with the reason instead of an error. The Streamlit app calls the matching code directly rather than going through the API, so one command runs the whole demo, and both use the same `match()` function.

## Known limits

- Straight-line distance, no road or travel time.
- No capacity limit per nurse.
- Weekly slots only, so no dates or leave.
- Weights are hand-set, not learned.
- The fairness simulation treats every first pick as accepted.
- Everything is on synthetic data, so the numbers show behaviour, not real-world performance.

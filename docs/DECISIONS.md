# Design decisions

Each entry says what I chose, what else I considered, and why. Numbers quoted here come from the seeded dataset (seed 42).

## 1. Hard rules and soft scores are separate

Skill, language and availability are yes/no filters. Distance, rating, experience and continuity are scored. A travel radius is available as an optional fourth filter.

The alternative was one big score with a heavy penalty for missing a skill. I did not do that because a penalty can always be outweighed, and a family that needs Marathi should never be sent a nurse who does not speak it, however good her rating is. Separating the two also makes the explanation honest: eligibility is a fact, ranking is a judgement.

## 2. Distance is scored, not a hard rule (the radius is optional)

The brief lists skill, availability and language as hard rules and distance as scored, and an earlier version of mine added a 25 km cap as a hard rule. A review pointed out that this quietly goes beyond the brief and can leave a family with no recommendation at all. I changed it: distance is only scored by default, and `max_distance_km` is an optional radius that adds a fourth rule when someone sets it.

The cost of that choice is visible in the data. With no radius, 29 of the 497 top-5 slots (5.8%) go to a nurse in a different city than the patient, because nobody closer qualified (R004's #4 is a Mumbai nurse 131.6 km away). The explanation says so and the demo marks those cards. With a 25 km radius, 12 requests have fewer than 5 eligible nurses and two (R030, R092) have none. Which of those outcomes is better is an operations decision, so both are offered and the default follows the brief.

## 3. Straight-line distance

Haversine on latitude and longitude. Road distance would be better but needs a routing service, which adds cost, an external dependency and failure modes, for a ranking where relative order mostly survives. Listed under "next" in the README.

## 4. Linear distance decay

Factor is `1 - km/25`, and 0 beyond 25 km, so a nurse 40 km away and one 130 km away look the same on that factor. I looked at exponential decay but could not justify its steepness from any data, whereas a straight line is easy to say out loud: 5 km is 0.8, 12.5 km is 0.5.

## 5. Ratings are shrunk towards a typical value

`(reviews x rating + 10 x 4.2) / (reviews + 10)`. A nurse with one five-star review should not beat a nurse with 80 reviews averaging 4.7. The prior (4.2, worth 10 reviews) is a judgement call, deliberately modest so a handful of reviews moves the number but a handful of reviews cannot make it perfect. There is a test for exactly this case. A nurse with zero reviews is scored on the typical value, not on a 0.

## 6. Experience stops counting at 10 years

A 20-year nurse is not twice as suitable as a 10-year nurse for most home care. Capping removes the effect of long tenure dominating the other factors.

## 7. Continuity weighs as much as distance (0.30)

For dementia and elderly care, a familiar face matters. The weight is high enough to beat a somewhat better nurse (test: a previous nurse at 5 km with a 4.0 rating beats a 1 km, 4.9-rated nurse, 0.6975 to 0.66675) but not high enough to override quality entirely: a previous nurse who is far away and poorly rated loses to a great nurse next door. If the previous nurse fails a hard rule, they are left out and the result carries a note saying why, so the coordinator knows to call the family.

## 8. A weighted sum, not a black-box model

There is no real feedback data, and the brief says not to use any real records. A ranking model trained on invented labels would just learn my own assumptions, and it could not be explained to a coordinator. A weighted sum can be read, tuned in the demo and defended. Decision 16 covers the one place where I do learn something, which is the weights themselves.

## 9. Explanations are templates, not a language model

The explanation is built from the same factor values that produced the score, so it cannot say something the numbers do not support, it costs nothing, and it gives the same text every time. A language model would sound more natural but could invent reasons.

The comparison line ("Ranks below #1 mainly on distance") is the part I care about most, since "why is she third" is the question a coordinator actually gets asked.

## 10. How fairness is measured

What I count: how often each nurse is ranked first and in the top 5, over all 100 requests.

- **Always first.** A nurse who is #1 for every request they qualify for (at least 5). The result is zero.
- **Concentration.** Share of first picks held by the busiest nurse and by the busiest 10. Busiest nurse: 8%. Busiest 10: 43%.
- **Gini of first picks.** 0.83, but the number needs context. With 100 requests and 200 nurses, even a perfectly even spread leaves half the nurses with zero picks, so a Gini of 0 is impossible. I compare against a baseline: picking a random eligible nurse for each request gives 0.76. So score ranking is a little more concentrated than random (0.83), which is expected when better nurses really do rank higher.

"Nurses never in a top 5" is reported but is not a pass/fail measure. Of the 81 such nurses, 29 do not qualify for any of the 100 requests, so no ranking could have picked them. The remaining 52 qualified at least once and never ranked high enough, and that group is the one to look at.

## 11. Workload balancing is available but off by default

The fix for concentration is a small penalty for nurses already picked first (`workload` weight, 0.10 in the evaluation). In the simulation it takes the busiest nurse from 8 first picks to 5, the busiest 10's share from 43% to 35%, and the Gini from 0.83 to 0.78, while the average score of the first pick moves from 0.555 to 0.552 and its average distance from 9.6 to 10.2 km.

It is off by default because it is a policy decision about trading match quality for spread, and it needs a real notion of load (current patients, not just this simulation's count). The simulation is also pessimistic: it assumes every first pick is accepted and nobody ever finishes a case. So treat the improvement as an indication, not a forecast.

## 12. Availability is weekly slots

A slot is a day plus a shift, such as `Mon-day` or `Sat-night`. A nurse is available if the request's slots are a subset of hers. This is simple to explain and to test. It cannot represent leave, one-off visits or partial shifts.

## 13. Tie-breaking

Scores are compared at full precision (no rounding), equal scores go to the nearer nurse, then the lower id. Results are the same on every run, which matters for tests and for a coordinator seeing a stable list.

## 14. Generated data

The generator is seeded, so everything is repeatable. Language mix depends on the city, dementia nurses usually also do elderly care, ratings drift up slightly with experience, and request patterns are the common ones (weekdays, all week, three days, weekends). I loosened the availability mix once because too many requests had nobody eligible. Now every request has at least 3 eligible nurses (smallest pool: R080), so the "no match" path is covered by tests and shows in the demo only when you narrow a request or switch on the 25 km radius. This is invented data and shows nothing about real nurse supply.

## 15. API and demo

`POST /match` accepts a request body, because that is how the CRM would call it. `GET /requests/{id}/matches` exists so the sample data is easy to try. Input is validated with Pydantic, so unknown skills, malformed slots and out-of-range coordinates return a 422 with the reason instead of an error. The Streamlit app calls the matching code directly rather than going through the API, so one command runs the whole demo, and both use the same `match()` function.

## 16. Learned weights: fit the weights, not the ranking

The bonus item asks for weights learned from simulated feedback. I kept the weighted sum (decision 8) and learned only the four weights, with a logistic regression on "did the match go well" against the four factor values. The coefficients are readable, so the result is still four numbers a coordinator can inspect and change, and the explanations keep working unchanged.

Choices worth knowing:

- **Hidden preferences.** The simulator needs a ground truth to generate outcomes from. I made one up (distance 0.10, rating 0.40, experience 0.20, continuity 0.30) and chose it to differ from my defaults, since otherwise there would be nothing to learn. That makes the result a check that the method works, not evidence about real families.
- **Exploration.** Past matches are the engine's top pick 60% of the time and a random one of ranks 2 to 5 otherwise. Real feedback only covers nurses who were actually sent, so if the engine always sent its top pick the model would only see top picks and could not tell what makes them good. This selection effect is the main way real feedback differs from my simulation.
- **Negative coefficients are set to zero.** For these four factors, "more of it made things worse" is not a rule I want applied, and a negative weight would invert the ranking on that factor. This does happen with small histories: in the 18 fits behind the learning curve, 5 had a negative coefficient (the most negative was -0.87), all at 500 matches or fewer. With 1,000 or more matches none did. That is noise, and part of why the curve is bad at small sizes.
- **Light regularisation** (`C=10`) so a small history does not produce wild weights.
- **How it is judged.** On 100 requests the model never saw, I rank with each set of weights and score the first pick with the hidden preferences. Ranking with the hidden weights themselves is the ceiling, since nothing can pick better under that rule.
- **What the numbers say.** The learned weights come out at 0.08, 0.46, 0.18 and 0.28, against hidden 0.10, 0.40, 0.20 and 0.30. The first pick goes well 64.6% of the time against 60.7% with my weights and a ceiling of 64.7%. The learning curve (average of 3 simulated histories) shows 50 to 250 matches barely help (61.8% to 62.7%), and 1,000 matches get close to the ceiling (64.3%).
- **Not switched on by default.** The learned weights are an option in the demo. Replacing the defaults should wait until there are a few hundred matches with real feedback.

## 17. The demo is built for a laptop screen

The Streamlit app uses a wide layout: controls on the left, then the top 5 as cards with a stacked bar showing what each factor contributed, a map next to them, and the funnel of how many nurses each rule removed. A card's bar is built from the same contributions that add up to the score, and the exact numbers are in an expander. The map is pydeck; I fit its view to the points myself because the default fit clipped the markers. I checked the layout in headless Chrome at laptop and phone widths.

## Known limits

- Straight-line distance, no road or travel time, and no difference between 40 km and 130 km once past 25 km.
- No capacity limit per nurse.
- Weekly slots only, so no dates or leave.
- Weights are hand-set by default. The learned ones come from simulated feedback only.
- The fairness simulation treats every first pick as accepted.
- Everything is on synthetic data, so the numbers show behaviour, not real-world performance.

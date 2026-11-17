"""Demo UI for coordinators.

    streamlit run app/streamlit_app.py

The app calls the matching code directly rather than going through the API, so
one command is enough to try it. The API in api/main.py wraps the same function.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from matcher import CareRequest, Weights, match
from matcher.config import DAYS, LANGUAGES, MAX_DISTANCE_KM, SHIFTS, SKILLS, label
from matcher.dataset import load_dataset
from matcher.fairness import comparison_table, evaluate

st.set_page_config(page_title="Nurse match engine", layout="centered")


@st.cache_resource
def get_data():
    return load_dataset()


@st.cache_data(show_spinner="Running every request through the engine...")
def get_fairness(balance_weight: float):
    nurses, requests = get_data()
    return evaluate(nurses, requests), evaluate(nurses, requests, balance_weight=balance_weight)


def request_label(r: CareRequest) -> str:
    return f"{r.request_id} · {r.patient_name} · {label(r.required_skill)} · {r.language} · {r.city}"


def request_editor(base: CareRequest) -> CareRequest:
    """Let the user change the sample request. Keys include the request id so a new selection resets the fields."""
    k = base.request_id
    base_days = sorted({s.split("-")[0] for s in base.slots}, key=DAYS.index)
    base_shift = next(iter(base.slots)).split("-")[1]

    with st.expander("Change the request"):
        skill = st.selectbox("Care needed", SKILLS, index=SKILLS.index(base.required_skill),
                             format_func=label, key=f"skill_{k}")
        language = st.selectbox("Language the family needs", LANGUAGES,
                                index=LANGUAGES.index(base.language), key=f"lang_{k}")
        days = st.multiselect("Days", DAYS, default=base_days, key=f"days_{k}")
        shift = st.radio("Shift", SHIFTS, index=SHIFTS.index(base_shift), horizontal=True, key=f"shift_{k}")
        previous = st.text_input("Previous nurse id (optional)", value=", ".join(base.previous_nurse_ids),
                                 key=f"prev_{k}", help="A nurse who has looked after this patient before, e.g. N042")

    if not days:
        st.warning("Pick at least one day.")
        st.stop()

    return CareRequest(
        request_id=base.request_id,
        patient_id=base.patient_id,
        patient_name=base.patient_name,
        required_skill=skill,
        language=language,
        city=base.city,
        lat=base.lat,
        lon=base.lon,
        slots=frozenset(f"{d}-{shift}" for d in days),
        previous_nurse_ids=tuple(p.strip().upper() for p in previous.split(",") if p.strip()),
    )


def weights_editor() -> tuple[Weights, float]:
    defaults = Weights()
    with st.expander("Adjust how nurses are scored"):
        st.caption("Weights are normalised, so only their proportions matter.")
        d = st.slider("Distance", 0.0, 1.0, defaults.distance, 0.05)
        r = st.slider("Rating", 0.0, 1.0, defaults.rating, 0.05)
        e = st.slider("Experience", 0.0, 1.0, defaults.experience, 0.05)
        c = st.slider("Same nurse as before", 0.0, 1.0, defaults.continuity, 0.05)
        max_km = st.slider("Furthest a nurse may travel (km)", 5, 50, int(MAX_DISTANCE_KM))
    if d + r + e + c == 0:
        st.error("At least one weight must be above zero.")
        st.stop()
    return Weights(distance=d, rating=r, experience=e, continuity=c), float(max_km)


def show_matches(request: CareRequest, weights: Weights, max_km: float) -> None:
    nurses, _ = get_data()
    result = match(request, nurses, weights=weights, max_distance_km=max_km)

    st.subheader(f"Top {len(result.matches)} for {request.patient_name or 'this request'}")
    st.caption(
        f"{label(request.required_skill).capitalize()} · {request.language} · {request.city} · "
        f"{len(request.slots)} slots a week · {result.eligible_count} eligible nurses"
    )

    for note in result.notes:
        st.warning(note)

    for m in result.matches:
        n = m.nurse
        with st.container(border=True):
            st.markdown(f"**#{m.rank} {n.name}** · {n.nurse_id}")
            st.progress(min(max(m.score, 0.0), 1.0), text=f"Match score {m.score:.2f}")
            st.write(m.explanation)
            st.caption(
                f"{n.city} · {', '.join(label(s) for s in sorted(n.skills))} · "
                f"speaks {', '.join(sorted(n.languages))}"
            )
            with st.expander("Score breakdown"):
                share = {k: v / weights.total for k, v in weights.as_dict().items()}
                rows = [
                    {
                        "Factor": k.capitalize(),
                        "Value (0-1)": round(m.factors[k], 2),
                        "Weight": f"{share[k]:.0%}",
                        "Adds to score": round(m.contributions[k], 3),
                    }
                    for k in ("distance", "rating", "experience", "continuity")
                ]
                st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

    with st.expander("How the list was narrowed down"):
        for step, remaining in result.funnel:
            st.write(f"**{remaining}** · {step}")


def find_tab() -> None:
    nurses, requests = get_data()
    by_id = {r.request_id: r for r in requests}
    choice = st.selectbox("Care request", list(by_id), format_func=lambda i: request_label(by_id[i]))
    request = request_editor(by_id[choice])
    weights, max_km = weights_editor()
    show_matches(request, weights, max_km)


def fairness_tab() -> None:
    st.write(
        "Every sample request is run through the engine and we count who comes first. "
        "A good result is no nurse always first, and work spread at least as evenly as picking "
        "a random eligible nurse."
    )
    balance_weight = st.slider("Workload balancing weight", 0.0, 0.30, 0.10, 0.05,
                               help="How much a nurse's earlier first-picks count against them.")
    plain, balanced = get_fairness(balance_weight)

    st.dataframe(comparison_table(plain, balanced, balance_weight), hide_index=True, width="stretch")

    st.caption("First picks per nurse, busiest 20 (score only)")
    top = plain.per_nurse.sort_values("top1", ascending=False).head(20).set_index("nurse_id")
    st.bar_chart(top["top1"])


def main() -> None:
    st.title("Nurse match engine")
    st.caption("Hard rules decide who is eligible. Scores decide the order. Every rank comes with a reason.")
    tab_find, tab_fair = st.tabs(["Find a nurse", "Fairness check"])
    with tab_find:
        find_tab()
    with tab_fair:
        fairness_tab()


main()

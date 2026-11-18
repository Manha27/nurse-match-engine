"""Demo UI for coordinators.

    streamlit run app/streamlit_app.py

The app calls the matching code directly rather than going through the API, so
one command is enough to try it. The API in api/main.py wraps the same function.
"""

from __future__ import annotations

import math
from html import escape

import pandas as pd
import pydeck as pdk
import streamlit as st

from matcher import CareRequest, Match, MatchResult, Weights, eligible_nurses, match
from matcher.config import CITIES, DAYS, LANGUAGES, SHIFTS, SKILLS, SUGGESTED_RADIUS_KM, label
from matcher.dataset import load_dataset
from matcher.fairness import comparison_table, evaluate
from matcher.learning import run_experiment, weights_table

st.set_page_config(page_title="Nurse match engine", layout="wide")

FACTOR_COLORS = {
    "distance": "#2563EB",
    "rating": "#F59E0B",
    "experience": "#7C3AED",
    "continuity": "#059669",
}
FACTOR_NAMES = {
    "distance": "Distance",
    "rating": "Rating",
    "experience": "Experience",
    "continuity": "Continuity",
}

CSS = """
<style>
html { font-size: 112.5%; }
.block-container { padding-top: 1rem; padding-bottom: 3rem; max-width: 1800px; }
.st-key-mapbox [data-testid="stElementToolbar"] { display: none; }
[data-testid="stMetricValue"] { font-size: 1.65rem; }
[data-testid="stMetric"] { padding: 10px 14px; }
.nm-hero { background: linear-gradient(120deg, #0F766E, #134E4A); color: #fff; border-radius: 16px;
  padding: 16px 26px; display: flex; justify-content: space-between; align-items: center;
  gap: 16px; flex-wrap: wrap; margin-bottom: 8px; }
.nm-hero h1 { margin: 0; font-size: 1.6rem; font-weight: 700; letter-spacing: -0.01em; color: #fff; }
.nm-hero p { margin: 2px 0 0; opacity: 0.92; font-size: 0.95rem; }
.nm-badges span { display: inline-block; background: rgba(255,255,255,0.16); border-radius: 999px;
  padding: 5px 13px; margin-left: 8px; font-size: 0.82rem; }
.nm-summary { background: #fff; border: 1px solid #DCE5E6; border-radius: 14px; padding: 12px 18px 8px; margin-bottom: 10px; }
.nm-summary .who { font-size: 1.15rem; font-weight: 650; margin-bottom: 8px; }
.nm-chip { display: inline-block; background: #E8F1F0; color: #0F4F4A; border-radius: 999px;
  padding: 3px 11px; margin: 0 6px 6px 0; font-size: 0.86rem; font-weight: 500; }
.nm-chip.ok { background: #DCFCE7; color: #166534; }
.nm-chip.warn { background: #FEF3C7; color: #92400E; }
.nm-chip.quiet { background: #EEF2F3; color: #3F5560; }
.nm-legend { font-size: 0.88rem; color: #3F5560; margin: 4px 0 10px; }
.nm-legend i { display: inline-block; width: 10px; height: 10px; border-radius: 3px; margin: 0 5px 0 12px; vertical-align: -1px; }
.nm-card { display: flex; gap: 14px; background: #fff; border: 1px solid #DCE5E6; border-radius: 14px;
  padding: 15px 18px; margin-bottom: 12px; box-shadow: 0 1px 2px rgba(15,50,55,0.05); }
.nm-card.top { border-color: #0F766E; box-shadow: 0 0 0 2px rgba(15,118,110,0.16); }
.nm-rank { flex: 0 0 34px; height: 34px; border-radius: 50%; background: #E3EFEE; color: #0F766E;
  font-weight: 700; display: flex; align-items: center; justify-content: center; }
.nm-card.top .nm-rank { background: #0F766E; color: #fff; }
.nm-main { flex: 1; min-width: 0; }
.nm-head { display: flex; align-items: baseline; gap: 10px; flex-wrap: wrap; margin-bottom: 8px; }
.nm-name { font-size: 1.08rem; font-weight: 650; }
.nm-id { color: #6B7F88; font-size: 0.85rem; }
.nm-score { margin-left: auto; font-weight: 700; font-size: 1.05rem; color: #0F766E; }
.nm-track { display: flex; height: 9px; background: #E7EDEE; border-radius: 99px; overflow: hidden; margin-bottom: 10px; }
.nm-track span { display: block; height: 100%; }
.nm-expl { font-size: 1rem; line-height: 1.45; margin-bottom: 9px; }
.nm-fn { margin-bottom: 11px; }
.nm-fn-top { display: flex; justify-content: space-between; font-size: 0.88rem; margin-bottom: 4px; }
.nm-fn-track { height: 8px; background: #E7EDEE; border-radius: 99px; overflow: hidden; }
.nm-fn-fill { height: 100%; background: #0F766E; border-radius: 99px; }
.nm-avail { font-size: 0.88rem; color: #3F5560; margin: 2px 0 8px; }
.nm-empty { background: #FFF7ED; border: 1px solid #FDBA74; border-radius: 12px; padding: 16px 18px; color: #7C2D12; }
</style>
"""


# ---- data -------------------------------------------------------------------


@st.cache_resource
def get_data():
    return load_dataset()


@st.cache_data(show_spinner="Running every request through the engine...")
def get_fairness(balance_weight: float):
    nurses, requests = get_data()
    return evaluate(nurses, requests), evaluate(nurses, requests, balance_weight=balance_weight)


@st.cache_data(show_spinner="Simulating past matches and learning weights...")
def get_learning():
    nurses, requests = get_data()
    return run_experiment(nurses, requests)


# ---- small formatting helpers -----------------------------------------------


def request_label(r: CareRequest) -> str:
    return f"{r.request_id} · {r.patient_name} · {label(r.required_skill)} · {r.language} · {r.city}"


def describe_days(days: list[str]) -> str:
    days = sorted(days, key=DAYS.index)
    if len(days) == 7:
        return "Every day"
    if days == list(DAYS[:5]):
        return "Weekdays"
    if days == list(DAYS[5:]):
        return "Weekends"
    return ", ".join(days)


def describe_availability(slots: frozenset[str]) -> str:
    """e.g. 'days Mon-Fri · nights every day' from a nurse's weekly slots."""
    parts = []
    for shift in SHIFTS:
        days = [d for d in DAYS if f"{d}-{shift}" in slots]
        if days:
            parts.append(f"{shift}s: {describe_days(days)}")
    return " · ".join(parts) or "no weekly slots"


def chip(text: str, kind: str = "") -> str:
    return f'<span class="nm-chip {kind}">{escape(text)}</span>'


# ---- request and scoring controls -------------------------------------------


def request_editor(base: CareRequest, known_ids: set[str]) -> CareRequest:
    """Fields start from the sample request. Keys include its id, so picking another one resets them."""
    k = base.request_id
    base_days = sorted({s.split("-")[0] for s in base.slots}, key=DAYS.index)
    base_shift = next(iter(base.slots)).split("-")[1]

    skill = st.selectbox("Care needed", SKILLS, index=SKILLS.index(base.required_skill),
                         format_func=label, key=f"skill_{k}")
    language = st.selectbox("Language the family needs", LANGUAGES,
                            index=LANGUAGES.index(base.language), key=f"lang_{k}")
    days = st.pills("Days", DAYS, selection_mode="multi", default=base_days, key=f"days_{k}")
    shift = st.segmented_control("Shift", SHIFTS, default=base_shift, key=f"shift_{k}") or base_shift
    city = st.selectbox("Patient's city", list(CITIES), index=list(CITIES).index(base.city), key=f"city_{k}",
                        help="Changing the city moves the patient to that city's centre (simulated location).")
    previous = st.text_input("Nurse who looked after this patient before (optional)",
                             value=", ".join(base.previous_nurse_ids), key=f"prev_{k}",
                             placeholder="e.g. N042")

    previous_ids = tuple(p.strip().upper() for p in previous.split(",") if p.strip())
    unknown = [p for p in previous_ids if p not in known_ids]
    if unknown:
        st.warning(f"No nurse with id {', '.join(unknown)}.")
    if not days:
        st.warning("Pick at least one day.")
        st.stop()

    return CareRequest(
        request_id=base.request_id,
        patient_id=base.patient_id,
        patient_name=base.patient_name,
        required_skill=skill,
        language=language,
        city=city,
        lat=base.lat if city == base.city else CITIES[city][0],
        lon=base.lon if city == base.city else CITIES[city][1],
        slots=frozenset(f"{d}-{shift}" for d in days),
        previous_nurse_ids=previous_ids,
    )


def scoring_controls() -> tuple[Weights, float | None]:
    defaults = Weights()
    with st.expander("Scoring weights"):
        use_learned = st.checkbox(
            "Use weights learned from simulated feedback",
            help="Fitted on simulated past matches. See the 'Learned weights' tab.",
        )
        if use_learned:
            weights = get_learning().learned
            shares = {k: v / weights.total for k, v in weights.as_dict().items() if k != "workload"}
            st.caption(" · ".join(f"{FACTOR_NAMES[k]} {v:.0%}" for k, v in shares.items()))
        else:
            st.caption("Only the proportions matter. They are normalised.")
            d = st.slider("Distance", 0.0, 1.0, defaults.distance, 0.05)
            r = st.slider("Rating", 0.0, 1.0, defaults.rating, 0.05)
            e = st.slider("Experience", 0.0, 1.0, defaults.experience, 0.05)
            c = st.slider("Same nurse as before", 0.0, 1.0, defaults.continuity, 0.05)
            if d + r + e + c == 0:
                st.error("At least one weight must be above zero.")
                st.stop()
            weights = Weights(distance=d, rating=r, experience=e, continuity=c)

    limit = st.toggle(
        "Only nurses within a travel radius",
        help="Off: distance only affects the score, so a far-away nurse can still appear. "
             "On: nurses beyond the radius are left out entirely.",
    )
    radius = st.slider("Radius (km)", 5, 100, int(SUGGESTED_RADIUS_KM), disabled=not limit)
    return weights, float(radius) if limit else None


# ---- results ----------------------------------------------------------------


def summary_html(request: CareRequest, result: MatchResult) -> str:
    days = sorted({s.split("-")[0] for s in request.slots}, key=DAYS.index)
    shift = next(iter(request.slots)).split("-")[1]
    chips = "".join([
        chip(label(request.required_skill).capitalize()),
        chip(request.language),
        chip(request.city),
        chip(f"{describe_days(days)}, {shift} shift", "quiet"),
    ])
    return (f'<div class="nm-summary"><div class="who">{escape(request.patient_name or "Care request")}</div>'
            f"{chips}</div>")


def card_html(m: Match, request: CareRequest) -> str:
    n = m.nurse
    segments = "".join(
        f'<span style="width:{m.contributions[k] * 100:.2f}%;background:{FACTOR_COLORS[k]}" '
        f'title="{FACTOR_NAMES[k]} {m.contributions[k]:.2f}"></span>'
        for k in FACTOR_COLORS
    )
    chips = [chip("✓ Free for every requested slot", "ok"), chip(f"{m.distance_km:.1f} km away"), chip(f"Rated {n.rating:.1f} ({n.review_count})")
             if n.review_count else chip("No reviews yet", "quiet"),
             chip(f"{n.years_experience} yrs experience")]
    if n.city != request.city:
        chips.append(chip(f"Based in {n.city}", "warn"))
    skills = "".join(chip(label(s), "quiet") for s in sorted(n.skills))
    langs = chip("Speaks " + ", ".join(sorted(n.languages)), "quiet")

    return (
        f'<div class="nm-card{" top" if m.rank == 1 else ""}">'
        f'<div class="nm-rank">{m.rank}</div><div class="nm-main">'
        f'<div class="nm-head"><span class="nm-name">{escape(n.name)}</span>'
        f'<span class="nm-id">{escape(n.nurse_id)}</span><span class="nm-score">{m.score:.2f}</span></div>'
        f'<div class="nm-track">{segments}</div>'
        f'<div class="nm-expl">{escape(m.explanation)}</div>'
        f'<div>{"".join(chips)}</div>'
        f'<div class="nm-avail">Weekly availability, {escape(describe_availability(n.available_slots))}</div>'
        f'<div>{skills}{langs}</div>'
        f"</div></div>"
    )


def legend_html() -> str:
    items = "".join(f'<i style="background:{c}"></i>{FACTOR_NAMES[k]}' for k, c in FACTOR_COLORS.items())
    return f'<div class="nm-legend">Score bar:{items}</div>'


def funnel_html(funnel: list[tuple[str, int]]) -> str:
    top = funnel[0][1] or 1
    rows = "".join(
        f'<div class="nm-fn"><div class="nm-fn-top"><span>{escape(step)}</span><b>{n}</b></div>'
        f'<div class="nm-fn-track"><div class="nm-fn-fill" style="width:{n / top * 100:.1f}%"></div></div></div>'
        for step, n in funnel
    )
    return f"<div>{rows}</div>"


def breakdown_table(matches: list[Match]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Rank": m.rank,
                "Nurse": f"{m.nurse.name} ({m.nurse.nurse_id})",
                "Score": m.score,
                **{FACTOR_NAMES[k]: m.contributions[k] for k in FACTOR_COLORS},
            }
            for m in matches
        ]
    )


def map_view(points: pd.DataFrame, width_px: int = 360, height_px: int = 380) -> pdk.ViewState:
    """Centre and zoom so every point fits a panel of roughly this size, with some margin.

    pydeck's own helper assumes a different panel size and clips points at the edges.
    At zoom z, deck.gl's Web Mercator world is 512 * 2**z pixels around.
    """
    lat, lon = points["lat"], points["lon"]
    span_lon = max(lon.max() - lon.min(), 0.005) * 1.6
    span_lat = max(lat.max() - lat.min(), 0.005) * 1.6
    centre_lat = float((lat.max() + lat.min()) / 2)
    zoom_lon = math.log2(width_px * 360 / (512 * span_lon))
    zoom_lat = math.log2(height_px * 360 * math.cos(math.radians(centre_lat)) / (512 * span_lat))
    zoom = max(3.0, min(13.0, zoom_lon, zoom_lat))
    return pdk.ViewState(latitude=centre_lat, longitude=float((lon.max() + lon.min()) / 2), zoom=zoom)


def build_map(request: CareRequest, matches: list[Match], radius_km: float | None,
              width_px: int = 460, height_px: int = 460) -> pdk.Deck:
    patient = pd.DataFrame([{
        "lat": request.lat, "lon": request.lon,
        "name": request.patient_name or "Patient", "detail": "Patient's home",
    }])
    nurses = pd.DataFrame([{
        "lat": m.nurse.lat, "lon": m.nurse.lon, "label": str(m.rank), "name": m.nurse.name,
        "detail": f"#{m.rank} · {m.distance_km:.1f} km · score {m.score:.2f}",
        "from_lat": request.lat, "from_lon": request.lon,
    } for m in matches])

    # Nurses living almost on the same spot would hide each other. Nudge them apart on the map only.
    seen: dict[tuple[float, float], int] = {}
    for i, row in nurses.iterrows():
        key = (round(row["lat"], 3), round(row["lon"], 3))
        seen[key] = seen.get(key, 0) + 1
        if seen[key] > 1:
            nurses.loc[i, "lon"] += 0.0012 * (seen[key] - 1)

    layers = [
        pdk.Layer("LineLayer", nurses, get_source_position="[from_lon, from_lat]",
                  get_target_position="[lon, lat]", get_color=[148, 163, 184, 200], get_width=2),
        pdk.Layer("ScatterplotLayer", nurses, get_position="[lon, lat]", get_fill_color=[15, 118, 110, 235],
                  radius_min_pixels=14, radius_max_pixels=14, pickable=True),
        pdk.Layer("TextLayer", nurses, get_position="[lon, lat]", get_text="label", get_size=15,
                  get_color=[255, 255, 255, 255], get_alignment_baseline="'center'", get_text_anchor="'middle'"),
        pdk.Layer("ScatterplotLayer", patient, get_position="[lon, lat]", get_fill_color=[220, 38, 38, 245],
                  radius_min_pixels=10, radius_max_pixels=10, pickable=True),
    ]
    if radius_km is not None:
        ring = patient.assign(radius_m=radius_km * 1000)
        layers.insert(0, pdk.Layer("ScatterplotLayer", ring, get_position="[lon, lat]", get_radius="radius_m",
                                   filled=False, stroked=True, get_line_color=[15, 118, 110, 200],
                                   line_width_min_pixels=2))

    points = pd.concat([patient[["lon", "lat"]], nurses[["lon", "lat"]]])
    if radius_km is not None:  # keep the whole ring in view
        dlat = radius_km / 111.195
        dlon = dlat / math.cos(math.radians(request.lat))
        ring_edges = pd.DataFrame({
            "lon": [request.lon - dlon, request.lon + dlon],
            "lat": [request.lat - dlat, request.lat + dlat],
        })
        points = pd.concat([points, ring_edges])
    view = map_view(points, width_px, height_px)
    return pdk.Deck(layers=layers, initial_view_state=view, map_style="light",
                    tooltip={"html": "<b>{name}</b><br/>{detail}"})


MAP_NOTE = ("Red dot: patient. Teal numbers: the top 5, by rank. Grey lines are straight-line distance, "
            "not driving routes. Locations are simulated, and nurses at almost the same spot are nudged apart.")


@st.dialog("Where they are", width="large")
def map_dialog(request: CareRequest, matches: list[Match], radius_km: float | None) -> None:
    st.pydeck_chart(build_map(request, matches, radius_km, width_px=1150, height_px=600), height=600)
    st.caption(MAP_NOTE)
    if st.button("Close map", type="primary"):
        st.rerun()


def previous_nurse_kpi(
    request: CareRequest, matches: list[Match], eligible_ids: set[str]
) -> tuple[str, str]:
    if not request.previous_nurse_ids:
        return "Previous nurse", "None"
    prev = request.previous_nurse_ids[0]
    ranks = {m.nurse.nurse_id: m.rank for m in matches}
    if prev in ranks:
        return f"Previous nurse ({prev})", f"Ranked #{ranks[prev]}"
    return f"Previous nurse ({prev})", "Not in top 5" if prev in eligible_ids else "Not eligible"


def show_results(request: CareRequest, weights: Weights, radius_km: float | None) -> None:
    nurses, _ = get_data()
    result = match(request, nurses, weights=weights, max_distance_km=radius_km)
    eligible_ids = {n.nurse_id for n, _ in eligible_nurses(request, nurses, radius_km)}

    st.html(summary_html(request, result))

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Eligible nurses", f"{result.eligible_count} of {len(nurses)}", border=True)
    k2.metric("Best score", f"{result.matches[0].score:.2f}" if result.matches else "n/a", border=True)
    k3.metric("Closest in top 5", f"{min(m.distance_km for m in result.matches):.1f} km" if result.matches else "n/a",
              border=True)
    k4.metric(*previous_nurse_kpi(request, result.matches, eligible_ids), border=True)

    for note in result.notes:
        st.warning(note)

    if not result.matches:
        st.html('<div class="nm-empty"><b>No nurse meets every requirement.</b> The bars below show '
                "which rule removed the most nurses. Try other days, a different shift or a wider radius.</div>")
        st.html(funnel_html(result.funnel))
        return

    cards, side = st.columns([1.35, 1], gap="large")
    with cards:
        st.markdown(f"##### Top {len(result.matches)} nurses")
        st.html(legend_html())
        st.html("".join(card_html(m, request) for m in result.matches))
        with st.expander("Exact score breakdown"):
            st.dataframe(
                breakdown_table(result.matches), hide_index=True, width="stretch",
                column_config={
                    "Score": st.column_config.ProgressColumn("Score", min_value=0.0, max_value=1.0, format="%.3f"),
                    **{FACTOR_NAMES[k]: st.column_config.NumberColumn(format="%.3f") for k in FACTOR_COLORS},
                },
            )
            st.caption("Each column is that factor's weight times its 0 to 1 value. The score is their sum. "
                       "Cards show it to 2 decimals, so a card can differ by 0.01 from the rounded parts.")
    with side:
        st.markdown("##### Where they are")
        with st.container(key="mapbox"):
            st.pydeck_chart(build_map(request, result.matches, radius_km), height=460)
            if st.button("Expand map", icon=":material/open_in_full:", width="stretch"):
                map_dialog(request, result.matches, radius_km)
        st.caption(MAP_NOTE)
        st.markdown("##### How the list was narrowed down")
        st.html(funnel_html(result.funnel))


def find_tab() -> None:
    nurses, requests = get_data()
    by_id = {r.request_id: r for r in requests}
    known_ids = {n.nurse_id for n in nurses}

    controls, results = st.columns([1, 3.2], gap="large")
    with controls:
        with st.container(border=True):
            st.markdown("##### Care request")
            choice = st.selectbox("Sample request", list(by_id), format_func=lambda i: request_label(by_id[i]))
            request = request_editor(by_id[choice], known_ids)
        with st.container(border=True):
            st.markdown("##### Scoring")
            weights, radius_km = scoring_controls()
    with results:
        show_results(request, weights, radius_km)


# ---- other tabs -------------------------------------------------------------


def fairness_tab() -> None:
    st.write(
        "Every sample request is run through the engine and we count who comes first. "
        "A good result is no nurse always first, and work spread at least as evenly as picking "
        "a random eligible nurse."
    )
    balance_weight = st.slider("Workload balancing weight", 0.0, 0.30, 0.10, 0.05,
                               help="How much a nurse's earlier first-picks count against them.")
    plain, balanced = get_fairness(balance_weight)

    a, b, c, d = st.columns(4)
    a.metric("Always-first nurses", len(plain.always_first), border=True)
    b.metric("Busiest nurse's share of first picks", f"{plain.max_top1_share:.0%}", border=True)
    c.metric("Gini of first picks", f"{plain.gini_top1:.2f}", border=True)
    d.metric("With balancing, Gini", f"{balanced.gini_top1:.2f}",
             delta=f"{balanced.gini_top1 - plain.gini_top1:+.2f}", delta_color="inverse", border=True)

    st.caption(
        "Gini is 0 when work is spread perfectly evenly. Picking a random eligible nurse for each "
        f"request scores {plain.gini_top1_random:.2f} on this data, so that is the number to compare with."
    )
    df = plain.per_nurse
    qualify = df[df["eligible_requests"] > 0]
    stuck = qualify[qualify["top5"] == 0]
    st.caption(
        f"Of the {len(df)} nurses, {len(df) - len(qualify)} qualify for none of the 100 requests, so no ranking "
        f"could pick them. {len(stuck)} more qualify at least once but never reach a top 5; those are the ones "
        "worth looking at. First-pick counts measure how recommendations are spread, not who is actually "
        "assigned work."
    )
    with st.expander("Exposure: how often each nurse reaches a top 5 when they qualify"):
        exp = qualify[qualify["eligible_requests"] >= 5].assign(
            exposure=lambda d: d["top5"] / d["eligible_requests"]
        ).sort_values("exposure").head(15)
        st.dataframe(
            exp[["nurse_id", "name", "eligible_requests", "top5", "exposure"]].rename(columns={
                "nurse_id": "Nurse", "name": "Name", "eligible_requests": "Requests qualified for",
                "top5": "Times in top 5", "exposure": "Exposure rate"}),
            hide_index=True, width="stretch",
            column_config={"Exposure rate": st.column_config.NumberColumn(format="percent")},
        )
        st.caption("Exposure rate = times in a top 5 divided by requests the nurse qualifies for. "
                   "Shown for the 15 lowest, among nurses who qualify for at least 5 requests.")
    left, right = st.columns([1.1, 1], gap="large")
    with left:
        st.dataframe(comparison_table(plain, balanced, balance_weight), hide_index=True, width="stretch")
    with right:
        st.caption("First picks per nurse, for the 20 nurses picked most often with score only")
        top = plain.per_nurse.sort_values("top1", ascending=False).head(20).set_index("nurse_id")
        chart = pd.DataFrame({
            "Score only": top["top1"],
            "With balancing": balanced.per_nurse.set_index("nurse_id").loc[top.index, "top1"],
        })
        st.bar_chart(chart, stack=False, color=["#0F766E", "#F59E0B"])


def learning_tab() -> None:
    st.write(
        "The scoring weights are a starting guess. Here they are learned from feedback on past "
        "matches instead. The feedback is simulated: a hidden set of family preferences decides "
        "whether each past match went well, and the engine never sees it. A logistic regression is "
        "fitted on how each factor related to the outcome, and its coefficients become the weights."
    )
    report = get_learning()
    table = weights_table(report)

    left, right = st.columns([1, 1.2], gap="large")
    with left:
        st.dataframe(
            table, hide_index=True, width="stretch",
            column_config={c: st.column_config.NumberColumn(format="%.2f") for c in table.columns[1:]},
        )
        st.caption("Chance that the first pick goes well, on the 100 sample requests the model never saw.")
        a, b, c = st.columns(3)
        a.metric("Hand-set", f"{report.satisfaction_default:.1%}", border=True)
        b.metric("Learned", f"{report.satisfaction_learned:.1%}", border=True)
        c.metric("Best possible", f"{report.satisfaction_oracle:.1%}", border=True)
    with right:
        st.bar_chart(table.set_index("Factor")[["Hand-set", "Hidden preference", "Learned"]],
                     stack=False, color=["#94A3B8", "#F59E0B", "#0F766E"])

    st.markdown("##### How much feedback is enough?")
    st.caption("Distance between learned and hidden weights, by number of past matches (lower is better).")
    st.line_chart(report.curve.set_index("n_matches")["weight_error"], color="#0F766E", height=240)
    st.caption(
        f"The weights above were learned from {report.n_training_matches} simulated matches. "
        "Because the feedback comes from a known rule, this shows the method works, not how well it "
        "would do on real feedback."
    )


def main() -> None:
    nurses, requests = get_data()
    st.html(CSS)
    st.html(
        '<div class="nm-hero"><div><h1>Nurse match engine</h1>'
        "<p>Hard rules decide who is eligible. Scores decide the order. Every rank comes with a reason.</p></div>"
        f'<div class="nm-badges"><span>{len(nurses)} nurses</span><span>{len(requests)} sample requests</span>'
        "<span>Synthetic data</span></div></div>"
    )
    tab_find, tab_fair, tab_learn = st.tabs(["Find a nurse", "Fairness check", "Learned weights"])
    with tab_find:
        find_tab()
    with tab_fair:
        fairness_tab()
    with tab_learn:
        learning_tab()


main()

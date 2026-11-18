"""Demo UI for coordinators.

    streamlit run app/streamlit_app.py

The app calls the matching code directly rather than going through the API, so
one command is enough to try it. The API in api/main.py wraps the same function.
"""

from __future__ import annotations

import base64
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

st.set_page_config(page_title="Nurse Match Engine", layout="wide", initial_sidebar_state="collapsed")

TEAL = "#0F5F5A"
FACTOR_NAMES = {"distance": "Distance", "rating": "Rating", "experience": "Experience", "continuity": "Continuity"}
PAGES = ["Find a Nurse", "Fairness Analytics", "Learned Weights"]

CSS = """
<style>
/* ---- page frame: fit the window, scroll inside panels instead ---- */
html { font-size: clamp(17px, 0.95vw, 30px); }
html, body, .stApp { background: #F3F7F7; }
header[data-testid="stHeader"] { display: none; }
[data-testid="stMain"] .block-container { padding: 0.9rem 1.6rem 0.5rem 1.6rem; max-width: 100%; }
[data-testid="stMain"] { overflow: hidden; }
[data-testid="stAppViewContainer"] { height: 100vh; overflow: hidden; }
.stApp { height: 100vh; overflow: hidden; }
[data-testid="stMain"] .block-container { height: 100vh; overflow: hidden; }
[data-testid="stVerticalBlock"] { gap: 0.6rem; }

/* ---- no sidebar: navigation lives in the top-right tabs ---- */
[data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"], [data-testid="stSidebarCollapseButton"] { display: none !important; }

/* ---- header ---- */
.nm-title h1 { margin: 0; font-size: 2rem; font-weight: 600; color: #10252B; letter-spacing: -0.01em; line-height: 1.15; }
.nm-title p { margin: 2px 0 0; color: #26404A; font-size: 1rem; }
.st-key-hero { background: linear-gradient(120deg, #0B4A47 0%, #0F5F5A 55%, #1B8579 100%);
  border-radius: 16px; padding: 0.9rem 1.4rem; margin-bottom: 0.7rem;
  box-shadow: 0 2px 8px rgba(11,74,71,0.18); }
.st-key-hero .nm-title h1 { color: #fff; }
.st-key-hero .nm-title p { color: #D6ECE9; }
.st-key-hero .nm-badge { display: inline-block; background: rgba(255,255,255,0.16); color: #fff;
  border-radius: 999px; padding: 2px 12px; font-size: 0.8rem; margin-right: 0.4rem; }
.st-key-hero [data-baseweb="button-group"] { background: rgba(255,255,255,0.14) !important; }
.st-key-hero .st-key-toptabs_box { background: transparent !important; }
.st-key-hero .st-key-toptabs_box button { color: #E3F2F0 !important; }
.st-key-hero .st-key-toptabs_box button[aria-checked="true"],
.st-key-hero .st-key-toptabs_box button[aria-pressed="true"],
.st-key-hero .st-key-toptabs_box button[kind="segmented_controlActive"] {
  background: #fff !important; color: #0F5F5A !important; }
.st-key-toptabs_box [data-baseweb="button-group"] { background: #E8F0F0; border-radius: 999px; padding: 4px; gap: 2px; }
.st-key-toptabs_box button { border: 0 !important; border-radius: 999px !important; background: transparent !important;
  color: #4B6168 !important; font-weight: 500; min-height: 44px; padding: 0 22px; }
.st-key-toptabs_box button p { font-size: 1.05rem; }
.st-key-toptabs_box button[aria-checked="true"], .st-key-toptabs_box button[aria-pressed="true"], .st-key-toptabs_box button[kind="segmented_controlActive"] {
  background: #0F5F5A !important; color: #fff !important; }

/* ---- panels (internal scroll) ---- */
.st-key-left { height: calc(100vh - 8.6rem) !important; max-height: calc(100vh - 8.6rem) !important; overflow-y: auto !important; padding-right: 4px; }
.st-key-mid, .st-key-right { height: calc(100vh - 13.6rem) !important; max-height: calc(100vh - 13.6rem) !important; overflow-y: auto !important; padding-right: 4px; }
.st-key-fullpage { height: calc(100vh - 8.6rem) !important; max-height: calc(100vh - 8.6rem) !important; overflow-y: auto !important; padding-right: 6px; }
.st-key-left > div, .st-key-right > div { max-width: 100%; }
.st-key-left::-webkit-scrollbar, .st-key-mid::-webkit-scrollbar, .st-key-right::-webkit-scrollbar,
.st-key-fullpage::-webkit-scrollbar { width: 8px; }
.st-key-left::-webkit-scrollbar-thumb, .st-key-mid::-webkit-scrollbar-thumb, .st-key-right::-webkit-scrollbar-thumb,
.st-key-fullpage::-webkit-scrollbar-thumb { background: #C4D4D4; border-radius: 8px; }
.nm-panel-title { font-size: 1.15rem; font-weight: 700; color: #10252B; margin: 2px 0 6px; }
.nm-sub { font-weight: 700; font-size: 1.05rem; margin: 8px 0 2px; color: #10252B; }
[data-testid="stForm"] { border: 0; padding: 0; }
[data-testid="stWidgetLabel"] p { font-size: 0.95rem; color: #26404A; }
.st-key-left [data-baseweb="select"] > div { min-height: 38px; }

/* ---- KPI cards ---- */
.nm-kpis { display: grid; grid-template-columns: 1.15fr 0.9fr 1fr 1.35fr; gap: 14px; margin-bottom: 4px; }
.nm-kpi { display: flex; align-items: center; gap: 14px; background: #fff; border: 1px solid #DCE6E7;
  border-radius: 14px; padding: 10px 12px; min-width: 0; gap: 10px; }
.nm-kpi .ic { flex: 0 0 2.6rem; height: 2.6rem; border-radius: 50%; background: #E3F0EF; display: flex; align-items: center; justify-content: center; }
.nm-kpi .ic img { width: 22px; height: 22px; }
.nm-kpi .lab { font-size: 0.95rem; white-space: normal; line-height: 1.2; color: #26404A; }
.nm-kpi .val { font-size: 1.4rem; font-weight: 500; color: #10252B; line-height: 1.2; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }

/* ---- nurse cards ---- */
.st-key-mid [data-testid="stVerticalBlockBorderWrapper"] { background: #fff; border: 1px solid #DCE6E7; border-radius: 14px; }
.nm-rank { width: 2rem; height: 2rem; border-radius: 50%; background: #E3F0EF; color: #0F5F5A; font-weight: 700;
  display: flex; align-items: center; justify-content: center; font-size: 1rem; margin-top: 3px; }
.nm-rank.top { background: #0F5F5A; color: #fff; }
.nm-head { display: flex; align-items: baseline; gap: 10px; flex-wrap: wrap; margin-bottom: 6px; }
.nm-name { font-size: 1.3rem; font-weight: 650; color: #10252B; }
.nm-id { color: #5B7078; font-size: 0.95rem; }
.nm-pill { background: #D7EEE9; color: #0F5F5A; border-radius: 999px; padding: 2px 12px; font-size: 0.8rem; font-weight: 600; }
.nm-line { font-size: 1rem; color: #1D343B; margin-bottom: 8px; }
.nm-chip { display: inline-block; background: #EEF3F4; color: #28434B; border-radius: 6px; padding: 3px 10px; margin: 0 6px 4px 0; font-size: 0.95rem; }
.nm-chip.warn { background: #FEF3C7; color: #92400E; }
.nm-expl { font-size: 1rem; color: #1D343B; line-height: 1.5; margin: 6px 0 10px; }
.nm-score { text-align: center; }
.nm-score .lab { font-size: 0.95rem; font-weight: 600; color: #10252B; }
.nm-score .num { font-size: 2.3rem; font-weight: 700; color: #0F5F5A; line-height: 1.15; }
.st-key-mid .stButton > button { background: #0F5F5A; color: #fff; border: 0; border-radius: 8px; width: 100%; font-weight: 600; min-height: 36px; }
.st-key-mid .stButton > button:hover { background: #0B4A47; color: #fff; }
.st-key-mid .stButton > button[kind="primary"] { box-shadow: 0 0 0 3px rgba(15,95,90,0.25); }

/* ---- right column ---- */
.st-key-mapbox button { min-height: 32px; padding: 0 8px; white-space: nowrap; }
.st-key-toptabs_box { background: #E8F0F0; border-radius: 999px; }
.st-key-mapbox, .st-key-breakbox { background: #fff; border: 1px solid #DCE6E7; border-radius: 14px; padding: 12px 14px; }
.nm-fn { margin-bottom: 8px; }
.nm-fn-top { display: flex; justify-content: space-between; font-size: 1rem; margin-bottom: 3px; }
.nm-fn-track { height: 7px; background: #E3EBEC; border-radius: 99px; overflow: hidden; }
.nm-fn-fill { height: 100%; background: #0F5F5A; border-radius: 99px; }
.nm-bt { width: 100%; table-layout: fixed; border-collapse: collapse; font-size: 1.05rem; }
.nm-bt th:first-child, .nm-bt td:first-child { width: 31%; }
.nm-bt th { text-align: right; font-weight: 500; color: #3B5159; padding: 5px 4px; border-bottom: 1px solid #E3EBEC; }
.nm-bt td { text-align: right; padding: 5px 4px; border-bottom: 1px solid #EEF3F4; }
.nm-bt th:first-child, .nm-bt td:first-child { text-align: left; }
.nm-bt tr.total td { background: #DDEFEC; font-weight: 700; font-size: 1rem; border: 0; }
.nm-empty { background: #FFF7ED; border: 1px solid #FDBA74; border-radius: 12px; padding: 16px 18px; color: #7C2D12; }
.nm-about { background: #fff; border: 1px solid #DCE6E7; border-radius: 14px; padding: 18px 24px; max-width: 820px; line-height: 1.6; }
[data-testid="stMetric"] { background: #fff; border: 1px solid #DCE6E7; border-radius: 12px; padding: 10px 14px; }

/* ---- cards: score + View Details centred together ---- */
.nm-score { text-align: center; }
.st-key-mid [data-testid="stColumn"]:last-child [data-testid="stVerticalBlock"] { align-items: center; gap: 0.35rem; }
.st-key-mid [data-testid="stColumn"]:last-child .stButton { width: 100%; display: flex; justify-content: center; }
.st-key-mid .stButton > button { min-height: 42px; }
.st-key-mid .stButton > button p { font-size: 1rem; }

/* ---- map: only the "Expand map" button, no built-in full-screen icon ---- */
.st-key-mapbox [data-testid="stElementToolbar"], .st-key-mapwrap [data-testid="stElementToolbar"], .st-key-mapwrap_big [data-testid="stElementToolbar"] { display: none !important; }
.st-key-mapbox .stButton > button { min-height: 42px; padding: 0 14px; background: #fff; border: 1px solid #0F5F5A; color: #0F5F5A; border-radius: 8px; }
.st-key-mapbox .stButton > button:hover { background: #E3F0EF; }
.st-key-mapbox .stButton > button p { font-size: 1.1rem; font-weight: 600; }
.st-key-mapwrap [data-testid="stDeckGlJsonChart"], .st-key-mapwrap_big [data-testid="stDeckGlJsonChart"] { border-radius: 12px; overflow: hidden; border: 1px solid #DCE6E7; }
.st-key-breakbox button[role="tab"] p { font-size: 1.05rem; font-weight: 600; }
/* ---- phones and tablets (up to 1100px wide): let the whole page scroll, stack everything ---- */
@media (max-width: 1100px) {
  html { font-size: 16px; }
  [data-testid="stHorizontalBlock"] { flex-wrap: wrap !important; }
  [data-testid="stColumn"] { min-width: 100% !important; flex: 1 1 100% !important; width: 100% !important; }
  .stApp, [data-testid="stAppViewContainer"] { height: 100dvh; }
  [data-testid="stMain"] { overflow-y: auto !important; height: 100% !important; }
  [data-testid="stMain"] .block-container { height: auto !important; overflow: visible !important; padding: 0.8rem 0.9rem 4rem; }
  .st-key-left, .st-key-mid, .st-key-right, .st-key-fullpage { height: auto !important; max-height: none !important; overflow: visible !important; padding-right: 0; }
  .nm-title h1 { font-size: 1.7rem; }
  .nm-title p { font-size: 0.95rem; }
  .st-key-toptabs_box button { padding: 0 6px; min-height: 40px; }
  .st-key-toptabs_box button p { font-size: 0.85rem; white-space: nowrap; }
  .nm-kpis { grid-template-columns: 1fr 1fr; gap: 10px; }
  .nm-kpi .val { font-size: 1.25rem; }
}
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


# ---- helpers ----------------------------------------------------------------


def request_label(r: CareRequest) -> str:
    return f"{r.request_id} · {r.patient_name} · {label(r.required_skill)}"


def chip(text: str, kind: str = "") -> str:
    return f'<span class="nm-chip {kind}">{escape(text)}</span>'


def icon_img(paths: str) -> str:
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="#0F5F5A" '
           f'stroke-width="2" stroke-linecap="round" stroke-linejoin="round">{paths}</svg>')
    return f'<img alt="" src="data:image/svg+xml;base64,{base64.b64encode(svg.encode()).decode()}">'


def pin_icon(text: str, fill: str) -> dict:
    """A crisp numbered circle as an SVG icon (text layers go blurry in WebGL)."""
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" width="96" height="96" viewBox="0 0 96 96">'
        f'<circle cx="48" cy="48" r="42" fill="{fill}" stroke="#ffffff" stroke-width="6"/>'
        f'<text x="48" y="48" text-anchor="middle" dominant-baseline="central" font-family="Arial, Helvetica, sans-serif" '
        f'font-size="46" font-weight="700" fill="#ffffff">{text}</text></svg>'
    )
    url = "data:image/svg+xml;base64," + base64.b64encode(svg.encode()).decode()
    return {"url": url, "width": 96, "height": 96, "anchorY": 48}


# ---- left panel: request + scoring --------------------------------------------


def request_form(base: CareRequest, known_ids: set[str]) -> tuple[CareRequest, Weights, float | None]:
    """Everything inside one form, so results only change when 'Find Top Matches' is pressed."""
    k = base.request_id
    base_days = sorted({s.split("-")[0] for s in base.slots}, key=DAYS.index)
    base_shift = next(iter(base.slots)).split("-")[1]
    cities = list(CITIES)
    defaults = Weights()

    with st.form(f"form_{k}", border=False):
        skill = st.selectbox("Care needed", SKILLS, index=SKILLS.index(base.required_skill), format_func=label)
        language = st.selectbox("Language the family needs", LANGUAGES, index=LANGUAGES.index(base.language))
        city = st.selectbox("Patient's city", cities, index=cities.index(base.city) if base.city in cities else 0)
        days = st.pills("Days", DAYS, selection_mode="multi", default=base_days)
        shift = st.segmented_control("Shift", SHIFTS, default=base_shift, format_func=str.capitalize) or base_shift
        previous = st.text_input("Nurse who looked after this patient before (optional)",
                                 value=", ".join(base.previous_nurse_ids), placeholder="e.g. N042")

        with st.expander("Scoring options"):
            use_learned = st.checkbox("Use weights learned from simulated feedback")
            d = st.slider("Distance", 0.0, 1.0, defaults.distance, 0.05)
            r = st.slider("Rating", 0.0, 1.0, defaults.rating, 0.05)
            e = st.slider("Experience", 0.0, 1.0, defaults.experience, 0.05)
            c = st.slider("Continuity (same nurse as before)", 0.0, 1.0, defaults.continuity, 0.05)
            limit = st.checkbox("Only nurses within a travel radius",
                                help="Off: distance only affects the score. On: nurses beyond the radius are left out.")
            radius = st.slider("Radius (km)", 5, 100, int(SUGGESTED_RADIUS_KM))
        st.form_submit_button("Find Top Matches", type="primary", width="stretch")

    previous_ids = tuple(p.strip().upper() for p in previous.split(",") if p.strip())
    unknown = [p for p in previous_ids if p not in known_ids]
    if unknown:
        st.warning(f"No nurse with id {', '.join(unknown)}.")
    if not days:
        st.warning("Pick at least one day.")
        st.stop()

    if use_learned:
        weights = get_learning().learned
    else:
        if d + r + e + c == 0:
            st.error("At least one weight must be above zero.")
            st.stop()
        weights = Weights(distance=d, rating=r, experience=e, continuity=c)

    lat, lon = (base.lat, base.lon) if city == base.city else CITIES[city]
    request = CareRequest(
        request_id=base.request_id, patient_id=base.patient_id, patient_name=base.patient_name,
        required_skill=skill, language=language, city=city, lat=lat, lon=lon,
        slots=frozenset(f"{x}-{shift}" for x in days), previous_nurse_ids=previous_ids,
    )
    return request, weights, float(radius) if limit else None


# ---- results ------------------------------------------------------------------


def kpi_html(items: list[tuple[str, str, str]]) -> str:
    cells = "".join(
        f'<div class="nm-kpi"><div class="ic">{icon_img(p)}</div>'
        f'<div style="min-width:0"><div class="lab">{escape(lab)}</div><div class="val">{escape(val)}</div></div></div>'
        for lab, val, p in items
    )
    return f'<div class="nm-kpis">{cells}</div>'


def previous_nurse_kpi(request: CareRequest, matches: list[Match], eligible_ids: set[str]) -> tuple[str, str]:
    if not request.previous_nurse_ids:
        return "Previous nurse", "None"
    prev = request.previous_nurse_ids[0]
    ranks = {m.nurse.nurse_id: m.rank for m in matches}
    if prev in ranks:
        return f"Previous nurse ({prev})", f"Ranked #{ranks[prev]}"
    return f"Previous nurse ({prev})", "Not in top 5" if prev in eligible_ids else "Not eligible"


def funnel_html(funnel: list[tuple[str, int]]) -> str:
    top = funnel[0][1] or 1
    return "".join(
        f'<div class="nm-fn"><div class="nm-fn-top"><span>{escape(step)}</span><b>{n}</b></div>'
        f'<div class="nm-fn-track"><div class="nm-fn-fill" style="width:{n / top * 100:.1f}%"></div></div></div>'
        for step, n in funnel
    )


def breakdown_html(m: Match, weights: Weights) -> str:
    total = weights.total
    rows = "".join(
        f"<tr><td>{FACTOR_NAMES[k]}</td><td>{weights.as_dict()[k] / total:.2f}</td>"
        f"<td>{m.factors[k]:.2f}</td><td>{m.contributions[k]:.2f}</td></tr>"
        for k in FACTOR_NAMES
    )
    return (
        '<table class="nm-bt"><tr><th>Factor</th><th>Weight</th><th>Value</th><th>Contrib.</th></tr>'
        f'{rows}<tr class="total"><td colspan="3">Total Score</td><td>{m.score:.2f}</td></tr></table>'
    )


def map_view(points: pd.DataFrame, width_px: int, height_px: int) -> pdk.ViewState:
    """Centre and zoom so every point fits the panel, with some margin (deck.gl world = 512 * 2**z px)."""
    lat, lon = points["lat"], points["lon"]
    span_lon = max(lon.max() - lon.min(), 0.005) * 1.9
    span_lat = max(lat.max() - lat.min(), 0.005) * 1.9
    centre_lat = float((lat.max() + lat.min()) / 2)
    zoom_lon = math.log2(width_px * 360 / (512 * span_lon))
    zoom_lat = math.log2(height_px * 360 * math.cos(math.radians(centre_lat)) / (512 * span_lat))
    zoom = max(3.0, min(13.0, zoom_lon, zoom_lat))
    return pdk.ViewState(latitude=centre_lat, longitude=float((lon.max() + lon.min()) / 2), zoom=zoom)


def build_map(request: CareRequest, matches: list[Match], radius_km: float | None,
              width_px: int = 560, height_px: int = 380) -> pdk.Deck:
    patient = pd.DataFrame([{
        "lat": request.lat, "lon": request.lon, "name": request.patient_name or "Patient",
        "detail": "Patient's home", "icon": pin_icon("", "#DC2626"),
    }])
    nurses = pd.DataFrame([{
        "lat": m.nurse.lat, "lon": m.nurse.lon, "name": m.nurse.name,
        "detail": f"#{m.rank} · {m.distance_km:.1f} km · score {m.score:.2f}",
        "from_lat": request.lat, "from_lon": request.lon, "icon": pin_icon(str(m.rank), "#0F5F5A"),
    } for m in matches])

    layers = [
        pdk.Layer("LineLayer", nurses, get_source_position="[from_lon, from_lat]",
                  get_target_position="[lon, lat]", get_color=[15, 95, 90, 130], get_width=1.5),
        pdk.Layer("IconLayer", nurses, get_icon="icon", get_position="[lon, lat]", get_size=36,
                  size_scale=1, pickable=True),
        pdk.Layer("IconLayer", patient, get_icon="icon", get_position="[lon, lat]", get_size=30,
                  size_scale=1, pickable=True),
    ]
    if radius_km is not None:
        ring = patient[["lat", "lon"]].assign(radius_m=radius_km * 1000)
        layers.insert(0, pdk.Layer("ScatterplotLayer", ring, get_position="[lon, lat]", get_radius="radius_m",
                                   filled=False, stroked=True, get_line_color=[15, 95, 90, 200],
                                   line_width_min_pixels=2))

    points = pd.concat([patient[["lon", "lat"]], nurses[["lon", "lat"]]])
    if radius_km is not None:
        dlat = radius_km / 111.195
        dlon = dlat / math.cos(math.radians(request.lat))
        points = pd.concat([points, pd.DataFrame({
            "lon": [request.lon - dlon, request.lon + dlon], "lat": [request.lat - dlat, request.lat + dlat]})])
    return pdk.Deck(layers=layers, initial_view_state=map_view(points, width_px, height_px), map_style="light",
                    tooltip={"html": "<b>{name}</b><br/>{detail}"})


@st.dialog("Where they are", width="large")
def map_dialog(request: CareRequest, matches: list[Match], radius_km: float | None) -> None:
    with st.container(key="mapwrap_big"):
        st.pydeck_chart(build_map(request, matches, radius_km, width_px=1150, height_px=560), height=560)
    st.markdown("**Red dot:** patient. **Numbers:** the ranks on the left.")


def select_nurse(nurse_id: str) -> None:
    """Button callback: remember the nurse and ask for the details pop-up on this run."""
    st.session_state["selected_nurse"] = nurse_id
    st.session_state["open_details"] = nurse_id


def nurse_card(m: Match, request: CareRequest) -> None:
    n = m.nurse
    chips = "".join(chip(label(x).capitalize()) for x in sorted(n.skills))
    pill = '<span class="nm-pill">Previously served</span>' if n.nurse_id in request.previous_nurse_ids else ""
    warn = chip(f"Based in {n.city}", "warn") if n.city != request.city else ""
    with st.container(border=True):
        c_rank, c_main, c_score = st.columns([0.4, 4.0, 1.5], vertical_alignment="center", gap="small")
        c_rank.html(f'<div class="nm-rank{" top" if m.rank == 1 else ""}">{m.rank}</div>')
        c_main.html(
            f'<div class="nm-head"><span class="nm-name">{escape(n.name)}</span>'
            f'<span class="nm-id">{escape(n.nurse_id)}</span>{pill}</div>'
            f'<div class="nm-line">{m.distance_km:.1f} km away &bull; {n.years_experience} yrs exp &bull; '
            f'Speaks {escape(", ".join(sorted(n.languages)))}</div>'
            f'<div>{chips}{warn}</div>'
        )
        with c_score:
            st.html(f'<div class="nm-score"><div class="lab">Match Score</div><div class="num">{m.score:.2f}</div></div>')
            st.button("View Details", key=f"view_{n.nurse_id}", on_click=select_nurse, args=(n.nurse_id,),
                      width="stretch")


@st.dialog("Why this nurse", width="large")
def details_dialog(m: Match, weights: Weights) -> None:
    st.html(f'<div class="nm-panel-title" style="font-size:1.4rem">#{m.rank} {escape(m.nurse.name)} '
            f'<span class="nm-id">{escape(m.nurse.nurse_id)}</span></div>')
    st.html(f'<div class="nm-expl">{escape(m.explanation)}</div>')
    st.html(breakdown_html(m, weights))


def show_results(request: CareRequest, weights: Weights, radius_km: float | None, kpi_box, mid, right) -> None:
    nurses, _ = get_data()
    result = match(request, nurses, weights=weights, max_distance_km=radius_km)
    eligible_ids = {n.nurse_id for n, _ in eligible_nurses(request, nurses, radius_km)}
    matches = result.matches

    prev_label, prev_value = previous_nurse_kpi(request, matches, eligible_ids)
    kpis = [
        ("Eligible nurses", f"{result.eligible_count} of {len(nurses)}",
         '<circle cx="9" cy="8" r="3.2"/><path d="M3 20c0-3.5 2.7-6 6-6s6 2.5 6 6"/><path d="M16 5.5a3 3 0 0 1 0 5.5M18 14.5c2 .7 3.3 2.7 3.3 5"/>'),
        ("Best score", f"{matches[0].score:.2f}" if matches else "n/a",
         '<path d="m12 3 2.7 5.6 6.1.9-4.4 4.3 1 6.1L12 17l-5.4 2.9 1-6.1-4.4-4.3 6.1-.9z"/>'),
        ("Closest in top 5", f"{min(m.distance_km for m in matches):.1f} km" if matches else "n/a",
         '<path d="M12 21s-7-6.2-7-11.5a7 7 0 0 1 14 0C19 14.800 12 21 12 21z"/><circle cx="12" cy="9.5" r="2.5"/>'),
        (prev_label, prev_value, '<circle cx="12" cy="8" r="4"/><path d="M4 21c0-4.4 3.6-7 8-7s8 2.6 8 7"/>'),
    ]
    kpi_box.html(kpi_html(kpis))
    for note in result.notes:
        if matches or not note.startswith("No nurse meets"):
            kpi_box.warning(note)

    if not matches:
        mid.html('<div class="nm-empty"><b>No nurse meets every requirement.</b> The bars on the right show '
                 "which rule removed the most nurses. Try other days, a different shift or a wider radius.</div>")
        with right:
            st.html(f'<div class="nm-panel-title">How the list was narrowed down</div>{funnel_html(result.funnel)}')
        return

    by_id = {m.nurse.nurse_id: m for m in matches}
    selected = by_id.get(st.session_state.get("selected_nurse")) or matches[0]

    with mid:
        st.html(f'<div class="nm-panel-title" style="font-size:1.5rem">Top {len(matches)} Nurses</div>')
        for m in matches:
            nurse_card(m, request)

    with right:
        with st.container(key="mapbox"):
            h1, h2 = st.columns([1.5, 1.3], vertical_alignment="center")
            h1.html('<div class="nm-panel-title" style="margin:0;font-size:1.3rem">Where they are</div>')
            if h2.button("Expand map", key="expand_map", width="stretch"):
                map_dialog(request, matches, radius_km)
            with st.container(key="mapwrap"):
                st.pydeck_chart(build_map(request, matches, radius_km, height_px=340), height=340)
            st.markdown("**Red dot:** patient. **Numbers:** the ranks on the left.")
        with st.container(key="breakbox"):
            tab_why, tab_funnel = st.tabs(["Why this nurse", "How the list was narrowed"])
            with tab_why:
                st.html(f'<div class="nm-expl"><b>{escape(selected.nurse.name)}.</b> {escape(selected.explanation)}</div>'
                        + breakdown_html(selected, weights))
            with tab_funnel:
                st.html(funnel_html(result.funnel))

    # View Details was clicked on this run: show the pop-up as well, so the click is never silent.
    opened = st.session_state.pop("open_details", None)
    if opened in by_id:
        details_dialog(by_id[opened], weights)


def find_page() -> None:
    nurses, requests = get_data()
    by_id = {r.request_id: r for r in requests}
    known_ids = {n.nurse_id for n in nurses}

    left, rest = st.columns([1.15, 4.0], gap="medium")
    with left:
        with st.container(key="left", border=False):
            with st.container(border=True):
                st.html('<div class="nm-panel-title" style="font-size:1.3rem">Care Request</div>')
                choice = st.selectbox("Sample request", list(by_id), format_func=lambda i: request_label(by_id[i]))
                request, weights, radius_km = request_form(by_id[choice], known_ids)
    with rest:
        kpi_box = st.container()
        mid_col, right_col = st.columns([2.1, 1.5], gap="medium")
        with mid_col:
            mid_box = st.container(key="mid")
        with right_col:
            right_box = st.container(key="right")
    show_results(request, weights, radius_km, kpi_box, mid_box, right_box)


# ---- other pages ------------------------------------------------------------


def fairness_page() -> None:
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


def learning_page() -> None:
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


def sync_from_top() -> None:
    """The top tabs are the only navigation. Clicking the active tab again must not deselect it."""
    value = st.session_state.get("toptabs")
    if value:
        st.session_state["page"] = value
    else:
        st.session_state["toptabs"] = st.session_state.get("page", PAGES[0])


def main() -> None:
    get_data()
    st.html(CSS)
    st.session_state.setdefault("page", PAGES[0])
    page = st.session_state["page"]
    st.session_state.setdefault("toptabs", page)

    with st.container(key="hero"):
        head_l, head_r = st.columns([2.2, 1.5], vertical_alignment="center")
        head_l.html('<div class="nm-title"><h1>Nurse Match Engine</h1>'
                    "<p>Find the best nurse for your care request, with complete transparency.</p></div>")
        with head_r:
            with st.container(key="toptabs_box"):
                st.segmented_control("Pages", PAGES, key="toptabs", label_visibility="collapsed",
                                     on_change=sync_from_top, width="stretch")
    if page == "Find a Nurse":
        find_page()
    else:
        with st.container(key="fullpage"):
            {"Fairness Analytics": fairness_page, "Learned Weights": learning_page}[page]()


main()

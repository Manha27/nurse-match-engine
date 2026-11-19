"""Demo UI for coordinators.

    streamlit run app/streamlit_app.py

The app calls the matching code directly rather than going through the API, so
one command is enough to try it. The API in api/main.py wraps the same function.

Layout: a header with the page tabs, then three panels (request, ranked nurses, map and
reasons) sized to the browser window, so the main screen fits without page scrolling. Each
panel scrolls on its own. Below 1100px wide everything stacks and the page scrolls normally.
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

st.set_page_config(page_title="Nurse match engine", layout="wide", initial_sidebar_state="collapsed")

FACTOR_COLORS = {
    "distance": "#0F5C58",
    "rating": "#E5A93D",
    "experience": "#6C8EBF",
    "continuity": "#5CB88A",
}
FACTOR_NAMES = {
    "distance": "Distance",
    "rating": "Rating",
    "experience": "Experience",
    "continuity": "Continuity",
}
PAGES = ["Find a Nurse", "Fairness Analytics", "Learned Weights"]
SUGGESTION_NOTE = ("Recommendations are suggestions. Final assignment needs the nurse's confirmation "
                   "that she is available and accepts.")

CSS = """
<style>
:root {
  --ink: #10263B; --muted: #5E7384; --line: #E1E9EE; --surface: #FFFFFF;
  --teal: #0F5C58; --teal-mid: #137A73; --teal-soft: #E2F4F0; --mint: #D6F1EA; --mint-line: #9FD6CB;
  --green-soft: #DDF3E4; --amber-soft: #FFF4DF;
  --shadow: 0 1px 2px rgba(16,38,59,.05), 0 8px 22px rgba(16,38,59,.05);
  --font: Inter, "Segoe UI", system-ui, -apple-system, "Helvetica Neue", Arial, sans-serif;
}
.stApp, .stApp div, .stApp p, .stApp label, .stApp button, .stApp input, .stApp textarea, .stApp li,
.stApp td, .stApp th, .stApp a { font-family: var(--font); }
.stApp { color: var(--ink); background: linear-gradient(135deg, #E8F1F3 0%, #F4F8FA 45%, #F8FBFC 100%); }
.block-container { max-width: 100%; padding: .9rem 1.5rem .5rem; }
#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"] { display: none; }
[data-testid="stHeader"] { background: transparent; height: 0; }
[data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"], [data-testid="collapsedControl"] { display: none; }
html { font-size: clamp(13px, 0.5vw + 7px, 17px); }

/* header: logo, title, page tabs */
.nm-brand { display: flex; align-items: center; gap: 14px; }
.nm-logo { flex: 0 0 50px; height: 50px; border-radius: 15px; background-color: var(--teal);
  box-shadow: 0 6px 14px rgba(15,92,88,.28); background-repeat: no-repeat; background-position: center; background-size: 28px;
  background-image: url("data:image/svg+xml;utf8,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='white' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M12 21s-7-4.4-9.3-9A5.4 5.4 0 0 1 12 6.2 5.4 5.4 0 0 1 21.3 12C19 16.6 12 21 12 21z'/%3E%3Cpath d='M12 9.5v5M9.5 12h5'/%3E%3C/svg%3E"); }
.nm-brand .t { font-size: 1.95rem; font-weight: 800; letter-spacing: -.035em; color: var(--ink); line-height: 1.1; }
.nm-brand .t em { font-style: normal; color: var(--teal-mid); }
.nm-brand .s { color: var(--muted); font-size: .95rem; margin-top: 3px; }
.nm-pill { display: inline-block; vertical-align: middle; margin-left: 10px; font-size: .68rem; font-weight: 700; letter-spacing: .02em;
  color: var(--teal); background: var(--mint); border: 1px solid var(--mint-line); border-radius: 999px; padding: 2px 9px; }

.st-key-nav { display: flex; justify-content: flex-end; }
[data-testid="stButtonGroup"] button { border-radius: 999px !important; }
.st-key-nav [data-testid="stButtonGroup"] { background: rgba(255,255,255,.7); border: 1px solid var(--line); border-radius: 999px; padding: 4px; gap: 4px; box-shadow: var(--shadow); }
.st-key-nav button { border: 0 !important; background: transparent; padding: 8px 20px; }
.st-key-nav button p { font-size: .92rem; font-weight: 600; color: #45596B; }


/* page header on the other pages */
.nm-title { font-size: 1.5rem; font-weight: 800; letter-spacing: -.03em; margin: 0; color: var(--ink); }
.nm-sub { color: var(--muted); font-size: .92rem; margin: 2px 0 10px; }

/* panels sized to the window */
.st-key-ctrl, .st-key-side, .st-key-page { height: calc(100vh - 8rem) !important; max-height: calc(100vh - 8rem) !important; overflow-y: auto !important; padding-right: 4px; }
.st-key-ranked { height: calc(100vh - 13rem) !important; max-height: calc(100vh - 13rem) !important; overflow-y: auto !important; padding-right: 4px; }
@media (min-width: 1101px) { [data-testid="stMain"] { overflow: clip; } }
.st-key-ctrl { background: #fff; border: 1px solid var(--line); border-radius: 16px; padding: 16px 18px; box-shadow: var(--shadow); }
.st-key-ctrl [data-testid="stVerticalBlock"] { gap: .55rem; }
.st-key-ranked [data-testid="stVerticalBlock"], .st-key-side [data-testid="stVerticalBlock"] { gap: .65rem; }
[data-testid="stMain"] ::-webkit-scrollbar { width: 6px; }
[data-testid="stMain"] ::-webkit-scrollbar-thumb { background: #C3D3D9; border-radius: 6px; }

/* widgets */
.stSelectbox label, .stTextInput label, .stSlider label, .stCheckbox label, .stSegmentedControl label,
.stPills label, .stToggle label { color: #2B4256; font-size: .82rem; font-weight: 600; }
[data-baseweb="select"] > div, .stTextInput input { background: #fff; border-color: #D5E0E6; border-radius: 10px; }
.stButton button, [data-testid="stFormSubmitButton"] button { border-radius: 10px; font-weight: 700; }
.stButton button[kind="primary"], [data-testid="stFormSubmitButton"] button[kind="primaryFormSubmit"] {
  background: var(--teal); border-color: var(--teal); color: #fff; box-shadow: 0 4px 10px rgba(15,92,88,.22); }
.stButton button[kind="primary"]:hover, [data-testid="stFormSubmitButton"] button[kind="primaryFormSubmit"]:hover {
  background: #0B4744; border-color: #0B4744; }
.stButton button[kind="secondary"] { background: #fff; border: 1px solid var(--mint-line); color: var(--teal); }
button[data-variant="pills"] { border-radius: 999px !important; background: #fff; border: 1px solid #D5E0E6; }
button[data-variant="pills"][data-selected="true"] { background: var(--mint) !important; border-color: var(--teal-mid) !important; }
button[data-variant="pills"][data-selected="true"] * { color: var(--teal) !important; font-weight: 700; }
button[data-variant="segmented_control"][aria-checked="true"] { background: var(--teal) !important; border-color: var(--teal) !important; }
button[data-variant="segmented_control"][aria-checked="true"] * { color: #fff !important; }
[data-testid="stExpander"] { border: 1px solid var(--line); border-radius: 12px; background: #F6FAFA; }
[data-testid="stDataFrame"] { border: 1px solid var(--line); border-radius: 12px; overflow: hidden; }
[data-testid="stMetric"] { background: #fff; border: 1px solid var(--line); border-radius: 14px; padding: 14px 16px; box-shadow: var(--shadow); }
[data-testid="stMetricLabel"] { color: var(--muted); font-size: .82rem; font-weight: 600; }
[data-testid="stMetricValue"] { font-size: 1.6rem; font-weight: 800; color: var(--teal); }
.section-title { font-size: 1.12rem; font-weight: 800; letter-spacing: -.01em; margin: 2px 0 2px; color: var(--ink); }
.section-note { color: var(--muted); font-size: .84rem; line-height: 1.5; margin: 0 0 8px; }

/* summary and KPIs */
.nm-summary { background: #fff; border: 1px solid var(--line); border-radius: 14px; padding: 12px 16px 7px; box-shadow: var(--shadow); }
.nm-summary .who { font-size: 1.1rem; font-weight: 800; margin-bottom: 7px; }
.nm-kpis { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; margin: 4px 0 6px; }
.nm-kpi { background: #fff; border: 1px solid var(--line); border-radius: 14px; padding: 12px 14px; box-shadow: var(--shadow);
  display: flex; align-items: center; gap: 12px; min-width: 0; }
.nm-kpi .ic { flex: 0 0 40px; height: 40px; border-radius: 50%; background-color: var(--mint);
  background-repeat: no-repeat; background-position: center; background-size: 20px; }
.nm-kpi .ic.group { background-image: url("data:image/svg+xml;utf8,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'%3E%3Cpath fill='%230F5C58' d='M16 11c1.66 0 2.99-1.34 2.99-3S17.66 5 16 5s-3 1.34-3 3 1.34 3 3 3zm-8 0c1.66 0 2.99-1.34 2.99-3S9.66 5 8 5 5 6.34 5 8s1.34 3 3 3zm0 2c-2.33 0-7 1.17-7 3.5V19h14v-2.5c0-2.33-4.67-3.5-7-3.5zm8 0c-.29 0-.62.02-.97.05 1.16.84 1.97 1.97 1.97 3.45V19h6v-2.5c0-2.33-4.67-3.5-7-3.5z'/%3E%3C/svg%3E"); }
.nm-kpi .ic.star { background-image: url("data:image/svg+xml;utf8,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'%3E%3Cpath fill='%230F5C58' d='M12 17.27L18.18 21l-1.64-7.03L22 9.24l-7.19-.61L12 2 9.19 8.63 2 9.24l5.46 4.73L5.82 21z'/%3E%3C/svg%3E"); }
.nm-kpi .ic.place { background-image: url("data:image/svg+xml;utf8,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'%3E%3Cpath fill='%230F5C58' d='M12 2C8.13 2 5 5.13 5 9c0 5.25 7 13 7 13s7-7.75 7-13c0-3.87-3.13-7-7-7zm0 9.5c-1.38 0-2.5-1.12-2.5-2.5s1.12-2.5 2.5-2.5 2.5 1.12 2.5 2.5-1.12 2.5-2.5 2.5z'/%3E%3C/svg%3E"); }
.nm-kpi .ic.person { background-image: url("data:image/svg+xml;utf8,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'%3E%3Cpath fill='%230F5C58' d='M12 12c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm0 2c-2.67 0-8 1.34-8 4v2h16v-2c0-2.66-5.33-4-8-4z'/%3E%3C/svg%3E"); }
.nm-kpi .lb { font-size: .78rem; color: var(--muted); font-weight: 600; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.nm-kpi .vl { font-size: 1.28rem; font-weight: 800; line-height: 1.2; white-space: nowrap; color: var(--ink); }

/* chips */
.nm-chip { display: inline-block; background: var(--teal-soft); color: #145D57; border: 1px solid #CFE9E4; border-radius: 999px;
  padding: 3px 10px; margin: 0 5px 5px 0; font-size: .78rem; font-weight: 600; }
.nm-chip.ok { background: var(--green-soft); color: #1E6B45; border-color: #C5E6D1; }
.nm-chip.warn { background: var(--amber-soft); color: #8A5A13; border-color: #F4E1BC; }
.nm-chip.quiet { background: #F1F5F7; color: #4F6272; border-color: #E4EBEF; }

/* nurse cards */
.nm-legend { font-size: .8rem; color: var(--muted); margin: 0 0 2px; }
.nm-legend i { display: inline-block; width: 9px; height: 9px; border-radius: 3px; margin: 0 5px 0 12px; vertical-align: -1px; }
[class*="st-key-card_"] { background: #fff; border: 1px solid var(--line); border-radius: 14px; padding: 14px 16px; box-shadow: var(--shadow); }
[class*="st-key-card_"][class*="_sel"] { border-color: var(--teal-mid); box-shadow: 0 0 0 2px rgba(19,122,115,.18), var(--shadow); }
[class*="st-key-card_"] [data-testid="stVerticalBlock"] { gap: .3rem; }
[class*="st-key-card_"] [data-testid="stColumn"]:last-child { border-left: 1px solid var(--line); padding-left: 16px; display: flex; flex-direction: column; justify-content: center; }
.nm-cardbody { display: flex; gap: 14px; }
.nm-rank { flex: 0 0 34px; height: 34px; border-radius: 50%; background: var(--mint); color: var(--teal);
  font-size: .95rem; font-weight: 800; display: flex; align-items: center; justify-content: center; }
[class*="_sel"] .nm-rank { background: var(--teal); color: #fff; }
.nm-main { flex: 1; min-width: 0; }
.nm-head { display: flex; align-items: center; gap: 9px; flex-wrap: wrap; margin-bottom: 5px; }
.nm-name { font-size: 1.12rem; font-weight: 800; letter-spacing: -.01em; }
.nm-id { color: #7A8D9B; font-size: .82rem; font-weight: 500; }
.nm-badge { background: var(--green-soft); color: #1E6B45; border-radius: 999px; padding: 2px 10px; font-size: .74rem; font-weight: 700; }
.nm-l1 { font-size: .9rem; color: #2B4256; margin-bottom: 3px; }
.nm-l1 b { color: var(--ink); font-weight: 700; }
.nm-l1 .sep { color: #A8B7C1; margin: 0 6px; }
.nm-l1 .warn { color: #9A5B0C; font-weight: 700; }
.nm-l2 { font-size: .86rem; color: #50657A; margin-bottom: 2px; }
.nm-track { display: flex; height: 6px; background: #EAF0F3; border-radius: 99px; overflow: hidden; margin: 8px 0 0; }
.nm-track span { display: block; height: 100%; }
.nm-expl { color: #2B4256; font-size: .92rem; line-height: 1.5; margin-bottom: 8px; }
.nm-avail { color: #6A7E8C; font-size: .8rem; margin: 0 0 5px; }
.nm-scorebox { text-align: center; margin-bottom: 4px; }
.nm-scorebox .k { font-size: .78rem; color: var(--muted); font-weight: 600; }
.nm-scorebox .v { font-size: 1.9rem; font-weight: 800; color: var(--teal); line-height: 1.1; letter-spacing: -.02em; }

/* funnel and tables */
.nm-fn { margin-bottom: 10px; }
.nm-fn-top { display: flex; justify-content: space-between; color: #46596B; font-size: .86rem; margin-bottom: 4px; }
.nm-fn-top b { color: var(--ink); }
.nm-fn-track { height: 7px; background: #E6EEF1; border-radius: 99px; overflow: hidden; }
.nm-fn-fill { height: 100%; background: var(--teal-mid); border-radius: 99px; }
.nm-box { background: #fff; border: 1px solid var(--line); border-radius: 14px; padding: 14px 16px; box-shadow: var(--shadow); }
.nm-empty { background: #FFF8EC; border: 1px solid #F1DFC2; border-radius: 14px; padding: 16px 18px; color: #6E4D1C; font-size: .94rem; line-height: 1.55; }
.nm-table { width: 100%; border-collapse: collapse; font-size: .9rem; }
.nm-table th { text-align: right; color: var(--muted); font-weight: 600; padding: 7px 8px; border-bottom: 1px solid var(--line); }
.nm-table td { text-align: right; padding: 7px 8px; border-bottom: 1px solid #EDF2F5; }
.nm-table th:first-child, .nm-table td:first-child { text-align: left; }
.nm-table tr.total td { font-weight: 800; background: #E3F5DC; border-bottom: none; }
.nm-table tr.total td:first-child { border-radius: 8px 0 0 8px; }
.nm-table tr.total td:last-child { border-radius: 0 8px 8px 0; }

/* map */
.nm-maplegend { font-size: .8rem; color: var(--muted); line-height: 1.5; }
.nm-maplegend .dot { display: inline-block; width: 9px; height: 9px; border-radius: 50%; margin-right: 4px; }
.st-key-mapbox [data-testid="stElementToolbar"] { display: none; }
.st-key-mapbox { background: #fff; border: 1px solid var(--line); border-radius: 16px; padding: 10px 10px 6px; box-shadow: var(--shadow); }
.st-key-form_wrap [data-testid="stForm"] { padding: 0; }

@media (max-width: 1100px) {
  .block-container { padding: .8rem .8rem 2rem; }
  .st-key-ctrl, .st-key-side, .st-key-ranked, .st-key-page { height: auto !important; max-height: none !important; overflow: visible !important; }
  .nm-brand .t { font-size: 1.45rem; }
  .nm-brand .s { font-size: .82rem; }
  .nm-logo { flex-basis: 42px; height: 42px; }
  .st-key-nav [data-testid="stButtonGroup"] { flex-wrap: wrap; }
  .st-key-nav button { padding: 6px 13px; }
  [data-testid="stHorizontalBlock"] { flex-wrap: wrap; }
  [data-testid="stColumn"] { min-width: 100% !important; }
  [class*="st-key-card_"] [data-testid="stColumn"]:last-child { border-left: 0; padding-left: 0; border-top: 1px solid var(--line); padding-top: 10px; }
  .nm-kpis { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .st-key-nav { justify-content: flex-start; }
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
    """e.g. 'days: Weekdays · nights: Every day' from a nurse's weekly slots."""
    parts = []
    for shift in SHIFTS:
        days = [d for d in DAYS if f"{d}-{shift}" in slots]
        if days:
            parts.append(f"{shift}s: {describe_days(days)}")
    return " · ".join(parts) or "no weekly slots"


def chip(text: str, kind: str = "") -> str:
    return f'<span class="nm-chip {kind}">{escape(text)}</span>'


# ---- request and scoring controls -------------------------------------------


# ---- results ----------------------------------------------------------------


def summary_html(request: CareRequest) -> str:
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


def kpi_html(items: list[tuple[str, str, str]]) -> str:
    cells = "".join(
        f'<div class="nm-kpi"><div class="ic {icon}"></div>'
        f'<div><div class="lb">{escape(lb)}</div><div class="vl">{escape(vl)}</div></div></div>'
        for icon, lb, vl in items
    )
    return f'<div class="nm-kpis">{cells}</div>'


def card_body_html(m: Match, request: CareRequest) -> str:
    n = m.nurse
    segments = "".join(
        f'<span style="width:{m.contributions[k] * 100:.2f}%;background:{FACTOR_COLORS[k]}" '
        f'title="{FACTOR_NAMES[k]} {m.contributions[k]:.3f}"></span>'
        for k in FACTOR_COLORS
    )
    sep = '<span class="sep">•</span>'
    facts = [f"{m.distance_km:.1f} km away", f"{n.years_experience} yrs exp"]
    facts.append(f"★ {n.rating:.1f} ({n.review_count})" if n.review_count else "No reviews yet")
    line1 = "<b>Free for all requested slots</b>" + "".join(sep + escape(f) for f in facts)
    if n.city != request.city:
        line1 += sep + f'<span class="warn">Based in {escape(n.city)}</span>'
    badge = '<span class="nm-badge">Previously served</span>' if n.nurse_id in request.previous_nurse_ids else ""
    return (
        '<div class="nm-cardbody">'
        f'<div class="nm-rank">{m.rank}</div><div class="nm-main">'
        f'<div class="nm-head"><span class="nm-name">{escape(n.name)}</span>'
        f'<span class="nm-id">{escape(n.nurse_id)}</span>{badge}</div>'
        f'<div class="nm-l1">{line1}</div>'
        f'<div class="nm-l2">{escape(", ".join(label(x).capitalize() for x in sorted(n.skills)))}</div>'
        f'<div class="nm-l2">Speaks {escape(", ".join(sorted(n.languages)))}</div>'
        f'<div class="nm-track">{segments}</div>'
        "</div></div>"
    )


def legend_html() -> str:
    items = "".join(f'<i style="background:{c}"></i>{FACTOR_NAMES[k]}' for k, c in FACTOR_COLORS.items())
    return f'<div class="nm-legend">Score bar:{items}</div>'


def funnel_html(funnel: list[tuple[str, int]]) -> str:
    top = funnel[0][1] or 1
    rows = ""
    for i, (step, n) in enumerate(funnel):
        removed = f' <span style="color:#B45309;font-weight:500">(−{funnel[i - 1][1] - n})</span>' if i else ""
        rows += (
            f'<div class="nm-fn"><div class="nm-fn-top"><span>{escape(step)}</span><b>{n}{removed}</b></div>'
            f'<div class="nm-fn-track"><div class="nm-fn-fill" style="width:{n / top * 100:.1f}%"></div></div></div>'
        )
    return f'<div class="nm-box">{rows}</div>'


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


def factor_table_html(m: Match, weights: Weights) -> str:
    total_w = weights.total
    w = weights.as_dict()
    rows = "".join(
        f"<tr><td>{FACTOR_NAMES[k]}</td><td>{w[k] / total_w:.3f}</td><td>{m.factors[k]:.3f}</td>"
        f"<td>{m.contributions[k]:.3f}</td></tr>"
        for k in FACTOR_COLORS
    )
    return (
        '<table class="nm-table"><tr><th>Factor</th><th>Weight</th><th>Value (0 to 1)</th><th>Contribution</th></tr>'
        f'{rows}<tr class="total"><td>Total score</td><td></td><td></td><td>{m.score:.3f}</td></tr></table>'
    )


@st.dialog("Nurse details", width="large")
def details_dialog(m: Match, request: CareRequest, weights: Weights) -> None:
    n = m.nurse
    st.html(
        f'<div class="nm-head"><span class="nm-name" style="font-size:1.25rem">{escape(n.name)}</span>'
        f'<span class="nm-id">{escape(n.nurse_id)} · rank #{m.rank} · {escape(n.city)}</span></div>'
        f'<div class="nm-expl" style="font-size:.95rem">{escape(m.explanation)}</div>'
    )
    left, right = st.columns([1.2, 1], gap="large")
    with left:
        st.markdown("**How the score was built**")
        st.html(factor_table_html(m, weights))
        st.caption("Contribution = weight × value. The contributions add up to the score.")
    with right:
        st.markdown("**About this nurse**")
        st.html(
            chip(f"{m.distance_km:.1f} km straight-line from the patient")
            + chip(f"{n.years_experience} yrs experience")
            + (chip(f"Rated {n.rating:.1f} from {n.review_count} reviews") if n.review_count
               else chip("No reviews yet", "quiet"))
            + "".join(chip(label(s), "quiet") for s in sorted(n.skills))
            + chip("Speaks " + ", ".join(sorted(n.languages)), "quiet")
            + f'<div class="nm-avail" style="margin-top:6px">Weekly availability, '
              f"{escape(describe_availability(n.available_slots))}</div>"
        )
    st.info(SUGGESTION_NOTE)


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
              width_px: int = 400, height_px: int = 300, selected_id: str | None = None) -> pdk.Deck:
    patient = pd.DataFrame([{
        "lat": request.lat, "lon": request.lon,
        "name": request.patient_name or "Patient", "detail": "Patient's home",
    }])
    nurses = pd.DataFrame([{
        "lat": m.nurse.lat, "lon": m.nurse.lon, "label": str(m.rank), "name": m.nurse.name,
        "detail": f"#{m.rank} · {m.distance_km:.1f} km · score {m.score:.2f}",
        "from_lat": request.lat, "from_lon": request.lon,
    } for m in matches])

    # Markers that would overlap on screen are nudged apart (on the map only), based on the zoom.
    raw = pd.concat([patient[["lon", "lat"]], nurses[["lon", "lat"]]])
    deg_per_px = 360 / (512 * 2 ** map_view(raw, width_px, height_px).zoom)
    placed: list[tuple[float, float]] = []
    for i, row in nurses.iterrows():
        lon, lat = row["lon"], row["lat"]
        while any(abs(lon - x) < 40 * deg_per_px and abs(lat - y) < 40 * deg_per_px for x, y in placed):
            lon += 40 * deg_per_px
        placed.append((lon, lat))
        nurses.loc[i, "lon"] = lon

    selected_names = [m.nurse.name for m in matches if m.nurse.nurse_id == selected_id]
    layers = [
        pdk.Layer("LineLayer", nurses, get_source_position="[from_lon, from_lat]",
                  get_target_position="[lon, lat]", get_color=[148, 163, 184, 200], get_width=2),
        pdk.Layer("ScatterplotLayer", nurses, get_position="[lon, lat]", get_fill_color=[15, 92, 88, 255],
                  get_line_color=[255, 255, 255, 255], stroked=True, line_width_min_pixels=3,
                  radius_min_pixels=17, radius_max_pixels=17, pickable=True),
        # Two passes half a pixel apart give the default font a bolder, easier-to-read stroke.
        *[pdk.Layer("TextLayer", nurses, get_position="[lon, lat]", get_text="label", get_size=19,
                    get_color=[255, 255, 255, 255], get_pixel_offset=[dx, 0],
                    get_alignment_baseline="'center'", get_text_anchor="'middle'") for dx in (-0.6, 0.6)],
        pdk.Layer("ScatterplotLayer", nurses[nurses["name"].isin(selected_names)], get_position="[lon, lat]",
                  filled=False, stroked=True, get_line_color=[245, 158, 11, 255], line_width_min_pixels=3,
                  radius_min_pixels=23, radius_max_pixels=23),
        pdk.Layer("ScatterplotLayer", patient, get_position="[lon, lat]", get_fill_color=[220, 38, 38, 255],
                  get_line_color=[255, 255, 255, 255], stroked=True, line_width_min_pixels=2,
                  radius_min_pixels=11, radius_max_pixels=11, pickable=True),
    ]
    if radius_km is not None:
        ring = patient.assign(radius_m=radius_km * 1000)
        layers.insert(0, pdk.Layer("ScatterplotLayer", ring, get_position="[lon, lat]", get_radius="radius_m",
                                   filled=False, stroked=True, get_line_color=[19, 122, 115, 200],
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


MAP_LEGEND = (
    '<div class="nm-maplegend"><span class="dot" style="background:#DC2626"></span>Patient &nbsp; '
    '<span class="dot" style="background:#0F5C58"></span>Top 5 by rank &nbsp; Grey lines: straight-line '
    "distance, not driving routes. Locations are simulated; nurses at almost the same spot are nudged apart.</div>"
)


@st.dialog("Where they are", width="large")
def map_dialog(request: CareRequest, matches: list[Match], radius_km: float | None,
               selected_id: str | None = None) -> None:
    st.pydeck_chart(build_map(request, matches, radius_km, width_px=1150, height_px=600,
                              selected_id=selected_id), height=600)
    st.html(MAP_LEGEND)
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


def why_panel(m: Match, weights: Weights) -> None:
    n = m.nurse
    st.html(
        f'<div class="nm-box"><div class="nm-head"><span class="nm-name">{escape(n.name)}</span>'
        f'<span class="nm-id">{escape(n.nurse_id)} · rank #{m.rank} · {m.distance_km:.1f} km straight-line</span></div>'
        f'<div class="nm-expl">{escape(m.explanation)}</div>{factor_table_html(m, weights)}'
        '<div class="nm-avail" style="margin-top:8px">Contribution = weight × value. Contributions add up to the '
        "score. Distance is an estimate from coordinates, not travel distance.</div></div>"
    )


def show_results(request: CareRequest, weights: Weights, radius_km: float | None) -> None:
    nurses, _ = get_data()
    result = match(request, nurses, weights=weights, max_distance_km=radius_km)
    eligible_ids = {n.nurse_id for n, _ in eligible_nurses(request, nurses, radius_km)}

    ids = [m.nurse.nurse_id for m in result.matches]
    if st.session_state.get("sel") not in ids:
        st.session_state["sel"] = ids[0] if ids else None
    sel = st.session_state["sel"]
    selected = next((m for m in result.matches if m.nurse.nurse_id == sel), None)

    def choose(nurse_id: str) -> None:
        st.session_state["sel"] = nurse_id
        st.session_state["open_details"] = nurse_id

    prev_label, prev_value = previous_nurse_kpi(request, result.matches, eligible_ids)
    st.html(kpi_html([
        ("group", "Eligible nurses", f"{result.eligible_count} of {len(nurses)}"),
        ("star", "Best score", f"{result.matches[0].score:.2f}" if result.matches else "n/a"),
        ("place", "Closest in top 5",
         f"{min(m.distance_km for m in result.matches):.1f} km" if result.matches else "n/a"),
        ("person", prev_label, prev_value),
    ]))

    centre, side = st.columns([1.55, 1], gap="medium")
    with centre:
        with st.container(key="ranked"):
            st.html(summary_html(request))
            for note in result.notes:
                if result.matches or not note.startswith("No nurse meets"):  # the empty-state box says this
                    st.warning(note)
            if not result.matches:
                st.html('<div class="nm-empty"><b>No eligible nurses found.</b> No nurse meets every requirement. '
                        "Try expanding the travel radius, selecting other days or a different shift, or "
                        "checking whether another language is acceptable. The funnel on the right shows which "
                        "rule removed the most nurses.</div>")
            else:
                st.html(legend_html())
                for m in result.matches:
                    n = m.nurse
                    is_sel = n.nurse_id == sel
                    with st.container(key=f"card_{m.rank}{'_sel' if is_sel else ''}"):
                        body, score = st.columns([4.2, 1.4], gap="small")
                        body.html(card_body_html(m, request))
                        score.html(f'<div class="nm-scorebox"><div class="k">Match score</div>'
                                   f'<div class="v">{m.score:.2f}</div></div>')
                        score.button("View Details", key=f"det_{m.rank}", width="stretch", type="primary",
                                     on_click=choose, args=(n.nurse_id,))
                st.caption(SUGGESTION_NOTE)

    with side:
        with st.container(key="side"):
            if result.matches:
                with st.container(key="mapbox"):
                    st.html('<div class="section-title" style="padding:2px 4px">Where they are</div>')
                    st.pydeck_chart(build_map(request, result.matches, radius_km, selected_id=sel), height=270)
                    if st.button("Expand map", icon=":material/open_in_full:", width="stretch"):
                        map_dialog(request, result.matches, radius_km, sel)
                st.html(MAP_LEGEND)
            if selected:
                tab_why, tab_funnel = st.tabs(["Why this nurse", "Eligibility funnel"])
                with tab_why:
                    why_panel(selected, weights)
                with tab_funnel:
                    st.html(funnel_html(result.funnel))
            else:  # nothing matched: show only where the nurses were removed
                st.html('<div class="section-title">Where the nurses were removed</div>')
                st.html(funnel_html(result.funnel))

    opened = st.session_state.pop("open_details", None)
    if opened:
        m = next((x for x in result.matches if x.nurse.nurse_id == opened), None)
        if m:
            details_dialog(m, request, weights)


def request_form(base: CareRequest, known_ids: set[str]) -> tuple[CareRequest, Weights, float | None] | None:
    """Everything is applied when "Find Top Matches" is pressed. Fields reset when another sample is chosen."""
    k = base.request_id
    base_days = sorted({s.split("-")[0] for s in base.slots}, key=DAYS.index)
    base_shift = next(iter(base.slots)).split("-")[1]
    defaults = Weights()

    with st.form("request_form", border=False):
        skill = st.selectbox("Care needed", SKILLS, index=SKILLS.index(base.required_skill),
                             format_func=label, key=f"skill_{k}")
        language = st.selectbox("Language the family needs", LANGUAGES,
                                index=LANGUAGES.index(base.language), key=f"lang_{k}")
        city = st.selectbox("Patient's city", list(CITIES), index=list(CITIES).index(base.city), key=f"city_{k}",
                            help="Changing the city moves the patient to that city's centre (simulated location).")
        days = st.pills("Days", DAYS, selection_mode="multi", default=base_days, key=f"days_{k}")
        shift = st.segmented_control("Shift", SHIFTS, default=base_shift, key=f"shift_{k}") or base_shift
        previous = st.text_input("Nurse who looked after this patient before (optional)",
                                 value=", ".join(base.previous_nurse_ids), key=f"prev_{k}",
                                 placeholder="e.g. N042")
        with st.expander("Scoring options"):
            use_learned = st.checkbox("Use weights learned from simulated feedback", key="use_learned",
                                      help="Fitted on simulated past matches. See the 'Learned Weights' page. "
                                           "When ticked, the sliders below are ignored.")
            st.caption("Only the proportions matter. They are normalised.")
            d = st.slider("Distance", 0.0, 1.0, defaults.distance, 0.05, key="w_d")
            r = st.slider("Rating", 0.0, 1.0, defaults.rating, 0.05, key="w_r")
            e = st.slider("Experience", 0.0, 1.0, defaults.experience, 0.05, key="w_e")
            c = st.slider("Same nurse as before", 0.0, 1.0, defaults.continuity, 0.05, key="w_c")
            limit = st.toggle("Only nurses within a travel radius", key="limit",
                              help="Off: distance only affects the score, so a far-away nurse can still appear. "
                                   "On: nurses beyond the radius are left out entirely.")
            radius = st.slider("Radius (km)", 5, 100, int(SUGGESTED_RADIUS_KM), key="radius")
        st.form_submit_button("Find Top Matches", type="primary", width="stretch", icon=":material/search:")

    previous_ids = tuple(p.strip().upper() for p in previous.split(",") if p.strip())
    unknown = [p for p in previous_ids if p not in known_ids]
    if unknown:
        st.warning(f"No nurse with id {', '.join(unknown)}.")
    if not days:
        st.warning("Pick at least one day.")
        return None
    if use_learned:
        weights = get_learning().learned
    elif d + r + e + c == 0:
        st.error("At least one weight must be above zero.")
        return None
    else:
        weights = Weights(distance=d, rating=r, experience=e, continuity=c)

    request = CareRequest(
        request_id=base.request_id, patient_id=base.patient_id, patient_name=base.patient_name,
        required_skill=skill, language=language, city=city,
        lat=base.lat if city == base.city else CITIES[city][0],
        lon=base.lon if city == base.city else CITIES[city][1],
        slots=frozenset(f"{day}-{shift}" for day in days), previous_nurse_ids=previous_ids,
    )
    return request, weights, float(radius) if limit else None


def find_page() -> None:
    nurses, requests = get_data()
    by_id = {r.request_id: r for r in requests}
    known_ids = {n.nurse_id for n in nurses}

    controls, results = st.columns([0.95, 4.0], gap="medium")
    with controls:
        with st.container(key="ctrl"):
            st.html('<div class="section-title">Care request</div>')
            choice = st.selectbox("Sample request", list(by_id), format_func=lambda i: request_label(by_id[i]))
            chosen = request_form(by_id[choice], known_ids)
    with results:
        if chosen is None:
            st.info("Fix the highlighted field on the left, then press Find Top Matches.")
        else:
            show_results(*chosen)


# ---- other pages ------------------------------------------------------------


def fairness_page() -> None:
    st.html('<div class="nm-title">Fairness Analytics</div>'
            '<div class="nm-sub">Is the work spread across nurses, and does balancing it cost match quality?</div>')
    st.write(
        "Every sample request is run through the engine and we count who comes first. "
        "A good result is no nurse always first, and work spread at least as evenly as picking "
        "a random eligible nurse."
    )
    balance_weight = st.slider("Workload balancing weight", 0.0, 0.30, 0.10, 0.05,
                               help="How much a nurse's earlier first-picks count against them.")
    plain, balanced = get_fairness(balance_weight)

    a, b, c, d = st.columns(4)
    a.metric("Always-first nurses", len(plain.always_first))
    b.metric("Busiest nurse's share of first picks", f"{plain.max_top1_share:.0%}")
    c.metric("Gini of first picks", f"{plain.gini_top1:.2f}")
    d.metric("With balancing, Gini", f"{balanced.gini_top1:.2f}",
             delta=f"{balanced.gini_top1 - plain.gini_top1:+.2f}", delta_color="inverse")

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
        "assigned work. Balancing only reorders nurses who already pass every hard rule."
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
        st.bar_chart(chart, stack=False, color=["#0F5C58", "#E5A93D"])


def learning_page() -> None:
    st.html('<div class="nm-title">Learned Weights</div>'
            '<div class="nm-sub">Fitting the four scoring weights from simulated feedback.</div>')
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
        a.metric("Hand-set", f"{report.satisfaction_default:.1%}")
        b.metric("Learned", f"{report.satisfaction_learned:.1%}")
        c.metric("Best possible", f"{report.satisfaction_oracle:.1%}")
    with right:
        st.bar_chart(table.set_index("Factor")[["Hand-set", "Hidden preference", "Learned"]],
                     stack=False, color=["#9FB3C0", "#E5A93D", "#0F5C58"])

    st.html('<div class="section-title">How much feedback is enough?</div>')
    st.caption("Distance between learned and hidden weights, by number of past matches (lower is better).")
    st.line_chart(report.curve.set_index("n_matches")["weight_error"], color="#0F5C58", height=240)
    st.caption(
        f"The weights above were learned from {report.n_training_matches} simulated matches. "
        "Because the feedback comes from a known rule, this shows the method works, not how well it "
        "would do on real feedback."
    )


# ---- main -------------------------------------------------------------------


def main() -> None:
    st.html(CSS)
    head, nav = st.columns([1.5, 1], gap="medium", vertical_alignment="center")
    head.html(
        '<div class="nm-brand"><div class="nm-logo"></div><div>'
        '<div class="t">Nurse Match <em>Engine</em><span class="nm-pill">Synthetic data</span></div>'
        "<div class=\"s\">Find the best nurse for your care request, with complete transparency.</div>"
        "</div></div>")
    with nav:
        with st.container(key="nav"):
            page = st.segmented_control("Page", PAGES, default=PAGES[0], key="page_choice",
                                        label_visibility="collapsed") or PAGES[0]
    if page == "Find a Nurse":
        find_page()
    else:
        with st.container(key="page"):
            {"Fairness Analytics": fairness_page, "Learned Weights": learning_page}[page]()


main()

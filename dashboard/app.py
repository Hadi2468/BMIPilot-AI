"""Streamlit dashboard: assess BMI for any age, see it on the right reference chart,
and generate a personalised plan.

The dashboard is a thin client over the FastAPI service, so every interface uses the
same validated assessment and agent.

Run:  streamlit run dashboard/app.py
"""

from __future__ import annotations

import os
from typing import Any

import altair as alt
import pandas as pd
import requests
import streamlit as st

REQUEST_TIMEOUT_S = 180

# Three age groups, three methods.
GROUPS = {
    "infant_toddler": ("👶 0-2 years", "WHO Child Growth Standards: BMI-for-age z-score"),
    "child_teen": ("🧒 2-19 years", "CDC Growth Charts: BMI-for-age percentile"),
    "adult": ("🧑 20+ years", "WHO adult BMI classification"),
}
PRESETS: dict[str, dict[str, dict[str, Any]]] = {
    "infant_toddler": {
        "Newborn girl, 1 day": {
            "name": "Baby Sara",
            "sex": "female",
            "days": 1,
            "w": 3.3,
            "h": 50.0,
        },
        "Boy, 9 months": {"name": "Baby Leo", "sex": "male", "days": 274, "w": 9.2, "h": 72.0},
        "Girl, 20 months": {"name": "Mina", "sex": "female", "days": 609, "w": 12.9, "h": 82.0},
    },
    "child_teen": {
        "Boy, 9 years": {
            "name": "Leo",
            "sex": "male",
            "years": 9,
            "months": 0,
            "w": 38.0,
            "h": 135.0,
        },
        "Girl, 15 years": {
            "name": "Ava",
            "sex": "female",
            "years": 15,
            "months": 6,
            "w": 52.0,
            "h": 162.0,
        },
        "Boy, 4 years": {
            "name": "Sam",
            "sex": "male",
            "years": 4,
            "months": 3,
            "w": 16.5,
            "h": 104.0,
        },
    },
    "adult": {
        "Woman, 49 years": {"name": "Mrs. ST", "sex": "female", "years": 49, "w": 62.0, "h": 157.0},
        "Man, 82 years": {"name": "Ali", "sex": "male", "years": 82, "w": 95.0, "h": 170.0},
        "Man, 30 years": {"name": "Omid", "sex": "male", "years": 30, "w": 55.0, "h": 180.0},
    },
}
# Risk is status: always icon + label, never color alone.
RISK_STATUS = {"Low": "✅", "Moderate": "⚠️", "High": "🟠", "Very high": "🔴"}

st.set_page_config(page_title="BMIPilot AI", page_icon="⚖️", layout="wide")


# ============================================================
# Theme tokens (light / dark selected separately, not flipped)
# ============================================================

DARK = getattr(getattr(st.context, "theme", None), "type", "light") == "dark"
INK = "#c3c2b7" if DARK else "#52514e"  # reference curves and labels
INK_STRONG = "#ffffff" if DARK else "#0b0b0b"
SURFACE = "#1a1a19" if DARK else "#fcfcfb"
POINT = "#3987e5" if DARK else "#2a78d6"  # the person: series slot 1
NEUTRAL_MID = "#383835" if DARK else "#f0efec"
HEALTHY_ZONE = "#0ca30c"  # status "good"; always paired with a text label
# Adult scale is diverging around "Normal": blue arm (under), red arm (over).
ADULT_BANDS = [
    ("Severe thinness", 12.0, 16.0, "#256abf" if DARK else "#6da7ec"),
    ("Underweight", 16.0, 18.5, "#1c5cab" if DARK else "#b7d3f6"),
    ("Normal", 18.5, 25.0, NEUTRAL_MID),
    ("Overweight", 25.0, 30.0, "#7a3b37" if DARK else "#f6c9c4"),
    ("Obesity I", 30.0, 35.0, "#a8423f" if DARK else "#ef9a93"),
    ("Obesity II", 35.0, 40.0, "#c94a48" if DARK else "#e5625f"),
    ("Obesity III", 40.0, 50.0, "#e66767" if DARK else "#c63c3b"),
]


# ============================================================
# API client
# ============================================================


def api(method: str, path: str, **kwargs: Any) -> Any | None:
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    try:
        resp = requests.request(
            method, f"{api_url}{path}", headers=headers, timeout=REQUEST_TIMEOUT_S, **kwargs
        )
    except requests.ConnectionError:
        st.error(f"Cannot reach the API at {api_url}. Is it running?")
        return None
    if resp.status_code == 422:
        details = resp.json().get("detail", [])
        msgs = [d.get("msg", "").removeprefix("Value error, ") for d in details]
        st.error("Invalid input: " + "; ".join(msgs))
        return None
    if not resp.ok:
        st.error(f"API error {resp.status_code}: {resp.text}")
        return None
    return resp.json()


@st.cache_data(show_spinner=False)
def reference(source: str, sex: str, url: str, auth: str) -> pd.DataFrame:
    headers = {"Authorization": f"Bearer {auth}"} if auth else {}
    resp = requests.get(f"{url}/reference/{source}/{sex}", headers=headers, timeout=30)
    resp.raise_for_status()
    return pd.DataFrame(resp.json())


# ============================================================
# Charts
# ============================================================


def growth_chart(group: str, profile: dict[str, Any], a: dict[str, Any]) -> alt.LayerChart:
    """Reference curves (muted ink, direct-labelled) with the person as one strong point."""
    source = "WHO" if group == "infant_toddler" else "CDC"
    df = reference(source, profile["sex"], api_url, token)
    if source == "WHO":
        curves = {"z-2": "−2 SD", "z+0": "Median", "z+1": "+1 SD", "z+2": "+2 SD"}
        healthy = ("z-2", "z+1")
    else:
        curves = {"p5": "5th", "p50": "50th", "p85": "85th", "p95": "95th"}
        healthy = ("p5", "p85")

    long = df.melt("age_years", list(curves), var_name="key", value_name="bmi")
    long["curve"] = long["key"].map(curves)
    long["median"] = long["key"].isin(["z+0", "p50"])

    x = alt.X(
        "age_years:Q",
        title="Age (years)",
        scale=alt.Scale(domain=[df.age_years.min(), df.age_years.max()], nice=False),
    )
    y = alt.Y("bmi:Q", title="BMI (kg/m²)", scale=alt.Scale(zero=False))
    zone = (
        alt.Chart(df)
        .mark_area(color=HEALTHY_ZONE, opacity=0.14)
        .encode(x=x, y=alt.Y(f"{healthy[0]}:Q"), y2=f"{healthy[1]}:Q")
    )
    first = df.iloc[[len(df) // 2]].assign(bmi=lambda d: (d[healthy[0]] + d[healthy[1]]) / 2)
    zone_label = (
        alt.Chart(first)
        .mark_text(fontSize=11, fontStyle="italic", color=INK, dy=14)
        .encode(x=x, y=y, text=alt.value("Healthy zone"))
    )
    lines = (
        alt.Chart(long)
        .mark_line(strokeWidth=2, color=INK)
        .encode(
            x=x,
            y=y,
            detail="curve:N",
            strokeDash=alt.condition("datum.median", alt.value([1, 0]), alt.value([4, 4])),
            opacity=alt.condition("datum.median", alt.value(0.9), alt.value(0.6)),
            tooltip=[
                alt.Tooltip("curve:N", title="Curve"),
                alt.Tooltip("age_years:Q", title="Age", format=".2f"),
                alt.Tooltip("bmi:Q", title="BMI", format=".1f"),
            ],
        )
    )
    last = long[long.age_years == long.age_years.max()]
    labels = (
        alt.Chart(last)
        .mark_text(align="left", dx=6, fontSize=11, color=INK)
        .encode(x=x, y=y, text="curve:N")
    )
    me = pd.DataFrame(
        [
            {
                "age_years": profile["age_years"],
                "bmi": a["bmi"],
                "who": profile["name"],
                "category": a["category"],
                "position": f"percentile {a['percentile']} (z {a['z_score']})",
            }
        ]
    )
    point = (
        alt.Chart(me)
        .mark_point(filled=True, size=160, color=POINT, stroke=SURFACE, strokeWidth=2, opacity=1)
        .encode(
            x=x,
            y=y,
            tooltip=[
                alt.Tooltip("who:N", title="Name"),
                alt.Tooltip("bmi:Q", title="BMI", format=".1f"),
                alt.Tooltip("category:N", title="Category"),
                alt.Tooltip("position:N", title="Position"),
            ],
        )
    )
    point_label = (
        alt.Chart(me)
        .mark_text(dy=-14, fontSize=12, fontWeight="bold", color=INK_STRONG)
        .encode(x=x, y=y, text="who:N")
    )
    return (zone + zone_label + lines + labels + point + point_label).properties(height=380)


def adult_scale(a: dict[str, Any]) -> alt.LayerChart:
    """WHO adult bands as a diverging scale around Normal, with the person's BMI marked."""
    bands = pd.DataFrame(ADULT_BANDS, columns=["category", "start", "end", "color"])
    bands["mid"] = (bands.start + bands.end) / 2
    bands["label_y"] = [2.2 if i % 2 == 0 else 0.6 for i in range(len(bands))]  # stagger rows
    domain = [12, 50]
    x_axis = alt.Axis(
        values=[16, 18.5, 25, 30, 35, 40], format=".1~f", grid=False, title="BMI (kg/m²)"
    )
    y_hidden = alt.Y("y:Q", scale=alt.Scale(domain=[0, 10]), axis=None)

    segments = (
        alt.Chart(bands.assign(y=3.6, y2=6.4))
        .mark_rect(stroke=SURFACE, strokeWidth=2, cornerRadius=2)
        .encode(
            x=alt.X("start:Q", scale=alt.Scale(domain=domain, nice=False), axis=x_axis),
            x2="end:Q",
            y=y_hidden,
            y2="y2:Q",
            color=alt.Color("color:N", scale=None),
            tooltip=[
                alt.Tooltip("category:N", title="Category"),
                alt.Tooltip("start:Q", title="From BMI"),
                alt.Tooltip("end:Q", title="To BMI"),
            ],
        )
    )
    band_labels = (
        alt.Chart(bands.rename(columns={"label_y": "y"}))
        .mark_text(fontSize=11, color=INK)
        .encode(x=alt.X("mid:Q", scale=alt.Scale(domain=domain)), y=y_hidden, text="category:N")
    )
    bmi = min(max(a["bmi"], domain[0] + 0.5), domain[1] - 0.5)
    me = pd.DataFrame([{"bmi": bmi, "y": 3.0, "y2": 7.0, "label": f"You: {a['bmi']}"}])
    x_me = alt.X("bmi:Q", scale=alt.Scale(domain=domain))
    marker = (
        alt.Chart(me)
        .mark_rule(color=INK_STRONG, strokeWidth=3)
        .encode(x=x_me, y=y_hidden, y2="y2:Q")
    )
    marker_label = (
        alt.Chart(me.assign(y=8.6))
        .mark_text(fontSize=13, fontWeight="bold", color=INK_STRONG)
        .encode(x=x_me, y=y_hidden, text="label:N")
    )
    return (segments + band_labels + marker + marker_label).properties(height=170)


# ============================================================
# Inputs (one form per age group)
# ============================================================


def profile_form(group: str) -> dict[str, Any]:
    preset_name = st.selectbox("Example", list(PRESETS[group]), key=f"preset_{group}")
    p = PRESETS[group][preset_name]
    k = f"{group}_{preset_name}"  # new preset -> fresh widget defaults

    c1, c2, c3, c4, c5 = st.columns([2, 1.3, 1.6, 1.2, 1.2])
    name = c1.text_input("Name", p["name"], key=f"name_{k}")
    if group == "adult":
        sex_options = ["female", "male", "prefer not to say"]
        sex = c2.selectbox("Sex", sex_options, sex_options.index(p["sex"]), key=f"sex_{k}")
        sex = None if sex == "prefer not to say" else sex
        age_years = c3.number_input("Age (years)", 20, 150, p["years"], key=f"age_{k}")
        height_label = "Height (cm)"
    else:
        sex = c2.selectbox(
            "Sex (required)", ["female", "male"], ["female", "male"].index(p["sex"]), key=f"sex_{k}"
        )
        if group == "infant_toddler":
            days = c3.number_input("Age (days)", 0, 730, p["days"], key=f"age_{k}")
            c3.caption(f"≈ {days / 30.4375:.1f} months")
            age_years = days / 365.25
            height_label = "Length (cm, lying down)"
        else:
            a1, a2 = c3.columns(2)
            years = a1.number_input("Years", 2, 19, p["years"], key=f"age_{k}")
            months = a2.number_input("Months", 0, 11, p["months"], key=f"mon_{k}")
            age_years = years + months / 12
            height_label = "Height (cm)"
    weight = c4.number_input("Weight (kg)", 0.5, 400.0, p["w"], 0.1, key=f"w_{k}")
    height = c5.number_input(height_label, 30.0, 260.0, p["h"], 0.5, key=f"h_{k}")
    return {
        "name": name,
        "age_years": age_years,
        "sex": sex,
        "weight_kg": weight,
        "height_m": round(height / 100, 4),
    }


# ============================================================
# Rendering
# ============================================================


def render_assessment(group: str, profile: dict[str, Any], a: dict[str, Any]) -> None:
    st.caption(f"Method: **{a['method']}**")
    cols = st.columns(4)
    cols[0].metric("BMI", a["bmi"])
    cols[1].metric("Category", a["category"])
    if a["percentile"] is not None:
        cols[2].metric("Percentile for age & sex", f"{a['percentile']}", f"z = {a['z_score']}")
    else:
        cols[2].metric("BMI Prime", a["bmi_prime"])
    cols[3].metric("Risk", f"{RISK_STATUS[a['risk_level']]} {a['risk_level']}")

    left, right = st.columns([3, 2])
    with left:
        if group == "adult":
            st.markdown("**Where you are on the WHO adult scale**")
            st.altair_chart(adult_scale(a), width="stretch")
        else:
            title = (
                "WHO BMI-for-age (0-2 years)"
                if group == "infant_toddler"
                else ("CDC BMI-for-age percentiles (2-20 years)")
            )
            st.markdown(f"**{title}, {profile['sex']}**")
            st.altair_chart(growth_chart(group, profile, a), width="stretch")
    with right:
        st.markdown("**Healthy range for this height**")
        st.markdown(
            f"- BMI **{a['healthy_bmi_min']}-{a['healthy_bmi_max']}**\n"
            f"- Weight **{a['healthy_weight_min_kg']}-{a['healthy_weight_max_kg']} kg**"
        )
        delta = a["weight_to_healthy_kg"]
        if delta is None:
            st.info("No weight target for children: focus on healthy growth and habits.")
        elif delta == 0:
            st.success("✅ Within the healthy range.")
        else:
            verb = "Gain" if delta > 0 else "Lose"
            st.warning(f"⚠️ {verb} about {abs(delta)} kg to reach the healthy range.")
        st.markdown("**WHO activity guideline for this age**")
        st.caption(a["activity_guideline"])
        for note in a["notes"]:
            st.caption(f"ℹ️ {note}")


def render_plan(result: dict[str, Any]) -> None:
    st.divider()
    st.subheader("🧭 Personal plan")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### 🏃 Activity")
        st.markdown(result["exercise_plan"])
        if result["exercises"]:
            with st.expander(f"Exercises from the database ({len(result['exercises'])})"):
                st.dataframe(pd.DataFrame(result["exercises"]), hide_index=True)
    with c2:
        st.markdown("#### 🥗 Diet")
        st.markdown(f"{result['diet_tip']['tip']}  \n*{result['diet_tip']['why']}*")
        st.markdown("#### 🌙 Lifestyle")
        st.markdown(f"{result['lifestyle_tip']['tip']}  \n*{result['lifestyle_tip']['why']}*")
    st.markdown("#### 💬 What this means")
    st.markdown(result["explanation"])
    for warning in result["warnings"]:
        st.warning(f"⚠️ {warning}")
    st.download_button(
        "⬇️ Download report (Markdown)",
        result["report"],
        file_name=f"bmipilot_{result['profile']['name'].replace(' ', '_')}.md",
    )


# ============================================================
# Page
# ============================================================

with st.sidebar:
    st.title("⚖️ BMIPilot AI")
    st.caption("Age-aware BMI coach: newborn to 150 years")
    api_url = st.text_input("API URL", os.getenv("BMIPILOT_API_URL", "http://localhost:8000"))
    token = st.text_input("API token", os.getenv("BMIPILOT_API_TOKEN", ""), type="password")
    health = api("GET", "/health")
    if health:
        st.success(f"API online · v{health['version']}")
    st.divider()
    st.markdown(
        "**Three methods, chosen by age**\n\n"
        "- **0-2 years:** WHO growth standards (z-score)\n"
        "- **2-19 years:** CDC percentiles\n"
        "- **20+ years:** WHO adult classes"
    )
    st.caption(
        "For general wellness information only, not medical advice. "
        "Consult a healthcare professional."
    )

st.header("BMI for every age")
group = st.radio(
    "Age group",
    list(GROUPS),
    format_func=lambda g: GROUPS[g][0],
    horizontal=True,
    label_visibility="collapsed",
)
st.caption(GROUPS[group][1])

profile = profile_form(group)
b1, b2, _ = st.columns([1, 1.4, 3])
if b1.button("Assess BMI", type="primary", width="stretch"):
    a = api("POST", "/assess", json=profile)
    st.session_state[f"assess_{group}"] = (profile, a) if a else None
    st.session_state.pop(f"plan_{group}", None)
if b2.button("✨ Generate full plan", width="stretch"):
    with st.spinner("Assessing, searching exercises and writing your plan…"):
        result = api("POST", "/coach", json=profile)
    if result:
        st.session_state[f"assess_{group}"] = (result["profile"], result["assessment"])
        st.session_state[f"plan_{group}"] = result

saved = st.session_state.get(f"assess_{group}")
if saved:
    st.divider()
    render_assessment(group, *saved)
plan = st.session_state.get(f"plan_{group}")
if plan:
    render_plan(plan)

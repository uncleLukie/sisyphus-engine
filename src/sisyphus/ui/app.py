"""Inside the Game - second-screen dashboard (Streamlit).

Two audiences, one engine: a casual fan sees plain-language moments in their
language with a single energy gauge; an analyst sees the evidence, tactical
labels, rolling physical metrics and the provenance of every sentence.
"""

from __future__ import annotations

import contextlib
import os
import time

import streamlit as st

from sisyphus.ui.charts import energy_gauge, energy_timeline, momentum_bar, pitch_figure
from sisyphus.ui.client import StreamClient, api_get, api_post

API = os.environ.get("SISYPHUS_API_URL", "http://localhost:8000")
LANGUAGES = {
    "English": "en",
    "Español": "es",
    "Português (BR)": "pt-BR",
    "Français": "fr",
    "Deutsch": "de",
    "Italiano": "it",
    "العربية": "ar",
    "日本語": "ja",
    "한국어": "ko",
    "हिन्दी": "hi",
}
KIND_ICON = {
    "goal": "⚽",
    "big_chance": "🎯",
    "fatigue_breakpoint": "🔋",
    "press_drop": "📉",
    "momentum_swing": "🔁",
    "substitution": "🔄",
    "card": "🟨",
    "milestone": "📏",
    "speed_burst": "⚡",
    "period_summary": "📋",
    "full_time": "🏁",
}

st.set_page_config(page_title="Inside the Game", page_icon="⚽", layout="wide")


# ---------------------------------------------------------------- data helpers
@st.cache_data(ttl=10)
def _health() -> dict:
    try:
        return api_get(API, "/health")
    except Exception as exc:  # noqa: BLE001
        return {"status": f"unreachable ({exc})"}


@st.cache_data(ttl=5)
def _matches() -> list[dict]:
    try:
        return api_get(API, "/matches")
    except Exception:  # noqa: BLE001
        return []


@st.cache_data(ttl=300)
def _meta(match_id: str) -> dict:
    return api_get(API, f"/matches/{match_id}")


def _names(meta: dict) -> dict[str, str]:
    return {p["player_id"]: p["short_name"] for t in (meta["home"], meta["away"]) for p in t["players"]}


def _full_names(meta: dict) -> dict[str, str]:
    return {p["player_id"]: p["name"] for t in (meta["home"], meta["away"]) for p in t["players"]}


def _persona_output(insight: dict, persona_id: str) -> dict | None:
    for p in insight["personalised"]:
        if p["persona_id"] == persona_id:
            return p
    return insight["personalised"][0] if insight["personalised"] else None


def _render_insight(ins: dict, persona_id: str, analyst: bool, team_colors: dict[str, str]) -> None:
    out = _persona_output(ins, persona_id)
    if out is None:
        return
    icon = KIND_ICON.get(ins["kind"], "•")
    color = team_colors.get(ins.get("team_id") or "", "#888")
    prov = ins["provenance"]
    badge = {"llm": "🤖 model", "mixed": "🤖+📐 mixed", "template": "📐 template"}[prov["mode"]]
    with st.container(border=True):
        st.markdown(
            f"<div style='border-left:4px solid {color};padding-left:8px'>"
            f"<span style='font-size:0.85em;opacity:0.7'>{ins['clock']} · {icon} {ins['kind'].replace('_', ' ')}</span><br>"
            f"<b style='font-size:1.05em'>{out['headline']}</b></div>",
            unsafe_allow_html=True,
        )
        st.write(out["body"])
        if out["stat_callouts"]:
            cols = st.columns(len(out["stat_callouts"]))
            for c, s in zip(cols, out["stat_callouts"], strict=False):
                unit = f" {s['unit']}" if s.get("unit") else ""
                arrow = {"up": " ↑", "down": " ↓", "flat": " →"}.get(s.get("trend") or "", "")
                c.metric(s["label"], f"{s['value']}{unit}{arrow}")
        if analyst:
            with st.expander("Why it matters · evidence · provenance"):
                x = ins["explanation"]
                st.markdown(f"**{x['headline']}** — _{x['tactical_label']}_, confidence {x['confidence']:.2f}")
                st.write(x["why_it_matters"])
                for e in x["evidence"]:
                    st.markdown(f"- {e}")
                st.caption(f"Studio note: {ins['commentary']['studio_note']}")
                st.caption(
                    f"{badge} · model={prov.get('model') or '-'} · agents={', '.join(prov['agents']) or '-'} · "
                    f"retries={prov['retries']} · fallbacks={', '.join(prov['fallbacks']) or 'none'} · "
                    f"grounded={'yes' if prov['grounding_ok'] else 'no'} · {prov['latency_ms']:.0f} ms"
                )
        else:
            st.caption(f"{badge}")


# ---------------------------------------------------------------------- sidebar
health = _health()
with st.sidebar:
    st.title("⚽ Inside the Game")
    st.caption(
        f"API {API} · {health.get('status')} · agents: {health.get('provider', '-')}"
        + (f" ({health.get('model')})" if health.get("model") else "")
    )
    matches = _matches()
    if not matches:
        st.warning("No matches yet. Generate one below.")
    match_ids = [m["match_id"] for m in matches]
    match_id = st.selectbox(
        "Match",
        match_ids,
        index=0 if match_ids else None,
        format_func=lambda mid: next(
            (
                f"{m['home']} v {m['away']} · {m['scenario_id']} · seed {m['seed']}"
                for m in matches
                if m["match_id"] == mid
            ),
            mid,
        ),
    )

    st.subheader("Who are you?")
    persona_kind = st.radio("Mode", ["Fan", "Analyst"], horizontal=True)
    lang_label = st.selectbox("Language", list(LANGUAGES), index=0)
    lang = LANGUAGES[lang_label]
    fav_options = ["Neutral"]
    meta = _meta(match_id) if match_id else None
    if meta:
        fav_options += [meta["home"]["name"], meta["away"]["name"]]
    favourite = st.selectbox("Supporting", fav_options)

    st.subheader("Playback")
    speed = st.select_slider("Speed", options=[1, 2, 5, 10, 20, 40, 60, 120], value=20)
    from_minute = st.slider("Join at minute", 0, 90, 55, step=5)
    live_agents = st.toggle("Run agents live (else replay stored insights)", value=True)
    only_big = st.toggle("Major moments only", value=True)

    c1, c2 = st.columns(2)
    start = c1.button("▶ Start", type="primary", width="stretch", disabled=not match_id)
    stop = c2.button("■ Stop", width="stretch")

    with st.expander("Generate a new match"):
        scenarios = []
        with contextlib.suppress(Exception):
            scenarios = api_get(API, "/scenarios")
        sc = st.selectbox("Scenario", [s["scenario_id"] for s in scenarios] or ["press_collapse"])
        seed = st.number_input("Seed", min_value=1, max_value=99999, value=7)
        narrate = st.checkbox("Pre-narrate (stored insights)", value=False)
        if st.button("Simulate", width="stretch"):
            job = api_post(API, "/simulate", {"scenario_id": sc, "seed": int(seed), "narrate": narrate})
            st.session_state["job"] = job["job_id"]
        if "job" in st.session_state:
            j = api_get(API, f"/jobs/{st.session_state['job']}")
            st.progress(min(1.0, float(j.get("progress", 0))), text=f"{j['status']} {j.get('match_id', '')}")
            if j["status"] in ("done", "failed"):
                st.caption(j.get("error") or "ready - refresh the match list")
                if st.button("Refresh matches"):
                    _matches.clear()
                    del st.session_state["job"]
                    st.rerun()

persona_id = f"{'analyst' if persona_kind == 'Analyst' else 'casual'}-{lang}"
analyst = persona_kind == "Analyst"
persona_spec = f"casual:{lang},analyst:{lang}" if lang != "en" else "casual:en,analyst:en"
kinds = (
    "goal,big_chance,fatigue_breakpoint,press_drop,momentum_swing,substitution,period_summary,full_time"
    if only_big
    else None
)

if stop and "client" in st.session_state:
    st.session_state["client"].stop()
    del st.session_state["client"]
if start and match_id:
    if "client" in st.session_state:
        st.session_state["client"].stop()
    params = {
        "speed": speed,
        "from_minute": from_minute,
        "personas": persona_spec,
        "live_agents": str(live_agents).lower(),
    }
    if favourite != "Neutral":
        params["favourite_team"] = favourite
    if kinds:
        params["kinds"] = kinds
    st.session_state["client"] = StreamClient(API, match_id, params)
    st.session_state["client_meta"] = meta

client: StreamClient | None = st.session_state.get("client")

# ------------------------------------------------------------------------ main
if client is None or meta is None:
    st.title("Inside the Game")
    st.markdown(
        "Synthetic tracking data → interpretable physical analytics → a multi-agent "
        "narrative, personalised per viewer. Pick a match and press **Start**."
    )
    if match_id and meta:
        tab1, tab2, tab3 = st.tabs(["Match review", "Triggers", "Evaluation"])
        with tab1:
            tl = api_get(API, f"/matches/{match_id}/timeline", every=5)
            trig = api_get(API, f"/matches/{match_id}/triggers")
            hist = [
                {
                    "minute": (r["period"] - 1) * 45 + r["t"] / 60,
                    "home_energy": r["teams"][0]["energy"],
                    "away_energy": r["teams"][1]["energy"],
                    "momentum": r["momentum"],
                }
                for r in tl
            ]
            if hist:
                st.plotly_chart(energy_timeline(hist, meta["home"], meta["away"], trig), width="stretch")
                st.caption(
                    "Energy index per team (observable, phase-normalised) with goals ⚽ gold, fatigue breakpoints red, substitutions blue."
                )
            else:
                st.info("Run `sisyphus analyze` on this match (or start a live replay) to build the timeline.")
            insights = api_get(API, f"/matches/{match_id}/insights")
            if insights:
                st.subheader("Stored insights")
                colors = {
                    meta["home"]["team_id"]: meta["home"]["primary_color"],
                    meta["away"]["team_id"]: meta["away"]["primary_color"],
                }
                for ins in insights:
                    if ins["kind"] in (
                        "goal",
                        "fatigue_breakpoint",
                        "press_drop",
                        "period_summary",
                        "full_time",
                        "substitution",
                    ):
                        _render_insight(ins, persona_id, analyst, colors)
        with tab2:
            trig = api_get(API, f"/matches/{match_id}/triggers")
            st.dataframe(
                [
                    {
                        "clock": t["clock"],
                        "kind": t["kind"],
                        "severity": t["severity"],
                        "team": t["team_id"],
                        "headline": t["headline"],
                    }
                    for t in trig
                ],
                width="stretch",
                hide_index=True,
            )
        with tab3:
            st.json(api_get(API, f"/matches/{match_id}/eval"))
    st.stop()

meta = st.session_state.get("client_meta") or meta
home, away = meta["home"], meta["away"]
names = _names(meta)
colors = {home["team_id"]: home["primary_color"], away["team_id"]: away["primary_color"]}


@st.fragment(run_every=0.5)
def live_view() -> None:
    c = st.session_state.get("client")
    if c is None:
        return
    with c.lock:
        tick = c.tick
        pos = c.positions
        seek = c.seek
        insights = list(c.insights)
        hist = list(c.history)
        status = c.status
        errors = list(c.errors)

    # score bug
    if tick:
        sh, sa = tick["score"]
        th, ta = tick["teams"]
        st.markdown(
            f"<div style='display:flex;gap:18px;align-items:baseline;font-size:1.6em'>"
            f"<span style='color:{home['primary_color']};font-weight:700'>{home['name']}</span>"
            f"<span style='font-weight:800'>{sh} – {sa}</span>"
            f"<span style='color:{away['primary_color']};font-weight:700'>{away['name']}</span>"
            f"<span style='opacity:0.6;font-size:0.7em'>⏱ {tick['clock']} · {status} · ×{c.params['speed']}</span></div>",
            unsafe_allow_html=True,
        )
    elif seek:
        st.progress(
            min(1.0, seek["minute"] / max(seek["target"], 1)),
            text=f"Catching up analytics to minute {seek['target']:.0f} (at {seek['minute']:.0f}')",
        )
    else:
        st.info(f"Connecting… ({status})")
    for e in errors[-2:]:
        st.error(e)

    left, right = st.columns([3, 2])
    with left:
        st.plotly_chart(pitch_figure(pos, home, away, names), width="stretch", key="pitch")
        if tick:
            st.markdown(momentum_bar(tick["momentum"], home, away), unsafe_allow_html=True)
            st.caption(
                f"Momentum (3-min threat balance) · possession {th['possession_pct']:.0f}% / {ta['possession_pct']:.0f}% · xG {th['xg']:.2f} / {ta['xg']:.2f}"
            )
    with right:
        if tick:
            g1, g2 = st.columns(2)
            g1.plotly_chart(
                energy_gauge(th["energy"], f"{home['short_name']} energy", home["primary_color"], th["energy_trend"]),
                width="stretch",
                key="g1",
            )
            g2.plotly_chart(
                energy_gauge(ta["energy"], f"{away['short_name']} energy", away["primary_color"], ta["energy_trend"]),
                width="stretch",
                key="g2",
            )
            if analyst:
                m1, m2, m3, m4 = st.columns(4)
                m1.metric(f"{home['short_name']} sprints/5'", th["sprints_5min"])
                m2.metric(f"{away['short_name']} sprints/5'", ta["sprints_5min"])
                m3.metric(f"{home['short_name']} press/min", f"{th['press_rate']:.1f}")
                m4.metric(f"{away['short_name']} press/min", f"{ta['press_rate']:.1f}")
        if hist:
            st.plotly_chart(energy_timeline(hist, home, away), width="stretch", key="tl")

    st.subheader("Commentary" if not analyst else "Analyst feed")
    if not insights:
        st.caption("Waiting for the first moment…")
    for ins in reversed(insights[-12:]):
        _render_insight(ins, persona_id, analyst, colors)


live_view()

if client is not None and not client.running and client.status == "finished":
    st.success("Replay finished.")
    time.sleep(0.1)

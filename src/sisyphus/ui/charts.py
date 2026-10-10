"""Plotly figures for the dashboard."""

from __future__ import annotations

import plotly.graph_objects as go

LENGTH, WIDTH = 105.0, 68.0


def _pitch_shapes() -> list[dict]:
    line = {"color": "rgba(255,255,255,0.55)", "width": 1.5}
    shapes = [
        {"type": "rect", "x0": 0, "y0": 0, "x1": LENGTH, "y1": WIDTH, "line": line},
        {"type": "line", "x0": LENGTH / 2, "y0": 0, "x1": LENGTH / 2, "y1": WIDTH, "line": line},
        {
            "type": "circle",
            "x0": LENGTH / 2 - 9.15,
            "y0": WIDTH / 2 - 9.15,
            "x1": LENGTH / 2 + 9.15,
            "y1": WIDTH / 2 + 9.15,
            "line": line,
        },
    ]
    for x0, x1 in ((0, 16.5), (LENGTH - 16.5, LENGTH)):
        shapes.append(
            {"type": "rect", "x0": x0, "y0": WIDTH / 2 - 20.16, "x1": x1, "y1": WIDTH / 2 + 20.16, "line": line}
        )
    for x0, x1 in ((0, 5.5), (LENGTH - 5.5, LENGTH)):
        shapes.append(
            {"type": "rect", "x0": x0, "y0": WIDTH / 2 - 9.16, "x1": x1, "y1": WIDTH / 2 + 9.16, "line": line}
        )
    for x0, x1 in ((-2, 0), (LENGTH, LENGTH + 2)):
        shapes.append(
            {"type": "rect", "x0": x0, "y0": WIDTH / 2 - 3.66, "x1": x1, "y1": WIDTH / 2 + 3.66, "line": line}
        )
    return shapes


def pitch_figure(positions: dict | None, home: dict, away: dict, names: dict[str, str]) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        shapes=_pitch_shapes(),
        plot_bgcolor="#2e7d32",
        paper_bgcolor="rgba(0,0,0,0)",
        margin={"l": 5, "r": 5, "t": 5, "b": 5},
        height=420,
        showlegend=False,
        xaxis={"range": [-3, LENGTH + 3], "visible": False, "fixedrange": True},
        yaxis={"range": [-2, WIDTH + 2], "visible": False, "scaleanchor": "x", "scaleratio": 1, "fixedrange": True},
    )
    if not positions:
        return fig
    for team in (home, away):
        pts = [p for p in positions["players"] if p[1] == team["team_id"]]
        fig.add_trace(
            go.Scatter(
                x=[p[2] for p in pts],
                y=[p[3] for p in pts],
                mode="markers+text",
                marker={"size": 16, "color": team["primary_color"], "line": {"color": "white", "width": 1}},
                text=[names.get(p[0], "").split("-")[-1] for p in pts],
                textposition="top center",
                textfont={"size": 9, "color": "white"},
                hovertext=[f"{names.get(p[0], p[0])}: {p[4]:.1f} km/h" for p in pts],
                hoverinfo="text",
            )
        )
    bx, by = positions["ball"]
    fig.add_trace(
        go.Scatter(
            x=[bx],
            y=[by],
            mode="markers",
            marker={"size": 9, "color": "white", "line": {"color": "black", "width": 1}},
            hoverinfo="skip",
        )
    )
    return fig


def energy_gauge(value: float, label: str, color: str, trend: str) -> go.Figure:
    arrow = {"rising": "↑", "falling": "↓"}.get(trend, "→")
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value,
            number={"suffix": f" {arrow}", "font": {"size": 30}},
            title={"text": label, "font": {"size": 14}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 0},
                "bar": {"color": color},
                "steps": [
                    {"range": [0, 50], "color": "rgba(255,80,80,0.25)"},
                    {"range": [50, 65], "color": "rgba(255,180,60,0.25)"},
                    {"range": [65, 80], "color": "rgba(255,230,80,0.2)"},
                    {"range": [80, 100], "color": "rgba(80,200,120,0.2)"},
                ],
            },
        )
    )
    fig.update_layout(height=200, margin={"l": 15, "r": 15, "t": 40, "b": 5}, paper_bgcolor="rgba(0,0,0,0)")
    return fig


def energy_timeline(history, home: dict, away: dict, triggers: list[dict] | None = None) -> go.Figure:
    rows = list(history)
    fig = go.Figure()
    if rows:
        m = [r["minute"] for r in rows]
        fig.add_trace(
            go.Scatter(
                x=m,
                y=[r["home_energy"] for r in rows],
                name=home["short_name"],
                line={"color": home["primary_color"], "width": 2.5},
            )
        )
        fig.add_trace(
            go.Scatter(
                x=m,
                y=[r["away_energy"] for r in rows],
                name=away["short_name"],
                line={"color": away["primary_color"], "width": 2.5},
            )
        )
        fig.add_trace(
            go.Scatter(
                x=m,
                y=[50 + 50 * r["momentum"] for r in rows],
                name="momentum (home ↑)",
                line={"color": "rgba(200,200,200,0.5)", "width": 1, "dash": "dot"},
                yaxis="y",
            )
        )
    for t in triggers or []:
        if t["kind"] in ("goal", "fatigue_breakpoint", "substitution"):
            color = {"goal": "gold", "fatigue_breakpoint": "tomato", "substitution": "deepskyblue"}[t["kind"]]
            fig.add_vline(x=t["minute"], line={"color": color, "width": 1, "dash": "dash"})
    fig.update_layout(
        height=260,
        margin={"l": 30, "r": 10, "t": 10, "b": 30},
        yaxis={"range": [0, 105], "title": "Energy index"},
        xaxis={"title": "minute", "range": [0, 96]},
        legend={"orientation": "h", "y": 1.1},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def momentum_bar(momentum: float, home: dict, away: dict) -> str:
    """A compact HTML momentum bar: fills left from centre for the away side, right for home."""
    m = max(-1.0, min(1.0, momentum))
    pct = abs(m) * 50
    if m >= 0:
        fill = f"left:50%;width:{pct:.1f}%;background:{home['primary_color']}"
    else:
        fill = f"right:50%;width:{pct:.1f}%;background:{away['primary_color']}"
    return (
        "<div style='position:relative;height:14px;border-radius:7px;background:rgba(255,255,255,0.12);margin:4px 0'>"
        f"<div style='position:absolute;top:0;bottom:0;{fill};border-radius:7px'></div>"
        "<div style='position:absolute;left:50%;top:-3px;bottom:-3px;width:2px;background:rgba(255,255,255,0.6)'></div>"
        "</div>"
        f"<div style='display:flex;justify-content:space-between;font-size:0.8em;opacity:0.7'>"
        f"<span>◀ {away['short_name']}</span><span>{home['short_name']} ▶</span></div>"
    )

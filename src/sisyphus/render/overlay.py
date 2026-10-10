"""Timed overlay payloads.

A rendering partner in a production path consumes *timed, machine-readable*
instructions rather than prose. Each :class:`OverlayPayload` says what to show,
when (match clock), for how long, with what priority, and for which audience.
Graphics templates (lower third, stat card, gauge, ticker, recap) are deliberately
generic so the same payload can drive broadcast CG, a streaming overlay or a
second-screen app.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from sisyphus.agents.contracts import InsightBundle, StatCallout
from sisyphus.analytics.contracts import MatchSnapshot

OverlayKind = Literal["lower_third", "stat_card", "gauge", "ticker", "recap"]

_DURATION: dict[str, float] = {
    "goal": 12.0,
    "big_chance": 8.0,
    "fatigue_breakpoint": 14.0,
    "press_drop": 12.0,
    "momentum_swing": 10.0,
    "substitution": 9.0,
    "card": 6.0,
    "speed_burst": 5.0,
    "milestone": 6.0,
    "period_summary": 20.0,
    "full_time": 30.0,
}


class OverlayPayload(BaseModel):
    overlay_id: str
    match_id: str
    trigger_id: str
    template: OverlayKind
    persona_id: str
    language: str
    show_at: str = Field(description="Match clock mm:ss (or 45+m:ss) when the graphic should appear")
    period: int
    t: float = Field(description="Seconds since period start, for frame-accurate sync with the feed")
    duration_s: float
    priority: int = Field(ge=1, le=10, description="10 = interrupt everything (goal); 1 = ticker filler")
    title: str
    subtitle: str | None = None
    body: str | None = None
    stats: list[StatCallout] = Field(default_factory=list)
    team_id: str | None = None
    team_color: str | None = None
    player_id: str | None = None
    kind: str
    provenance_mode: str


def _template_for(kind: str, persona_kind: str) -> OverlayKind:
    if kind in ("period_summary", "full_time"):
        return "recap"
    if kind in ("speed_burst", "milestone", "card"):
        return "ticker"
    if kind in ("fatigue_breakpoint", "press_drop", "momentum_swing"):
        return "gauge" if persona_kind == "casual" else "stat_card"
    return "lower_third"


def render_overlays(bundle: InsightBundle, snapshot: MatchSnapshot) -> list[OverlayPayload]:
    """One overlay per persona for a bundle."""
    team = snapshot.team(bundle.team_id) if bundle.team_id else None
    out: list[OverlayPayload] = []
    priority = max(1, min(10, round(1 + 9 * bundle.severity)))
    for p in bundle.personalised:
        persona_kind = "analyst" if p.persona_id.startswith("analyst") or len(p.body) > 300 else "casual"
        out.append(
            OverlayPayload(
                overlay_id=f"{bundle.trigger_id}-{p.persona_id}",
                match_id=bundle.match_id,
                trigger_id=bundle.trigger_id,
                template=_template_for(bundle.kind, persona_kind),
                persona_id=p.persona_id,
                language=p.language,
                show_at=bundle.clock,
                period=bundle.period,
                t=bundle.t,
                duration_s=_DURATION.get(bundle.kind, 8.0),
                priority=priority,
                title=p.headline,
                subtitle=bundle.commentary.broadcast_line if persona_kind == "casual" else bundle.explanation.headline,
                body=p.body,
                stats=p.stat_callouts,
                team_id=bundle.team_id,
                team_color=team.color if team else None,
                player_id=bundle.player_id,
                kind=bundle.kind,
                provenance_mode=bundle.provenance.mode,
            )
        )
    return out


def render_tick_overlay(snapshot: MatchSnapshot) -> dict:
    """Compact per-second payload for persistent on-screen elements (score bug, energy gauges)."""
    return {
        "match_id": snapshot.match_id,
        "clock": snapshot.clock,
        "period": snapshot.period,
        "t": snapshot.t,
        "score": [snapshot.home.score, snapshot.away.score],
        "teams": [
            {
                "team_id": s.team_id,
                "short_name": s.short_name,
                "color": s.color,
                "energy": s.energy.index,
                "energy_trend": s.energy.trend,
                "baseline_ready": s.energy.baseline_ready,
                "possession_pct": s.possession_pct,
                "xg": s.shooting.xg,
                "sprints_5min": s.physical.sprint_count,
                "press_rate": s.physical.press_rate,
            }
            for s in (snapshot.home, snapshot.away)
        ],
        "momentum": snapshot.momentum,
        "ball": list(snapshot.ball_xy),
        "possession_team": snapshot.possession_team,
    }

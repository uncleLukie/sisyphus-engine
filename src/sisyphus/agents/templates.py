"""Deterministic, evidence-grounded text for every trigger kind.

Used (a) when no model is configured, so the whole pipeline runs offline, and
(b) as the recovery path when an agent's output fails validation. Templates
only ever quote numbers from ``trigger.evidence``, which is the same grounding
rule the LLM agents are held to.
"""

from __future__ import annotations

from sisyphus.agents.contracts import (
    Commentary,
    Explanation,
    Persona,
    PersonalisedOutput,
    StatCallout,
)
from sisyphus.analytics.contracts import Trigger


def _ev(t: Trigger, key: str, default="n/a"):
    v = t.evidence.get(key)
    return default if v is None else v


def _pct(x) -> str:
    try:
        return f"{100 * float(x):.0f}%"
    except (TypeError, ValueError):
        return "n/a"


def _trend(team_energy_index: float) -> str:
    return "down" if team_energy_index < 65 else "flat" if team_energy_index < 85 else "up"


# ---------------------------------------------------------------------- explanation
def explanation(t: Trigger) -> Explanation:
    e = t.evidence
    k = t.kind
    if k == "goal":
        xg = _ev(t, "xg", 0)
        build = (
            f"a {int(_ev(t, 'passes_in_buildup', 0))}-pass move lasting {_ev(t, 'buildup_seconds', 0)} s"
            if _ev(t, "passes_in_buildup", 0)
            else "a direct move"
        )
        origin = " that started with a high regain" if e.get("started_from_high_recovery") else ""
        ce = e.get("conceding_team_energy")
        fatigue = (
            f" The conceding side's energy index is {ce:.0f}, so the gap was physical as much as tactical."
            if isinstance(ce, (int, float)) and ce < 65
            else ""
        )
        return Explanation(
            headline=f"{e.get('scorer')} finishes {build}{origin}",
            why_it_matters=(
                f"{e.get('team')} convert a chance worth {xg} xG from {e.get('shot_distance_m')} m"
                f"{' under pressure' if e.get('under_pressure') else ''}; it makes the score {e.get('score')}.{fatigue}"
            ),
            evidence=[
                f"Shot xG {xg} from {e.get('shot_distance_m')} m at {e.get('shot_speed_kmh')} km/h",
                f"Build-up: {int(_ev(t, 'passes_in_buildup', 0))} passes over {_ev(t, 'buildup_seconds', 0)} s",
                f"Energy index: scorers {e.get('scoring_team_energy')}, conceding side {ce}",
            ],
            tactical_label="transition" if (_ev(t, "buildup_seconds", 99) or 99) < 12 else "control",
            confidence=0.9,
        )
    if k == "big_chance":
        return Explanation(
            headline=f"{e.get('shooter')} wastes a {e.get('xg')} xG chance",
            why_it_matters=(
                f"A shot from {e.get('shot_distance_m')} m with an expected-goal value of {e.get('xg')} is the kind of "
                f"opening that decides tight matches; {e.get('team')} will feel this one."
            ),
            evidence=[
                f"xG {e.get('xg')} ({e.get('outcome')})",
                f"Shot speed {e.get('shot_speed_kmh')} km/h from {e.get('shot_distance_m')} m",
                f"Build-up of {int(_ev(t, 'passes_in_buildup', 0))} passes",
            ],
            tactical_label="individual",
            confidence=0.85,
        )
    if k == "fatigue_breakpoint":
        return Explanation(
            headline=f"{e.get('team')}'s legs are going: sprints down {_pct(1 - float(_ev(t, 'sprint_ratio', 1)))}",
            why_it_matters=(
                f"{e.get('team')} are producing {e.get('sprints_last_5min')} sprints per five minutes against "
                f"{e.get('sprints_opening_5min')} in the opening spell, and high-intensity running is at "
                f"{_pct(e.get('hi_distance_ratio'))} of baseline. A pressing structure that arrives late stops "
                f"being a press and becomes space for the opponent."
            ),
            evidence=[
                f"Energy index {e.get('energy_index')} (threshold {e.get('threshold')})",
                f"Sprints last 5 min: {e.get('sprints_last_5min')} vs {e.get('sprints_opening_5min')} early",
                f"Block length {e.get('block_length_m')} m, line at {e.get('defensive_line_m')} m",
                f"Most fatigued: {e.get('most_fatigued')}",
            ],
            tactical_label="fatigue",
            confidence=0.8,
        )
    if k == "press_drop":
        return Explanation(
            headline=f"{e.get('team')} press fading, {e.get('opponent')} finding time on the ball",
            why_it_matters=(
                f"Pressure actions have fallen to {e.get('press_rate_now')} per minute of opponent possession from "
                f"{e.get('press_rate_opening')} early on. With {e.get('opponent')} completing {_pct(e.get('opponent_pass_accuracy'))} "
                f"of passes, the game is tilting from chaos towards control for them."
            ),
            evidence=[
                f"Press rate {e.get('press_rate_now')}/min vs {e.get('press_rate_opening')} opening",
                f"High regains last 5 min: {e.get('high_recoveries_last_5min')}",
                f"Energy index {e.get('energy_index')}",
            ],
            tactical_label="pressure_shift",
            confidence=0.75,
        )
    if k == "momentum_swing":
        return Explanation(
            headline=f"Momentum swings to {e.get('team')}",
            why_it_matters=(
                f"Over the last three minutes {e.get('team')} hold a momentum score of {e.get('momentum')} and are "
                f"playing at {_pct(e.get('territory'))} pitch progress while the opponent's energy index sits at "
                f"{e.get('opponent_energy')} and is {e.get('opponent_trend')}."
            ),
            evidence=[
                f"Momentum {e.get('momentum')}",
                f"Territory {_pct(e.get('territory'))} vs {_pct(e.get('opponent_territory'))}",
                f"Opponent energy {e.get('opponent_energy')} ({e.get('opponent_trend')})",
            ],
            tactical_label="rhythm_change",
            confidence=0.7,
        )
    if k == "substitution":
        oe = e.get("out_energy")
        return Explanation(
            headline=f"{e.get('player_in')} on for {e.get('player_out')}",
            why_it_matters=(
                f"{e.get('player_out')} leaves after {float(_ev(t, 'out_distance_m', 0)) / 1000:.1f} km and "
                f"{e.get('out_sprints')} sprints with a personal energy estimate of {oe}; fresh legs arrive "
                f"while the team index reads {e.get('team_energy')}."
            ),
            evidence=[
                f"{e.get('player_out')}: {float(_ev(t, 'out_distance_m', 0)) / 1000:.1f} km, {e.get('out_sprints')} sprints, energy {oe}",
                f"Team energy index {e.get('team_energy')}",
            ],
            tactical_label="fatigue" if isinstance(oe, (int, float)) and oe < 60 else "neutral",
            confidence=0.8,
        )
    if k == "card":
        return Explanation(
            headline=f"{e.get('player')} booked",
            why_it_matters="A caution changes how aggressively this player can press and tackle for the rest of the match.",
            evidence=[f"{e.get('card')} card for {e.get('player')}"],
            tactical_label="individual",
            confidence=0.9,
        )
    if k == "speed_burst":
        return Explanation(
            headline=f"{e.get('player')} hits {e.get('speed_kmh')} km/h",
            why_it_matters=f"Top speed of the match so far for {e.get('player')}: {e.get('top_speed_kmh')} km/h across {e.get('sprints')} sprints.",
            evidence=[f"{e.get('speed_kmh')} km/h now, {e.get('top_speed_kmh')} km/h best"],
            tactical_label="individual",
            confidence=0.95,
        )
    if k == "milestone":
        if "distance_km" in e:
            return Explanation(
                headline=f"{e.get('player')} first in {e.get('team')} past {e.get('distance_km')} km",
                why_it_matters=f"{e.get('player')} has covered {e.get('distance_km')} km with {e.get('sprints')} sprints, the first {e.get('team')} player to do so; energy estimate {e.get('energy')}.",
                evidence=[
                    f"{e.get('distance_km')} km, {e.get('sprints')} sprints, top speed {e.get('top_speed_kmh')} km/h"
                ],
                tactical_label="individual",
                confidence=0.95,
            )
        return Explanation(
            headline=f"{e.get('team')} reach {e.get('passes')} passes",
            why_it_matters=f"{e.get('team')} have attempted {e.get('passes')} passes at {_pct(e.get('accuracy'))}, including {e.get('hard_completed')} difficult completions.",
            evidence=[
                f"{e.get('passes')} passes, {_pct(e.get('accuracy'))} accuracy, {e.get('progressive')} progressive"
            ],
            tactical_label="control",
            confidence=0.95,
        )
    # period_summary / full_time
    lead = (
        "level"
        if e.get("home_xg") == e.get("away_xg")
        else (e.get("home") if float(_ev(t, "home_xg", 0)) > float(_ev(t, "away_xg", 0)) else e.get("away"))
    )
    return Explanation(
        headline=t.headline,
        why_it_matters=(
            f"{e.get('home')} {_pct(float(_ev(t, 'home_possession', 50)) / 100)} possession, xG {e.get('home_xg')} to {e.get('away_xg')}; "
            f"energy indices {e.get('home_energy')} vs {e.get('away_energy')}. On chances created, {lead} have the edge."
        ),
        evidence=[
            f"Shots {e.get('home_shots')}-{e.get('away_shots')}, xG {e.get('home_xg')}-{e.get('away_xg')}",
            f"Energy {e.get('home_energy')} vs {e.get('away_energy')}",
            f"Pass accuracy {_pct(e.get('home_pass_accuracy'))} vs {_pct(e.get('away_pass_accuracy'))}",
        ],
        tactical_label="neutral",
        confidence=0.85,
    )


# ---------------------------------------------------------------------- commentary
def commentary(t: Trigger, x: Explanation) -> Commentary:
    e = t.evidence
    k = t.kind
    if k == "goal":
        line = (
            f"{e.get('scorer')} scores! {e.get('team')} lead"
            if _leading(t)
            else f"{e.get('scorer')} scores for {e.get('team')}"
        )
        line += f", {e.get('score')}."
    elif k == "big_chance":
        line = f"What a chance for {e.get('shooter')} - {str(e.get('outcome')).replace('_', ' ')} from {e.get('shot_distance_m')} metres."
    elif k == "fatigue_breakpoint":
        line = f"{e.get('team')} are tiring: {e.get('sprints_last_5min')} sprints in the last five minutes, down from {e.get('sprints_opening_5min')}."
    elif k == "press_drop":
        line = f"{e.get('team')}'s press has dropped off and {e.get('opponent')} are starting to pass through it."
    elif k == "momentum_swing":
        line = f"The tide is turning - {e.get('team')} have the momentum now."
    elif k == "substitution":
        line = f"{e.get('player_in')} replaces {e.get('player_out')} - fresh legs for {e.get('team')}."
    elif k == "speed_burst":
        line = f"{e.get('player')} clocks {e.get('speed_kmh')} kilometres an hour!"
    elif k == "milestone":
        line = x.headline + "."
    elif k == "card":
        line = f"Yellow card for {e.get('player')}."
    else:
        line = t.headline + "."
    return Commentary(broadcast_line=line[:200], studio_note=x.why_it_matters[:400])


def _leading(t: Trigger) -> bool:
    try:
        h, a = (int(v) for v in str(t.evidence.get("score", "0-0")).split("-"))
    except ValueError:
        return False
    is_home = t.team_id == t.snapshot.home.team_id
    return (h > a) if is_home else (a > h)


# ---------------------------------------------------------------------- personalisation
def personalise(t: Trigger, x: Explanation, c: Commentary, persona: Persona) -> PersonalisedOutput:
    s = t.snapshot
    team = s.team(t.team_id) if t.team_id else None
    callouts: list[StatCallout] = []
    if persona.kind == "analyst":
        if team is not None:
            callouts = [
                StatCallout(
                    label=f"{team.short_name} energy", value=f"{team.energy.index:.0f}", trend=_trend(team.energy.index)
                ),
                StatCallout(label="Sprints (5 min)", value=str(team.physical.sprint_count)),
                StatCallout(label="Press rate", value=f"{team.physical.press_rate:.1f}", unit="/min"),
                StatCallout(label="xG", value=f"{team.shooting.xg:.2f}"),
            ]
        else:
            callouts = [
                StatCallout(
                    label=f"{s.home.short_name} energy",
                    value=f"{s.home.energy.index:.0f}",
                    trend=_trend(s.home.energy.index),
                ),
                StatCallout(
                    label=f"{s.away.short_name} energy",
                    value=f"{s.away.energy.index:.0f}",
                    trend=_trend(s.away.energy.index),
                ),
                StatCallout(label="Momentum", value=f"{s.momentum:+.2f}"),
            ]
        body = f"{x.why_it_matters} " + " ".join(f"{b}." for b in x.evidence[:3])
        headline = x.headline
    else:
        headline = c.broadcast_line.rstrip(".!")
        body = _casual_body(t, x)
        if team is not None:
            callouts = [
                StatCallout(label="Energy", value=f"{team.energy.index:.0f}/100", trend=_trend(team.energy.index))
            ]
    if persona.favourite_player and persona.favourite_player.lower() in (body + headline).lower():
        headline = f"Your player: {headline}"
    if persona.favourite_team and team is not None and persona.favourite_team.lower() in team.name.lower():
        headline = f"{team.short_name} update - {headline}"
    return PersonalisedOutput(
        persona_id=persona.persona_id,
        language="en",
        headline=headline[:120],
        body=body[:700],
        stat_callouts=callouts[:4],
    )


def _casual_body(t: Trigger, x: Explanation) -> str:
    e = t.evidence
    k = t.kind
    if k == "fatigue_breakpoint":
        return (
            f"{e.get('team')} started like a train but the tank is emptying - they're sprinting about "
            f"{_pct(1 - float(_ev(t, 'sprint_ratio', 1)))} less than at kick-off. Watch for gaps opening up."
        )
    if k == "press_drop":
        return f"{e.get('team')} aren't chasing the ball like before, so {e.get('opponent')} can finally breathe and build attacks."
    if k == "goal":
        return f"{e.get('scorer')} makes it {e.get('score')} - a {('tough' if float(_ev(t, 'xg', 0) or 0) < 0.15 else 'good')} chance taken from {e.get('shot_distance_m')} m out."
    if k == "momentum_swing":
        return f"{e.get('team')} are on top right now - the ball keeps ending up in the other half."
    if k == "substitution":
        return f"{e.get('player_out')} has run {float(_ev(t, 'out_distance_m', 0)) / 1000:.1f} km and looks done; {e.get('player_in')} comes on fresh."
    return x.why_it_matters

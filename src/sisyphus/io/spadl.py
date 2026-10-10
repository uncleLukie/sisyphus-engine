"""Export the live event stream as a SPADL-style action table.

SPADL (Soccer Player Action Description Language, Decroos et al. 2019) is the
common denominator used by ``socceraction`` and most open football analytics
tooling. Following its conventions every action is expressed from the acting
team's perspective (playing left-to-right on a 105 x 68 pitch), which makes the
generated data drop-in compatible with existing VAEP / xT code.

Columns: game_id, period_id, time_seconds, team_id, player_id, start_x, start_y,
end_x, end_y, type_name, result_name, bodypart_name, action_id.
"""

from __future__ import annotations

import pandas as pd

from sisyphus.schema import Event, EventType, MatchMeta
from sisyphus.sim import pitch

_TYPE_MAP: dict[EventType, str] = {
    EventType.PASS: "pass",
    EventType.SHOT: "shot",
    EventType.TACKLE: "tackle",
    EventType.INTERCEPTION: "interception",
    EventType.FOUL: "foul",
    EventType.KEEPER_SAVE: "keeper_save",
    EventType.BALL_RECOVERY: "take_on",  # closest SPADL analogue for a loose-ball regain
}
_RESTART_MAP = {
    "throw_in": "throw_in",
    "corner": "corner_crossed",
    "goal_kick": "goalkick",
    "free_kick": "freekick_short",
    "kickoff": "pass",
}


def _flip(x: float | None, y: float | None, team_is_away: bool) -> tuple[float | None, float | None]:
    if x is None or y is None:
        return x, y
    if team_is_away:
        return round(pitch.LENGTH - x, 2), round(pitch.WIDTH - y, 2)
    return x, y


def events_to_spadl(events: list[Event], meta: MatchMeta) -> pd.DataFrame:
    """Convert resolved on-ball events to SPADL actions."""
    rows = []
    away_id = meta.away.team_id
    action_id = 0
    for ev in events:
        if ev.type not in _TYPE_MAP or ev.team_id is None:
            continue
        type_name = _TYPE_MAP[ev.type]
        flip = ev.team_id == away_id
        sx, sy = _flip(ev.start_x, ev.start_y, flip)
        ex, ey = _flip(ev.end_x, ev.end_y, flip)
        if ex is None:
            ex, ey = sx, sy
        if ev.type == EventType.PASS and ev.restart:
            type_name = _RESTART_MAP.get(ev.restart, "pass")
        elif ev.type == EventType.PASS and ev.qualifiers.get("lofted"):
            in_box = ex is not None and ex >= pitch.LENGTH - pitch.PENALTY_AREA_DEPTH
            type_name = "cross" if in_box else "pass"
        if ev.type == EventType.SHOT and ev.restart == "free_kick":
            type_name = "shot_freekick"
        if ev.type == EventType.FOUL:
            result = "fail"
        elif ev.type == EventType.SHOT:
            result = "success" if ev.outcome == "success" else "fail"
        else:
            result = ev.outcome or "success"
        rows.append(
            {
                "game_id": meta.match_id,
                "action_id": action_id,
                "period_id": ev.period,
                "time_seconds": ev.t,
                "team_id": ev.team_id,
                "player_id": ev.player_id,
                "start_x": sx,
                "start_y": sy,
                "end_x": ex,
                "end_y": ey,
                "type_name": type_name,
                "result_name": result,
                "bodypart_name": "foot",
                "xg": ev.xg,
                "pass_success_prob": ev.success_prob,
                "under_pressure": ev.under_pressure,
            }
        )
        action_id += 1
    return pd.DataFrame(rows)

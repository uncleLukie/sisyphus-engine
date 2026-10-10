"""Pass quality metrics estimated from observed event data.

The simulator has its own generative completion model; this module deliberately
does not use it. Difficulty is a transparent logistic score built from features
that any event feed exposes (length, direction, zone, pressure), so the same code
would work on real provider data.
"""

from __future__ import annotations

import math

from sisyphus.schema import Event
from sisyphus.sim import pitch


def _team_frame_x(x: float, team_is_home: bool) -> float:
    return x / pitch.LENGTH if team_is_home else 1.0 - x / pitch.LENGTH


def pass_features(ev: Event, team_is_home: bool) -> dict[str, float]:
    sx, sy, ex, ey = ev.start_x or 0.0, ev.start_y or 0.0, ev.end_x or 0.0, ev.end_y or 0.0
    dist = ev.pass_distance_m if ev.pass_distance_m is not None else math.hypot(ex - sx, ey - sy)
    progress_m = (_team_frame_x(ex, team_is_home) - _team_frame_x(sx, team_is_home)) * pitch.LENGTH
    end_tf = _team_frame_x(ex, team_is_home)
    return {
        "distance_m": float(dist),
        "progress_m": float(progress_m),
        "into_final_third": float(end_tf >= 2 / 3),
        "into_box": float(
            end_tf >= 1 - pitch.PENALTY_AREA_DEPTH / pitch.LENGTH
            and abs(ey - pitch.WIDTH / 2) < pitch.PENALTY_AREA_HALF_WIDTH
        ),
        "under_pressure": float(bool(ev.under_pressure)),
        "lofted": float(bool(ev.qualifiers.get("lofted"))),
    }


def pass_difficulty(features: dict[str, float]) -> float:
    """Probability-like difficulty in [0, 1]; higher is harder."""
    z = (
        -2.4
        + 0.045 * features["distance_m"]
        + 0.035 * max(features["progress_m"], 0.0)
        + 0.6 * features["into_final_third"]
        + 0.9 * features["into_box"]
        + 0.8 * features["under_pressure"]
        + 0.3 * features["lofted"]
    )
    return 1.0 / (1.0 + math.exp(-z))


def describe_difficulty(d: float) -> str:
    if d >= 0.7:
        return "very hard"
    if d >= 0.5:
        return "hard"
    if d >= 0.3:
        return "moderate"
    return "routine"

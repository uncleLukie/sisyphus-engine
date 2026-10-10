"""Probability models used by on-ball decisions.

These are deliberately simple, interpretable logistic models. They are the
simulator's *generative* truth; the analytics layer re-estimates quantities
such as pass difficulty independently from the observed data.
"""

from __future__ import annotations

import math

import numpy as np

from sisyphus.sim import pitch


def sigmoid(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-z))


def expected_goals(xy: np.ndarray, team: int, pressure: float, finishing: float) -> float:
    """Shot conversion probability from distance, goal angle and pressure."""
    dist = float(np.linalg.norm(pitch.attacking_goal(team) - xy))
    angle = pitch.goal_angle(xy, team)
    z = -1.35 + 1.45 * angle - 0.105 * dist - 0.55 * pressure + 0.6 * (finishing - 0.5)
    return sigmoid(z)


def pass_success_prob(
    distance: float,
    lane_clearance: float,
    receiver_space: float,
    pressure: float,
    passing: float,
    e_short: float,
    e_long: float,
) -> float:
    """Probability a pass reaches its target.

    Longer passes, crowded lanes, a marked receiver, pressure on the passer and
    fatigue all reduce completion probability.
    """
    z = (
        0.85
        + 1.6 * passing
        - 0.048 * distance
        - 1.0 * pressure
        + 0.26 * min(lane_clearance, 6.0)
        - 0.5 * (lane_clearance < 1.5)
        + 0.11 * min(receiver_space, 8.0)
        - 0.6 * (1.0 - e_short)
        - 1.1 * (1.0 - e_long)
    )
    return sigmoid(z)


def possession_value(tf: np.ndarray) -> float:
    """Value of holding the ball at a team-frame location (xT-like surface)."""
    x, y = float(tf[0]), float(tf[1])
    centrality = 1.0 - 0.7 * (2.0 * abs(y - 0.5)) ** 2
    return 0.01 + 0.03 * x + 0.38 * max(0.0, x - 0.6) ** 2 / 0.16 * centrality


def turnover_cost(tf: np.ndarray) -> float:
    """Cost of losing the ball at a team-frame location (opponent's value there)."""
    return possession_value(1.0 - np.asarray(tf))

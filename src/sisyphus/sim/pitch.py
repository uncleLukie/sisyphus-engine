"""Pitch geometry and coordinate helpers.

Raw coordinates are metres on a 105 x 68 pitch (origin bottom-left). Team 0
(home) attacks towards ``x = LENGTH``; team 1 (away) attacks towards ``x = 0``.

A *team frame* is a normalised, team-relative coordinate system in [0, 1]^2
where ``x = 1`` is the opponent's goal line. Converting between frames is a
180 degree rotation for the away team, which preserves left/right handedness.
"""

from __future__ import annotations

import numpy as np

LENGTH = 105.0
WIDTH = 68.0
GOAL_WIDTH = 7.32
GOAL_Y_MIN = WIDTH / 2 - GOAL_WIDTH / 2
GOAL_Y_MAX = WIDTH / 2 + GOAL_WIDTH / 2
PENALTY_AREA_DEPTH = 16.5
PENALTY_AREA_HALF_WIDTH = 20.16

FPS = 10
DT = 1.0 / FPS

_SCALE = np.array([LENGTH, WIDTH])


def to_team_frame(xy: np.ndarray, team: int) -> np.ndarray:
    """Convert pitch metres to the normalised frame of ``team`` (0 home, 1 away)."""
    norm = np.asarray(xy, dtype=float) / _SCALE
    return norm if team == 0 else 1.0 - norm


def from_team_frame(tf: np.ndarray, team: int) -> np.ndarray:
    """Convert a normalised team-frame position back to pitch metres."""
    tf = np.asarray(tf, dtype=float)
    norm = tf if team == 0 else 1.0 - tf
    return norm * _SCALE


def attacking_goal(team: int) -> np.ndarray:
    """Centre of the goal that ``team`` attacks, in pitch metres."""
    return np.array([LENGTH if team == 0 else 0.0, WIDTH / 2])


def defending_goal(team: int) -> np.ndarray:
    """Centre of the goal that ``team`` defends, in pitch metres."""
    return attacking_goal(1 - team)


def in_bounds(xy: np.ndarray) -> bool:
    """Whether a point lies on the field of play (lines included)."""
    return 0.0 <= xy[0] <= LENGTH and 0.0 <= xy[1] <= WIDTH


def goal_angle(xy: np.ndarray, team: int) -> float:
    """Angle (radians) subtended by the goal mouth that ``team`` attacks."""
    gx = attacking_goal(team)[0]
    a = np.array([gx, GOAL_Y_MIN]) - xy
    b = np.array([gx, GOAL_Y_MAX]) - xy
    cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
    return float(np.arccos(np.clip(cos, -1.0, 1.0)))


def segment_distances(points: np.ndarray, a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Distance from each point to segment ``a -> b`` and the projection parameter ``t``."""
    ab = b - a
    denom = float(np.dot(ab, ab)) or 1e-9
    t = np.clip(((points - a) @ ab) / denom, 0.0, 1.0)
    proj = a + t[:, None] * ab
    return np.linalg.norm(points - proj, axis=1), t

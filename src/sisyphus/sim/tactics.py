"""Formations and team-shape behaviour.

Formations are authored once in the normalised team frame (``x = 1`` is the
opponent's goal, ``y = 1`` is the team's left touchline when attacking). Slot
order matters: slot 0 is always the goalkeeper and starters in
:mod:`sisyphus.sim.roster` are listed in the same order.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

FORMATIONS: dict[str, np.ndarray] = {
    # GK, RB, RCB, LCB, LB, DM, RCM, LCM, RW, ST, LW
    "4-3-3": np.array(
        [
            [0.04, 0.50],
            [0.24, 0.12],
            [0.20, 0.37],
            [0.20, 0.63],
            [0.24, 0.88],
            [0.36, 0.50],
            [0.48, 0.30],
            [0.48, 0.70],
            [0.70, 0.14],
            [0.76, 0.50],
            [0.70, 0.86],
        ]
    ),
    # GK, RB, RCB, LCB, LB, RM, RCM, LCM, LM, RS, LS
    "4-4-2": np.array(
        [
            [0.04, 0.50],
            [0.23, 0.12],
            [0.20, 0.38],
            [0.20, 0.62],
            [0.23, 0.88],
            [0.47, 0.13],
            [0.43, 0.39],
            [0.43, 0.61],
            [0.47, 0.87],
            [0.72, 0.42],
            [0.72, 0.58],
        ]
    ),
}

_DEPTH_MIN, _DEPTH_MAX = 0.2, 0.76


@dataclass(frozen=True)
class TeamStyle:
    """Tactical parameters for a team's shape, pressing and tempo."""

    name: str
    press_players: int  # max simultaneous pressers on the ball carrier
    press_zone: tuple[float, float]  # ball x (own team frame) range where pressing triggers
    block_gap: float  # distance (team frame) the defensive line sits behind the ball
    min_line: float
    max_line: float
    defend_length: float
    defend_width: float
    attack_length: float
    attack_width: float
    work_rate: float  # multiplier on long-term energy drain from pressing intensity
    tempo: float  # >1 means quicker on-ball decisions
    directness: float  # 0..1 preference for forward / long passes


HIGH_PRESS = TeamStyle(
    name="high_press",
    press_players=2,
    press_zone=(0.25, 1.0),
    block_gap=0.34,
    min_line=0.24,
    max_line=0.56,
    defend_length=0.30,
    defend_width=0.66,
    attack_length=0.50,
    attack_width=0.92,
    work_rate=1.18,
    tempo=1.15,
    directness=0.35,
)

LOW_BLOCK_COUNTER = TeamStyle(
    name="low_block_counter",
    press_players=1,
    press_zone=(0.0, 0.42),
    block_gap=0.42,
    min_line=0.13,
    max_line=0.34,
    defend_length=0.28,
    defend_width=0.60,
    attack_length=0.52,
    attack_width=0.90,
    work_rate=0.95,
    tempo=1.0,
    directness=0.7,
)

STYLES: dict[str, TeamStyle] = {s.name: s for s in (HIGH_PRESS, LOW_BLOCK_COUNTER)}


def shape_targets(
    anchors: np.ndarray, in_possession: bool, ball_tf: np.ndarray, style: TeamStyle
) -> np.ndarray:
    """Target positions (team frame) for all 11 slots given the ball position.

    The block slides with the ball: its back line sits ``block_gap`` behind the
    ball (clamped by the style's line limits) and outfield anchors are mapped
    into ``[back, back + length]``. In possession the block is longer and wider.
    """
    depth = np.clip((anchors[:, 0] - _DEPTH_MIN) / (_DEPTH_MAX - _DEPTH_MIN), 0.0, 1.0)
    bx, by = float(ball_tf[0]), float(ball_tf[1])
    if in_possession:
        back = float(np.clip(bx - 0.42, 0.12, 0.5))
        length, width, shift = style.attack_length, style.attack_width, 0.12
    else:
        back = float(np.clip(bx - style.block_gap, style.min_line, style.max_line))
        length, width, shift = style.defend_length, style.defend_width, 0.28

    out = np.empty_like(anchors)
    out[:, 0] = back + depth * length
    out[:, 1] = 0.5 + (anchors[:, 1] - 0.5) * width + shift * (by - 0.5)
    # Goalkeeper: hold the line, drift with the ball.
    out[0, 0] = 0.025 + 0.06 * np.clip(bx, 0.0, 1.0)
    out[0, 1] = 0.5 + 0.22 * (by - 0.5)
    return np.clip(out, [0.01, 0.03], [0.99, 0.97])


def kickoff_targets(anchors: np.ndarray) -> np.ndarray:
    """Kick-off positions (team frame): everyone in their own half."""
    out = anchors.copy()
    out[:, 0] = np.minimum(anchors[:, 0] * 0.62, 0.47)
    out[0, 0] = 0.03
    return out

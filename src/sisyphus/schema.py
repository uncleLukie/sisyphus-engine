"""Canonical data contracts shared by the simulator, stats engine, agents and UI.

Coordinate system
-----------------
All raw coordinates are in metres on a 105 x 68 pitch with the origin at the
bottom-left corner. The home team always attacks towards ``x = 105`` (no side
switch at half time), which keeps downstream maths simple. The SPADL export
re-normalises every action so the acting team plays left-to-right, matching
the socceraction convention.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field

SCHEMA_VERSION = "0.1.0"


class EventType(StrEnum):
    """Event vocabulary emitted on the live match stream."""

    PERIOD_START = "period_start"
    PERIOD_END = "period_end"
    PASS = "pass"
    SHOT = "shot"
    GOAL = "goal"
    KEEPER_SAVE = "keeper_save"
    TACKLE = "tackle"
    INTERCEPTION = "interception"
    PRESSURE = "pressure"
    BALL_RECOVERY = "ball_recovery"
    FOUL = "foul"
    CARD = "card"
    BALL_OUT = "ball_out"
    POSSESSION_CHANGE = "possession_change"
    SUBSTITUTION = "substitution"


Outcome = Literal["success", "fail"]
RestartKind = Literal["kickoff", "throw_in", "goal_kick", "corner", "free_kick"]
Position = Literal["GK", "DF", "MF", "FW"]


class Event(BaseModel):
    """A single on-ball or match-state event.

    Events are emitted when they are *resolved* (e.g. a pass is emitted when it
    is received or intercepted), so ``frame`` is the emission frame while ``t``
    is the time the action started, measured in seconds since period start.
    """

    schema_version: str = SCHEMA_VERSION
    event_id: int
    match_id: str
    frame: int
    period: int
    t: float
    t_end: float | None = None
    type: EventType
    team_id: str | None = None
    player_id: str | None = None
    receiver_id: str | None = None
    outcome: Outcome | None = None
    start_x: float | None = None
    start_y: float | None = None
    end_x: float | None = None
    end_y: float | None = None
    possession_id: int | None = None
    restart: RestartKind | None = None
    under_pressure: bool | None = None
    pass_distance_m: float | None = None
    ball_speed_mps: float | None = None
    xg: float | None = None
    success_prob: float | None = None
    qualifiers: dict[str, str | float | int | bool] = Field(default_factory=dict)


class PlayerInfo(BaseModel):
    """Static roster information for a (fictional) player."""

    player_id: str
    team_id: str
    name: str
    short_name: str
    number: int
    position: Position
    starter: bool


class TeamInfo(BaseModel):
    """Static information for a (fictional) club."""

    team_id: str
    name: str
    short_name: str
    primary_color: str
    secondary_color: str
    formation: str
    style: str
    players: list[PlayerInfo]


class MatchMeta(BaseModel):
    """Match-level metadata written alongside generated data."""

    schema_version: str = SCHEMA_VERSION
    match_id: str
    scenario_id: str
    scenario_title: str
    seed: int
    fps: int
    pitch_length: float
    pitch_width: float
    home: TeamInfo
    away: TeamInfo
    final_score: tuple[int, int] | None = None
    period_lengths_s: list[float] = Field(default_factory=list)
    generator: str = "sisyphus-sim"

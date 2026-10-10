"""Typed outputs of the Interpret stage.

Everything an agent, overlay renderer or dashboard needs is in these models, so
the downstream stages never touch raw tracking data.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

TriggerKind = Literal[
    "goal",
    "big_chance",
    "fatigue_breakpoint",
    "press_drop",
    "momentum_swing",
    "substitution",
    "card",
    "milestone",
    "speed_burst",
    "period_summary",
    "full_time",
]


class TeamPhysical(BaseModel):
    """Rolling-window physical output of a team, derived purely from tracking."""

    team_id: str
    window_s: float
    distance_m: float
    hi_distance_m: float = Field(description="Distance covered above 5.5 m/s")
    sprint_count: int = Field(description="Bouts above 7.0 m/s lasting >= 0.5 s")
    mean_speed_mps: float
    accel_count: int = Field(description="Accelerations above 3 m/s^2")
    compactness_m2: float = Field(description="Mean convex-hull area of the 10 outfield players")
    block_length_m: float = Field(description="Mean distance between deepest and highest outfielder")
    defensive_line_m: float = Field(description="Mean height of the back line (m from own goal) out of possession")
    press_rate: float = Field(description="Pressure events per minute of opponent possession")
    high_recoveries: int = Field(description="Ball regains in the opponent's final 40%")


class TeamEnergy(BaseModel):
    """Team Energy Index: current output relative to the team's own opening baseline."""

    team_id: str
    index: float = Field(ge=0, le=100)
    sprint_ratio: float
    hi_distance_ratio: float
    speed_ratio: float
    press_ratio: float
    trend: Literal["rising", "stable", "falling"]
    baseline_ready: bool


class TeamPassing(BaseModel):
    team_id: str
    attempted: int
    completed: int
    accuracy: float
    mean_distance_m: float
    mean_difficulty: float = Field(description="Mean estimated pass difficulty 0..1")
    hard_completed: int = Field(description="Completed passes with difficulty >= 0.5")
    progressive: int = Field(description="Completed passes gaining >= 10 m towards goal")
    fastest_pass_kmh: float


class TeamShooting(BaseModel):
    team_id: str
    shots: int
    on_target: int
    goals: int
    xg: float
    fastest_shot_kmh: float


class PlayerPhysical(BaseModel):
    player_id: str
    name: str
    team_id: str
    number: int
    position: str
    on_pitch: bool
    distance_m: float
    hi_distance_m: float
    sprints: int
    top_speed_kmh: float
    current_speed_kmh: float
    window_sprints: int
    window_distance_m: float
    energy: float = Field(ge=0, le=100, description="Player energy estimate vs own opening baseline")


class TeamSnapshot(BaseModel):
    team_id: str
    name: str
    short_name: str
    color: str
    score: int
    possession_pct: float
    physical: TeamPhysical
    energy: TeamEnergy
    passing: TeamPassing
    shooting: TeamShooting
    territory: float = Field(description="Mean ball progress (0..1 towards opponent goal) while in possession")


class MatchSnapshot(BaseModel):
    """The complete interpreted match state at an instant."""

    match_id: str
    period: int
    t: float
    clock: str
    minute: float
    home: TeamSnapshot
    away: TeamSnapshot
    momentum: float = Field(ge=-1, le=1, description="+1 all home threat, -1 all away threat (3-min EMA)")
    ball_xy: tuple[float, float]
    possession_team: str | None
    players: list[PlayerPhysical]

    def team(self, team_id: str) -> TeamSnapshot:
        return self.home if self.home.team_id == team_id else self.away

    def opponent(self, team_id: str) -> TeamSnapshot:
        return self.away if self.home.team_id == team_id else self.home


class Trigger(BaseModel):
    """A moment the Explain stage should react to, with the evidence behind it."""

    trigger_id: str
    match_id: str
    kind: TriggerKind
    period: int
    t: float
    clock: str
    minute: float
    team_id: str | None = None
    player_id: str | None = None
    severity: float = Field(ge=0, le=1)
    headline: str
    evidence: dict[str, float | int | str | bool | None] = Field(default_factory=dict)
    snapshot: MatchSnapshot

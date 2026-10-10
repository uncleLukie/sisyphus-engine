"""Fictional clubs and players.

All names are invented to keep submissions free of third-party trademarks and
real-person likeness. Attributes are generated deterministically from a fixed
per-team seed so every run of a scenario uses identical squads.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sisyphus.schema import PlayerInfo, Position


@dataclass(frozen=True)
class PlayerProfile:
    """Physical and technical attributes of a simulated player."""

    player_id: str
    team_id: str
    name: str
    number: int
    position: Position
    top_speed: float  # m/s
    acceleration: float  # m/s^2
    endurance: float  # >1 drains long-term energy more slowly
    passing: float  # 0..1
    finishing: float  # 0..1
    tackling: float  # 0..1
    dribbling: float  # 0..1

    @property
    def short_name(self) -> str:
        return self.name.split()[-1]

    def info(self, starter: bool) -> PlayerInfo:
        return PlayerInfo(
            player_id=self.player_id,
            team_id=self.team_id,
            name=self.name,
            short_name=self.short_name,
            number=self.number,
            position=self.position,
            starter=starter,
        )


@dataclass(frozen=True)
class TeamProfile:
    """A club with an ordered starting XI (matching formation slots) and a bench."""

    team_id: str
    name: str
    short_name: str
    primary_color: str
    secondary_color: str
    formation: str
    starters: tuple[PlayerProfile, ...]
    bench: tuple[PlayerProfile, ...]

    def player(self, number: int) -> PlayerProfile:
        for p in (*self.starters, *self.bench):
            if p.number == number:
                return p
        raise KeyError(f"{self.team_id} has no player #{number}")


# Mean attribute values by position: speed, accel, endurance, pass, finish, tackle, dribble
_POSITION_MEANS: dict[str, tuple[float, ...]] = {
    "GK": (7.6, 5.0, 1.10, 0.55, 0.10, 0.20, 0.20),
    "DF": (9.0, 6.0, 1.00, 0.62, 0.25, 0.72, 0.45),
    "MF": (9.1, 6.3, 1.05, 0.75, 0.45, 0.58, 0.62),
    "FW": (9.5, 6.6, 0.95, 0.66, 0.72, 0.35, 0.72),
}
_ATTR_SPREAD = (0.35, 0.40, 0.06, 0.07, 0.08, 0.07, 0.08)


def _make_player(
    rng: np.random.Generator,
    team_id: str,
    number: int,
    name: str,
    position: Position,
    overrides: dict[str, float] | None = None,
) -> PlayerProfile:
    means = _POSITION_MEANS[position]
    vals = [float(m + s * rng.standard_normal()) for m, s in zip(means, _ATTR_SPREAD, strict=True)]
    attrs = dict(
        top_speed=round(vals[0], 2),
        acceleration=round(vals[1], 2),
        endurance=round(float(np.clip(vals[2], 0.85, 1.2)), 3),
        passing=round(float(np.clip(vals[3], 0.3, 0.95)), 3),
        finishing=round(float(np.clip(vals[4], 0.05, 0.95)), 3),
        tackling=round(float(np.clip(vals[5], 0.1, 0.95)), 3),
        dribbling=round(float(np.clip(vals[6], 0.1, 0.95)), 3),
    )
    attrs.update(overrides or {})
    return PlayerProfile(
        player_id=f"{team_id}-{number:02d}",
        team_id=team_id,
        name=name,
        number=number,
        position=position,
        **attrs,
    )


def _build_team(
    seed: int,
    team_id: str,
    name: str,
    short_name: str,
    colors: tuple[str, str],
    formation: str,
    starters: list[tuple[int, str, Position]],
    bench: list[tuple[int, str, Position]],
    overrides: dict[int, dict[str, float]] | None = None,
) -> TeamProfile:
    rng = np.random.default_rng(seed)
    overrides = overrides or {}

    def build(rows: list[tuple[int, str, Position]]) -> tuple[PlayerProfile, ...]:
        return tuple(_make_player(rng, team_id, num, nm, pos, overrides.get(num)) for num, nm, pos in rows)

    return TeamProfile(
        team_id=team_id,
        name=name,
        short_name=short_name,
        primary_color=colors[0],
        secondary_color=colors[1],
        formation=formation,
        starters=build(starters),
        bench=build(bench),
    )


# Starters are listed in formation-slot order (see tactics.FORMATIONS).
HARBOURSIDE = _build_team(
    seed=1101,
    team_id="HAR",
    name="Harbourside Athletic",
    short_name="Harbourside",
    colors=("#d7263d", "#f4f4f4"),
    formation="4-3-3",
    starters=[
        (1, "Elian Marsh", "GK"),
        (2, "Teo Ballard", "DF"),
        (5, "Kofi Arden", "DF"),
        (6, "Mats Lindqvist", "DF"),
        (3, "Rui Calloway", "DF"),
        (4, "Sami Dervish", "MF"),
        (8, "Jonah Pryce", "MF"),
        (10, "Luca Fenwick", "MF"),
        (7, "Idris Vale", "FW"),
        (9, "Callum Stroud", "FW"),
        (11, "Nico Amari", "FW"),
    ],
    bench=[
        (13, "Oskar Brandt", "GK"),
        (14, "Remy Tolliver", "MF"),
        (15, "Ade Whitlock", "DF"),
        (17, "Yusuf Kerrigan", "FW"),
        (18, "Pip Lazaro", "MF"),
    ],
)

KINGSMERE = _build_team(
    seed=2202,
    team_id="KMR",
    name="Kingsmere Rovers",
    short_name="Kingsmere",
    colors=("#2f6df6", "#f2c14e"),
    formation="4-4-2",
    starters=[
        (1, "Viktor Hale", "GK"),
        (2, "Bram Osterlund", "DF"),
        (4, "Gideon Rourke", "DF"),
        (5, "Henrik Moss", "DF"),
        (3, "Dani Quill", "DF"),
        (7, "Marco Selby", "MF"),
        (6, "Aaron Thistle", "MF"),
        (8, "Ezra Kincaid", "MF"),
        (11, "Leo Saranda", "MF"),
        (9, "Tariq Bellweather", "FW"),
        (10, "Sol Ashdown", "FW"),
    ],
    bench=[
        (12, "Jules Penrose", "GK"),
        (14, "Kai Montrose", "FW"),
        (15, "Ollie Fairbanks", "DF"),
        (16, "Nuno Clement", "MF"),
        (19, "Ravi Lockhart", "FW"),
    ],
    # Kai Montrose is the rapid impact substitute the scenario relies on.
    overrides={14: {"top_speed": 10.2, "acceleration": 7.2, "dribbling": 0.82, "finishing": 0.74}},
)

TEAMS: dict[str, TeamProfile] = {t.team_id: t for t in (HARBOURSIDE, KINGSMERE)}

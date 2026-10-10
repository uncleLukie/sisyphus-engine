"""Scenario definitions: who plays, how they play, and scripted interventions."""

from __future__ import annotations

from dataclasses import dataclass, field

from sisyphus.sim.roster import HARBOURSIDE, KINGSMERE, TeamProfile
from sisyphus.sim.tactics import HIGH_PRESS, LOW_BLOCK_COUNTER, TeamStyle


@dataclass(frozen=True)
class SubstitutionPlan:
    """A scripted substitution, executed at the first stoppage after ``minute``."""

    minute: float
    team_id: str
    out_number: int
    in_number: int


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    title: str
    description: str
    home: TeamProfile
    away: TeamProfile
    home_style: TeamStyle
    away_style: TeamStyle
    substitutions: tuple[SubstitutionPlan, ...] = field(default_factory=tuple)
    default_seed: int = 7
    period_seconds: float = 45 * 60


PRESS_COLLAPSE = Scenario(
    scenario_id="press_collapse",
    title="The Boulder Rolls Back",
    description=(
        "Harbourside Athletic open with a relentless high press against Kingsmere "
        "Rovers' compact low block. The press wins the ball high early, but its "
        "energy cost compounds; after the hour the pressers arrive late, the block "
        "stretches and Kingsmere's counter-attacks, sharpened by the rapid Kai "
        "Montrose off the bench, start to find space."
    ),
    home=HARBOURSIDE,
    away=KINGSMERE,
    home_style=HIGH_PRESS,
    away_style=LOW_BLOCK_COUNTER,
    substitutions=(
        SubstitutionPlan(minute=62, team_id="KMR", out_number=11, in_number=14),
        SubstitutionPlan(minute=70, team_id="HAR", out_number=8, in_number=14),
        SubstitutionPlan(minute=78, team_id="KMR", out_number=10, in_number=19),
    ),
    default_seed=7,
)

MIRROR_MATCH = Scenario(
    scenario_id="mirror_match",
    title="Both Teams Press",
    description=(
        "A control scenario: both sides press high. With no low block to hide "
        "behind, fatigue hits whichever squad has the thinner stamina profile first, "
        "and the analytics must find the breakpoint without a scripted narrative."
    ),
    home=KINGSMERE,
    away=HARBOURSIDE,
    home_style=HIGH_PRESS,
    away_style=HIGH_PRESS,
    substitutions=(
        SubstitutionPlan(minute=66, team_id="HAR", out_number=8, in_number=14),
        SubstitutionPlan(minute=72, team_id="KMR", out_number=11, in_number=14),
    ),
    default_seed=11,
)

SCENARIOS: dict[str, Scenario] = {s.scenario_id: s for s in (PRESS_COLLAPSE, MIRROR_MATCH)}


def get_scenario(scenario_id: str) -> Scenario:
    try:
        return SCENARIOS[scenario_id]
    except KeyError as exc:
        raise KeyError(f"Unknown scenario {scenario_id!r}; choose from {sorted(SCENARIOS)}") from exc

"""Shared fixtures: a short (2 x 3 minute) synthetic match generated once per session."""

from __future__ import annotations

import dataclasses
import os
from pathlib import Path

import pytest

os.environ["SISYPHUS_FORCE_TEMPLATE"] = "1"

from sisyphus.io import MatchWriter, load_match  # noqa: E402
from sisyphus.sim import MatchSimulator  # noqa: E402
from sisyphus.sim.scenario import PRESS_COLLAPSE, SubstitutionPlan  # noqa: E402

SHORT = dataclasses.replace(
    PRESS_COLLAPSE,
    scenario_id="press_collapse",
    period_seconds=180.0,
    substitutions=(SubstitutionPlan(minute=4.0, team_id="KMR", out_number=11, in_number=14),),
)


@pytest.fixture(scope="session")
def short_match_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("matches")
    sim = MatchSimulator(SHORT, seed=3)
    writer = MatchWriter(out, sim, batch_frames=500)
    for tick in sim.run():
        writer.write(tick)
    writer.close()
    return writer.dir


@pytest.fixture(scope="session")
def short_match(short_match_dir: Path):
    return load_match(short_match_dir)

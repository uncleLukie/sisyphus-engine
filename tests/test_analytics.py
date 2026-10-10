from __future__ import annotations

import numpy as np

from sisyphus.analytics import AnalyticsEngine, MatchSnapshot
from sisyphus.analytics.passes import describe_difficulty, pass_difficulty
from sisyphus.io import replay


def test_engine_runs_and_snapshots(short_match_dir, short_match):
    eng = AnalyticsEngine(short_match.meta, window_s=60, baseline_s=90)
    triggers = []
    for tick in replay(short_match_dir):
        triggers.extend(eng.ingest(tick))
    snap = eng.snapshot()
    assert isinstance(snap, MatchSnapshot)
    assert snap.period == 2
    assert 0 <= snap.home.energy.index <= 100
    assert 0 <= snap.away.energy.index <= 100
    assert abs(snap.home.possession_pct + snap.away.possession_pct - 100) < 1.0
    assert snap.home.passing.attempted > 10
    assert sum(p.on_pitch for p in snap.players) == 22
    assert len(snap.players) == 23  # one substitution happened
    assert all(p.top_speed_kmh < 42 for p in snap.players)
    kinds = {t.kind for t in triggers}
    assert "full_time" in kinds and "substitution" in kinds
    ids = [t.trigger_id for t in triggers]
    assert len(ids) == len(set(ids))
    for t in triggers:
        assert t.snapshot.match_id == short_match.meta.match_id
        assert 0 <= t.severity <= 1


def _feat(**kw) -> dict[str, float]:
    base = {
        "distance_m": 8.0,
        "progress_m": 0.0,
        "into_final_third": 0.0,
        "into_box": 0.0,
        "under_pressure": 0.0,
        "lofted": 0.0,
    }
    base.update(kw)
    return base


def test_pass_difficulty_monotonic():
    easy = pass_difficulty(_feat())
    harder = pass_difficulty(
        _feat(distance_m=35, progress_m=25, into_final_third=1, into_box=1, under_pressure=1, lofted=1)
    )
    assert 0 <= easy < harder <= 1
    assert describe_difficulty(easy) == "routine"
    assert describe_difficulty(harder) == "very hard"
    ds = [pass_difficulty(_feat(distance_m=x)) for x in np.linspace(5, 60, 12)]
    assert all(a <= b for a, b in zip(ds, ds[1:], strict=False))

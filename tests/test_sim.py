from __future__ import annotations

import numpy as np

from sisyphus.schema import EventType
from sisyphus.sim import MatchSimulator
from sisyphus.sim import pitch as P
from tests.conftest import SHORT


def test_match_dataset_shapes(short_match):
    m = short_match
    n_frames = len(m.frames)
    assert n_frames > 2 * 180 * P.FPS * 0.95
    assert len(m.tracking) == n_frames * 22
    assert set(m.tracking.columns) >= {"frame", "period", "t", "player_id", "team_id", "x", "y", "speed"}
    assert m.tracking["x"].between(-2.5, P.LENGTH + 2.5).all()
    assert m.tracking["y"].between(-2.5, P.WIDTH + 2.5).all()
    assert m.meta.final_score is not None
    assert m.spadl is not None and len(m.spadl) > 0


def test_events_are_plausible(short_match):
    evs = short_match.events
    kinds = {e.type for e in evs}
    assert EventType.PASS in kinds and EventType.PRESSURE in kinds
    passes = [e for e in evs if e.type == EventType.PASS]
    acc = sum(e.outcome == "success" for e in passes) / len(passes)
    assert 0.6 <= acc <= 0.95
    for e in passes:
        assert e.pass_distance_m is not None and e.ball_speed_mps is not None
        assert 0 < e.ball_speed_mps < 45
    subs = [e for e in evs if e.type == EventType.SUBSTITUTION]
    assert len(subs) == 1 and subs[0].team_id == "KMR"


def test_speeds_are_human(short_match):
    sp = short_match.tracking["speed"].to_numpy()
    assert sp.max() < 11.5  # ~41 km/h hard ceiling
    assert 0.5 < sp.mean() < 3.5


def test_outfield_players_stay_out_of_goal_areas(short_match):
    trk = short_match.tracking
    gk = {t.players[0].player_id for t in (short_match.meta.home, short_match.meta.away)}
    out = trk[~trk["player_id"].isin(gk)]
    in_box = ((out["x"] < 4.0) | (out["x"] > P.LENGTH - 4.0)) & ((out["y"] - P.WIDTH / 2).abs() < 8.0)
    assert in_box.mean() < 0.005


def test_same_seed_is_deterministic():
    a = MatchSimulator(SHORT, seed=5)
    b = MatchSimulator(SHORT, seed=5)
    xs_a = [t.frame.xy.copy() for _, t in zip(range(200), a.run(), strict=False)]
    xs_b = [t.frame.xy.copy() for _, t in zip(range(200), b.run(), strict=False)]
    assert len(xs_a) == 200
    assert all(np.array_equal(x, y) for x, y in zip(xs_a, xs_b, strict=True))

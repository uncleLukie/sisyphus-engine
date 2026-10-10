"""Load a generated match and replay it as a tick stream.

:func:`replay` reproduces exactly the :class:`~sisyphus.sim.Tick` sequence the
simulator emitted, so every downstream component (analytics, agents, API) is
agnostic to whether it is fed by a live simulation or a recorded dataset.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from sisyphus.schema import Event, MatchMeta
from sisyphus.sim.engine import Frame, FrameTruth, Tick


@dataclass
class MatchData:
    meta: MatchMeta
    events: list[Event]
    tracking: pd.DataFrame
    frames: pd.DataFrame
    truth: pd.DataFrame | None
    spadl: pd.DataFrame | None
    path: Path


def load_match(match_dir: Path, with_truth: bool = True) -> MatchData:
    match_dir = Path(match_dir)
    meta = MatchMeta.model_validate_json((match_dir / "meta.json").read_text(encoding="utf-8"))
    with (match_dir / "events.jsonl").open(encoding="utf-8") as fh:
        events = [Event.model_validate_json(line) for line in fh if line.strip()]
    tracking = pd.read_parquet(match_dir / "tracking.parquet")
    frames = pd.read_parquet(match_dir / "frames.parquet")
    truth = (
        pd.read_parquet(match_dir / "truth.parquet") if with_truth and (match_dir / "truth.parquet").exists() else None
    )
    spadl = pd.read_csv(match_dir / "spadl.csv") if (match_dir / "spadl.csv").exists() else None
    return MatchData(meta, events, tracking, frames, truth, spadl, match_dir)


def list_matches(root: Path) -> list[MatchMeta]:
    out = []
    for p in sorted(Path(root).glob("*/meta.json")):
        try:
            out.append(MatchMeta.model_validate_json(p.read_text(encoding="utf-8")))
        except (OSError, ValueError, json.JSONDecodeError):
            continue
    return out


def replay(match_dir: Path) -> Iterator[Tick]:
    """Yield the recorded match frame by frame as :class:`Tick` objects."""
    data = load_match(match_dir, with_truth=True)
    events_by_frame: dict[int, list[Event]] = {}
    for ev in data.events:
        events_by_frame.setdefault(ev.frame, []).append(ev)

    n_players = data.tracking.groupby("frame").size().iloc[0]
    trk = data.tracking.sort_values(["frame"], kind="stable")
    ids_all = trk["player_id"].to_numpy().reshape(-1, n_players)
    xy_all = trk[["x", "y"]].to_numpy(dtype=np.float32).reshape(-1, n_players, 2)
    sp_all = trk["speed"].to_numpy(dtype=np.float32).reshape(-1, n_players)
    if data.truth is not None:
        tr = data.truth.sort_values("frame", kind="stable")
        e_short = tr["e_short"].to_numpy(dtype=np.float32).reshape(-1, n_players)
        e_long = tr["e_long"].to_numpy(dtype=np.float32).reshape(-1, n_players)
        will = tr["willingness"].to_numpy(dtype=np.float32).reshape(-1, n_players)
    else:
        e_short = e_long = will = np.ones((len(ids_all), n_players), dtype=np.float32)

    fr = data.frames.sort_values("frame").reset_index(drop=True)
    for i, row in enumerate(fr.itertuples(index=False)):
        frame = Frame(
            frame=int(row.frame),
            period=int(row.period),
            t=float(row.t),
            player_ids=list(ids_all[i]),
            xy=xy_all[i],
            speed=sp_all[i],
            ball_xy=np.array([row.ball_x, row.ball_y], dtype=float),
            ball_speed=float(row.ball_speed),
            ball_state=str(row.ball_state),
            owner_id=None if pd.isna(row.owner_id) else str(row.owner_id),
            possession_team=None if pd.isna(row.possession_team) else str(row.possession_team),
            score=(int(row.score_home), int(row.score_away)),
            truth=FrameTruth(e_short=e_short[i], e_long=e_long[i], willingness=will[i]),
        )
        yield Tick(frame, events_by_frame.get(int(row.frame), []))

"""Persist a simulated match as an open, analysis-friendly dataset.

Layout of ``<out_dir>/<match_id>/``::

    meta.json         MatchMeta (teams, rosters, scenario, seed, final score)
    events.jsonl      one Event per line, in emission order
    tracking.parquet  long format: one row per (frame, player)
    frames.parquet    one row per frame: ball, possession, score
    truth.parquet     hidden simulator energy state per (frame, player) - evaluation only
    spadl.csv         SPADL-style action table derived from the events

Tracking is written in long format because it is the most interoperable shape
(easy to pivot, filter by player, or load into kloppy/pandas/Fabric).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from sisyphus.io.spadl import events_to_spadl
from sisyphus.schema import Event, MatchMeta, TeamInfo
from sisyphus.sim import MatchSimulator, Tick, get_scenario
from sisyphus.sim import pitch as _pitch
from sisyphus.sim.roster import TeamProfile

_TRACKING_SCHEMA = pa.schema(
    [
        ("frame", pa.int32()),
        ("period", pa.int8()),
        ("t", pa.float32()),
        ("player_id", pa.string()),
        ("team_id", pa.string()),
        ("x", pa.float32()),
        ("y", pa.float32()),
        ("speed", pa.float32()),
    ]
)
_TRUTH_SCHEMA = pa.schema(
    [
        ("frame", pa.int32()),
        ("player_id", pa.string()),
        ("e_short", pa.float32()),
        ("e_long", pa.float32()),
        ("willingness", pa.float32()),
    ]
)
_FRAMES_SCHEMA = pa.schema(
    [
        ("frame", pa.int32()),
        ("period", pa.int8()),
        ("t", pa.float32()),
        ("ball_x", pa.float32()),
        ("ball_y", pa.float32()),
        ("ball_speed", pa.float32()),
        ("ball_state", pa.string()),
        ("owner_id", pa.string()),
        ("possession_team", pa.string()),
        ("score_home", pa.int8()),
        ("score_away", pa.int8()),
    ]
)


def team_info(team: TeamProfile, style: str) -> TeamInfo:
    return TeamInfo(
        team_id=team.team_id,
        name=team.name,
        short_name=team.short_name,
        primary_color=team.primary_color,
        secondary_color=team.secondary_color,
        formation=team.formation,
        style=style,
        players=[p.info(True) for p in team.starters] + [p.info(False) for p in team.bench],
    )


class MatchWriter:
    """Streams :class:`Tick` objects to disk in row-group batches."""

    def __init__(self, out_dir: Path, sim: MatchSimulator, batch_frames: int = 3000):
        self.dir = Path(out_dir) / sim.match_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self.sim = sim
        self.batch_frames = batch_frames
        self._events: list[Event] = []
        self._buf: list[Tick] = []
        self._tracking = pq.ParquetWriter(self.dir / "tracking.parquet", _TRACKING_SCHEMA)
        self._truth = pq.ParquetWriter(self.dir / "truth.parquet", _TRUTH_SCHEMA)
        self._frames = pq.ParquetWriter(self.dir / "frames.parquet", _FRAMES_SCHEMA)
        self._events_fh = (self.dir / "events.jsonl").open("w", encoding="utf-8")

    def write(self, tick: Tick) -> None:
        self._buf.append(tick)
        for ev in tick.events:
            self._events.append(ev)
            self._events_fh.write(ev.model_dump_json(exclude_none=True) + "\n")
        if len(self._buf) >= self.batch_frames:
            self._flush()

    def _flush(self) -> None:
        if not self._buf:
            return
        n_players = len(self._buf[0].frame.player_ids)
        # Player ids can change mid-match (substitutions), so take them per frame.
        player_col = np.concatenate([np.array(t.frame.player_ids) for t in self._buf])
        team_col = np.array([pid.split("-")[0] for pid in player_col])
        xy = np.concatenate([t.frame.xy for t in self._buf])
        speed = np.concatenate([t.frame.speed for t in self._buf])
        frame_col = np.repeat([t.frame.frame for t in self._buf], n_players).astype(np.int32)
        period_col = np.repeat([t.frame.period for t in self._buf], n_players).astype(np.int8)
        t_col = np.repeat([t.frame.t for t in self._buf], n_players).astype(np.float32)
        self._tracking.write_table(
            pa.table(
                {
                    "frame": frame_col,
                    "period": period_col,
                    "t": t_col,
                    "player_id": player_col,
                    "team_id": team_col,
                    "x": xy[:, 0].astype(np.float32),
                    "y": xy[:, 1].astype(np.float32),
                    "speed": speed.astype(np.float32),
                },
                schema=_TRACKING_SCHEMA,
            )
        )
        self._truth.write_table(
            pa.table(
                {
                    "frame": frame_col,
                    "player_id": player_col,
                    "e_short": np.concatenate([t.frame.truth.e_short for t in self._buf]),
                    "e_long": np.concatenate([t.frame.truth.e_long for t in self._buf]),
                    "willingness": np.concatenate([t.frame.truth.willingness for t in self._buf]),
                },
                schema=_TRUTH_SCHEMA,
            )
        )
        self._frames.write_table(
            pa.table(
                {
                    "frame": np.array([t.frame.frame for t in self._buf], dtype=np.int32),
                    "period": np.array([t.frame.period for t in self._buf], dtype=np.int8),
                    "t": np.array([t.frame.t for t in self._buf], dtype=np.float32),
                    "ball_x": np.array([t.frame.ball_xy[0] for t in self._buf], dtype=np.float32),
                    "ball_y": np.array([t.frame.ball_xy[1] for t in self._buf], dtype=np.float32),
                    "ball_speed": np.array([t.frame.ball_speed for t in self._buf], dtype=np.float32),
                    "ball_state": [t.frame.ball_state for t in self._buf],
                    "owner_id": [t.frame.owner_id for t in self._buf],
                    "possession_team": [t.frame.possession_team for t in self._buf],
                    "score_home": np.array([t.frame.score[0] for t in self._buf], dtype=np.int8),
                    "score_away": np.array([t.frame.score[1] for t in self._buf], dtype=np.int8),
                },
                schema=_FRAMES_SCHEMA,
            )
        )
        self._buf.clear()

    def close(self) -> MatchMeta:
        self._flush()
        self._tracking.close()
        self._truth.close()
        self._frames.close()
        self._events_fh.close()
        sc, sim = self.sim.scenario, self.sim
        meta = MatchMeta(
            match_id=sim.match_id,
            scenario_id=sc.scenario_id,
            scenario_title=sc.title,
            seed=sim.seed,
            fps=_pitch.FPS,
            pitch_length=_pitch.LENGTH,
            pitch_width=_pitch.WIDTH,
            home=team_info(sc.home, sc.home_style.name),
            away=team_info(sc.away, sc.away_style.name),
            final_score=(sim.score[0], sim.score[1]),
            period_lengths_s=list(sim.period_lengths),
        )
        (self.dir / "meta.json").write_text(meta.model_dump_json(indent=2), encoding="utf-8")
        spadl = events_to_spadl(self._events, meta)
        spadl.to_csv(self.dir / "spadl.csv", index=False)
        return meta


def generate_match(
    scenario_id: str,
    out_dir: Path,
    seed: int | None = None,
    progress: Callable[[float], None] | None = None,
) -> tuple[MatchMeta, Path]:
    """Simulate a scenario and write the full dataset. Returns (meta, match_dir).

    ``progress`` receives the completed fraction (0..1), approximately.
    """
    scenario = get_scenario(scenario_id)
    sim = MatchSimulator(scenario, seed=seed)
    writer = MatchWriter(out_dir, sim)
    expected = 2 * scenario.period_seconds * _pitch.FPS
    for i, tick in enumerate(sim.run()):
        writer.write(tick)
        if progress and i % 3000 == 0:
            progress(min(0.99, i / expected))
    meta = writer.close()
    return meta, writer.dir


def events_dataframe(events: Iterable[Event]) -> pd.DataFrame:
    """Flatten events (qualifiers expanded as ``q_*`` columns) into a DataFrame."""
    rows = []
    for ev in events:
        d = ev.model_dump(exclude={"qualifiers"})
        d["type"] = ev.type.value
        d.update({f"q_{k}": v for k, v in ev.qualifiers.items()})
        rows.append(d)
    return pd.DataFrame(rows)

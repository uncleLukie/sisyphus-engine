"""Incremental match analytics: tracking frames + events in, snapshots + triggers out.

Design notes
------------
* Everything is computed from *observable* data (positions, speeds, events).
  The simulator's hidden energy state is never read here; it exists only so the
  pipeline can be evaluated against ground truth (see ``sisyphus eval``).
* Work is bucketed per second. Rolling windows keep running sums (add on append,
  subtract on evict), so a frame costs O(22) and a snapshot costs O(1).
* The Team Energy Index compares a team's current 5-minute output to its own
  opening-15-minute baseline. Comparing a team to itself removes style effects
  (a low block legitimately sprints less than a high press).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
from scipy.spatial import ConvexHull, QhullError

from sisyphus.analytics.contracts import (
    MatchSnapshot,
    PlayerPhysical,
    TeamEnergy,
    TeamPassing,
    TeamPhysical,
    TeamShooting,
    TeamSnapshot,
    Trigger,
)
from sisyphus.analytics.detectors import TriggerDetector
from sisyphus.analytics.passes import pass_difficulty, pass_features
from sisyphus.schema import Event, EventType, MatchMeta
from sisyphus.sim import pitch
from sisyphus.sim.engine import Tick

HI_SPEED = 5.5  # m/s, high-intensity running threshold
SPRINT_SPEED = 7.0  # m/s
SPRINT_MIN_FRAMES = 5  # 0.5 s at 10 fps
ACCEL_THRESHOLD = 3.0  # m/s^2
HULL_EVERY = 5  # frames
N_SLOTS = 22

# Team-level bucket fields (each a length-2 array: home, away).
TEAM_FIELDS = (
    "distance",
    "hi_distance",
    "sprints",
    "hi_ip",  # high-intensity distance while in possession
    "hi_oop",  # ... while out of possession
    "sprints_ip",
    "sprints_oop",
    "ip_frames",  # frames in possession (ball alive)
    "oop_frames",
    "accels",
    "speed_sum",
    "hull_sum",
    "hull_n",
    "length_sum",
    "line_sum",
    "line_n",
    "poss_frames",
    "territory_sum",
    "pressures",
    "closing_sum",
    "high_recoveries",
    "threat",
)


def clock_str(period: int, t: float, period_s: float = 2700.0) -> str:
    base = (period - 1) * period_s
    if t > period_s:
        extra = t - period_s
        return f"{int((base + period_s) // 60)}+{int(extra // 60)}:{int(extra % 60):02d}"
    total = base + t
    return f"{int(total // 60):02d}:{int(total % 60):02d}"


def threat_value(tf_x: float, tf_y: float) -> float:
    """Simple xT-like value of ball location in the possession team's frame (0..~0.4)."""
    centrality = 1.0 - 0.7 * (2.0 * abs(tf_y - 0.5)) ** 2
    return 0.01 + 0.03 * tf_x + 0.38 * max(0.0, tf_x - 0.6) ** 2 / 0.16 * centrality


class _Bucket:
    """One second of accumulated quantities."""

    __slots__ = ("team", "frames", "p_distance", "p_sprints")

    def __init__(self) -> None:
        self.team = np.zeros((len(TEAM_FIELDS), 2))
        self.frames = 0
        self.p_distance = np.zeros(N_SLOTS)
        self.p_sprints = np.zeros(N_SLOTS)


_IDX = {name: i for i, name in enumerate(TEAM_FIELDS)}


class _Window:
    """Fixed-length window of buckets with O(1) running sums."""

    def __init__(self, maxlen: int | None):
        self.buckets: deque[_Bucket] = deque(maxlen=maxlen)
        self.team = np.zeros((len(TEAM_FIELDS), 2))
        self.frames = 0
        self.p_distance = np.zeros(N_SLOTS)
        self.p_sprints = np.zeros(N_SLOTS)

    def __len__(self) -> int:
        return len(self.buckets)

    def append(self, b: _Bucket) -> None:
        if self.buckets.maxlen is not None and len(self.buckets) == self.buckets.maxlen:
            old = self.buckets[0]
            self.team -= old.team
            self.frames -= old.frames
            self.p_distance -= old.p_distance
            self.p_sprints -= old.p_sprints
        self.buckets.append(b)
        self.team += b.team
        self.frames += b.frames
        self.p_distance += b.p_distance
        self.p_sprints += b.p_sprints

    def get(self, name: str) -> np.ndarray:
        return self.team[_IDX[name]]

    def recent_threat(self, seconds: int) -> np.ndarray:
        n = min(seconds, len(self.buckets))
        if n == 0:
            return np.zeros(2)
        return np.sum([b.team[_IDX["threat"]] for b in list(self.buckets)[-n:]], axis=0)


@dataclass
class _PlayerTotals:
    distance: float = 0.0
    hi_distance: float = 0.0
    sprints: int = 0
    top_speed: float = 0.0
    frames: int = 0
    baseline_sprint_rate: float | None = None  # per window_s during opening window
    baseline_distance_rate: float | None = None


@dataclass
class _PassTotals:
    attempted: int = 0
    completed: int = 0
    distance_sum: float = 0.0
    difficulty_sum: float = 0.0
    hard_completed: int = 0
    progressive: int = 0
    fastest: float = 0.0


@dataclass
class _ShotTotals:
    shots: int = 0
    on_target: int = 0
    goals: int = 0
    xg: float = 0.0
    fastest: float = 0.0


class AnalyticsEngine:
    """Feed :class:`Tick` objects in order; read snapshots and triggers out."""

    def __init__(self, meta: MatchMeta, window_s: int = 300, baseline_s: int = 900):
        self.meta = meta
        self.window_s = window_s
        self.baseline_s = baseline_s
        self.team_ids = (meta.home.team_id, meta.away.team_id)
        self.players = {p.player_id: p for p in (*meta.home.players, *meta.away.players)}
        self.fps = meta.fps
        self.dt = 1.0 / meta.fps
        self.team_slot = np.repeat([0, 1], 11)

        self.window = _Window(window_s)
        self.baseline = _Window(None)
        self.baseline_done = False
        self.current = _Bucket()
        self.current_second = -1
        self.seconds_played = 0

        self.prev_speed = np.zeros(N_SLOTS)
        self.sprint_frames = np.zeros(N_SLOTS, dtype=int)
        # Per-slot accumulators are flushed into _PlayerTotals on substitution.
        self.slot_pid: list[str] = []
        self.slot_distance = np.zeros(N_SLOTS)
        self.slot_hi = np.zeros(N_SLOTS)
        self.slot_sprints = np.zeros(N_SLOTS, dtype=int)
        self.slot_top = np.zeros(N_SLOTS)
        self.slot_frames = np.zeros(N_SLOTS, dtype=int)
        self.totals: dict[str, _PlayerTotals] = {}

        self.passes = {t: _PassTotals() for t in self.team_ids}
        self.shots = {t: _ShotTotals() for t in self.team_ids}
        self.poss_total = np.zeros(2)
        self.momentum = 0.0
        self.energy_history: dict[str, deque[float]] = {t: deque(maxlen=180) for t in self.team_ids}
        self.energy_ema = {t: 100.0 for t in self.team_ids}
        self.last_frame = None
        self.score = (0, 0)
        self.period_lengths: list[float] = []
        self._last_detect_second = -1
        self.detector = TriggerDetector(self)

    # ------------------------------------------------------------------ ingestion
    def ingest(self, tick: Tick) -> list[Trigger]:
        f = tick.frame
        if self.last_frame is not None and f.period != self.last_frame.period:
            self._on_new_period()
        self.last_frame = f
        self.score = f.score
        if f.player_ids != self.slot_pid:
            self._sync_slots(f.player_ids)

        second = int(f.t)
        if second != self.current_second:
            self._roll_bucket()
            self.current_second = second

        b = self.current
        T = b.team
        speed = f.speed.astype(float)
        dist = speed * self.dt
        accel = np.abs(speed - self.prev_speed) / self.dt
        self.prev_speed = speed

        sprinting = speed >= SPRINT_SPEED
        self.sprint_frames = np.where(sprinting, self.sprint_frames + 1, 0)
        new_sprint = self.sprint_frames == SPRINT_MIN_FRAMES
        hi = speed >= HI_SPEED

        team_dist = np.bincount(self.team_slot, weights=dist, minlength=2)
        team_hi = np.bincount(self.team_slot, weights=dist * hi, minlength=2)
        team_spr = np.bincount(self.team_slot, weights=new_sprint, minlength=2)
        T[_IDX["distance"]] += team_dist
        T[_IDX["hi_distance"]] += team_hi
        T[_IDX["sprints"]] += team_spr
        T[_IDX["accels"]] += np.bincount(self.team_slot, weights=accel > ACCEL_THRESHOLD, minlength=2)
        T[_IDX["speed_sum"]] += np.bincount(self.team_slot, weights=speed, minlength=2)
        b.frames += 1
        b.p_distance += dist
        b.p_sprints += new_sprint

        self.slot_distance += dist
        self.slot_hi += dist * hi
        self.slot_sprints += new_sprint
        np.maximum(self.slot_top, speed, out=self.slot_top)
        self.slot_frames += 1

        if f.possession_team in self.team_ids and f.ball_state != "dead":
            k = self.team_ids.index(f.possession_team)
            tf = pitch.to_team_frame(f.ball_xy, k)
            T[_IDX["poss_frames"], k] += 1
            T[_IDX["territory_sum"], k] += float(tf[0])
            T[_IDX["threat"], k] += threat_value(float(tf[0]), float(tf[1]))
            self.poss_total[k] += 1
            # Phase split: the possession team's output is "in possession", the other's is "out".
            T[_IDX["ip_frames"], k] += 1
            T[_IDX["oop_frames"], 1 - k] += 1
            T[_IDX["hi_ip"], k] += team_hi[k]
            T[_IDX["hi_oop"], 1 - k] += team_hi[1 - k]
            T[_IDX["sprints_ip"], k] += team_spr[k]
            T[_IDX["sprints_oop"], 1 - k] += team_spr[1 - k]

        if f.frame % HULL_EVERY == 0:
            for k in (0, 1):
                pts = f.xy[11 * k + 1 : 11 * k + 11].astype(float)
                try:
                    T[_IDX["hull_sum"], k] += ConvexHull(pts).volume
                    T[_IDX["hull_n"], k] += 1
                except QhullError:
                    pass
                tfx = pitch.to_team_frame(pts, k)[:, 0]
                T[_IDX["length_sum"], k] += float(tfx.max() - tfx.min()) * pitch.LENGTH
                if f.possession_team is not None and f.possession_team != self.team_ids[k]:
                    T[_IDX["line_sum"], k] += float(np.sort(tfx)[1]) * pitch.LENGTH
                    T[_IDX["line_n"], k] += 1

        for ev in tick.events:
            self._ingest_event(ev)

        triggers: list[Trigger] = []
        if second != self._last_detect_second:
            self._last_detect_second = second
            self._update_energy()
            triggers.extend(self.detector.on_second(tick))
        if tick.events:
            triggers.extend(self.detector.on_events(tick))
        return triggers

    def _sync_slots(self, ids: list[str]) -> None:
        for i, old in enumerate(self.slot_pid):
            if i < len(ids) and ids[i] != old:
                self._flush_slot(i)
                self.slot_distance[i] = self.slot_hi[i] = self.slot_top[i] = 0.0
                self.slot_sprints[i] = self.slot_frames[i] = 0
                self.sprint_frames[i] = 0
        self.slot_pid = list(ids)
        for pid in ids:
            self.totals.setdefault(pid, _PlayerTotals())

    def _flush_slot(self, i: int) -> None:
        pid = self.slot_pid[i]
        pt = self.totals.setdefault(pid, _PlayerTotals())
        pt.distance += float(self.slot_distance[i])
        pt.hi_distance += float(self.slot_hi[i])
        pt.sprints += int(self.slot_sprints[i])
        pt.top_speed = max(pt.top_speed, float(self.slot_top[i]))
        pt.frames += int(self.slot_frames[i])

    def player_totals(self, pid: str) -> _PlayerTotals:
        """Cumulative totals for a player, including the live slot accumulator."""
        base = self.totals.get(pid, _PlayerTotals())
        if pid not in self.slot_pid:
            return base
        i = self.slot_pid.index(pid)
        return _PlayerTotals(
            distance=base.distance + float(self.slot_distance[i]),
            hi_distance=base.hi_distance + float(self.slot_hi[i]),
            sprints=base.sprints + int(self.slot_sprints[i]),
            top_speed=max(base.top_speed, float(self.slot_top[i])),
            frames=base.frames + int(self.slot_frames[i]),
            baseline_sprint_rate=base.baseline_sprint_rate,
            baseline_distance_rate=base.baseline_distance_rate,
        )

    def _roll_bucket(self) -> None:
        if self.current.frames == 0:
            return
        self.window.append(self.current)
        self.seconds_played += 1
        if not self.baseline_done:
            self.baseline.append(self.current)
            if len(self.baseline) >= self.baseline_s:
                self.baseline_done = True
                self._freeze_player_baselines()
        self.current = _Bucket()

    def _on_new_period(self) -> None:
        assert self.last_frame is not None
        self.period_lengths.append(self.last_frame.t)
        self._roll_bucket()
        self.current_second = -1
        self.prev_speed[:] = 0.0
        self.sprint_frames[:] = 0

    def _freeze_player_baselines(self) -> None:
        scale = self.window_s / max(len(self.baseline), 1)
        for i, pid in enumerate(self.slot_pid):
            pt = self.totals.setdefault(pid, _PlayerTotals())
            pt.baseline_distance_rate = float(self.baseline.p_distance[i] * scale)
            pt.baseline_sprint_rate = float(self.baseline.p_sprints[i] * scale)

    def _ingest_event(self, ev: Event) -> None:
        if ev.team_id not in self.team_ids:
            return
        k = self.team_ids.index(ev.team_id)
        T = self.current.team
        if ev.type == EventType.PRESSURE:
            T[_IDX["pressures"], k] += 1
            T[_IDX["closing_sum"], k] += float(ev.qualifiers.get("closing_speed", 1.0))
        elif ev.type == EventType.BALL_RECOVERY and ev.qualifiers.get("regained"):
            tfx = pitch.to_team_frame(np.array([ev.start_x, ev.start_y]), k)[0]
            if tfx >= 0.6:
                T[_IDX["high_recoveries"], k] += 1
        elif ev.type == EventType.PASS:
            p = self.passes[ev.team_id]
            feats = pass_features(ev, k == 0)
            diff = pass_difficulty(feats)
            p.attempted += 1
            p.distance_sum += feats["distance_m"]
            p.difficulty_sum += diff
            if ev.outcome == "success":
                p.completed += 1
                if diff >= 0.5:
                    p.hard_completed += 1
                if feats["progress_m"] >= 10:
                    p.progressive += 1
            p.fastest = max(p.fastest, ev.ball_speed_mps or 0.0)
        elif ev.type == EventType.SHOT:
            s = self.shots[ev.team_id]
            s.shots += 1
            s.xg += ev.xg or 0.0
            if ev.qualifiers.get("shot_outcome") in ("goal", "saved"):
                s.on_target += 1
            s.fastest = max(s.fastest, ev.ball_speed_mps or 0.0)
        elif ev.type == EventType.GOAL:
            self.shots[ev.team_id].goals += 1

    # ------------------------------------------------------------------ derived metrics
    def rates(self, w: _Window, k: int) -> dict[str, float]:
        """Window quantities normalised to a ``window_s`` period and to game phase.

        ``*_ip`` / ``*_oop`` rates are per minute in / out of possession, which
        removes the confound that a team pinned in its own half sprints less for
        tactical rather than physical reasons.
        """
        secs = max(len(w), 1)
        scale = self.window_s / secs
        poss_min = w.get("poss_frames")[1 - k] / self.fps / 60.0
        ip_min = w.get("ip_frames")[k] / self.fps / 60.0
        oop_min = w.get("oop_frames")[k] / self.fps / 60.0
        pressures = w.get("pressures")[k]
        return {
            "sprints": float(w.get("sprints")[k] * scale),
            "hi_distance": float(w.get("hi_distance")[k] * scale),
            "sprints_ip": float(w.get("sprints_ip")[k] / max(ip_min, 0.25)),
            "sprints_oop": float(w.get("sprints_oop")[k] / max(oop_min, 0.25)),
            "hi_ip": float(w.get("hi_ip")[k] / max(ip_min, 0.25)),
            "hi_oop": float(w.get("hi_oop")[k] / max(oop_min, 0.25)),
            "ip_share": float(ip_min / max(ip_min + oop_min, 1e-6)),
            "speed": float(w.get("speed_sum")[k] / max(w.frames, 1) / 11),
            "press_rate": float(pressures / max(poss_min, 0.25)),
            "press_intensity": float(w.get("closing_sum")[k] / max(pressures, 1)),
        }

    @staticmethod
    def energy_ratios(now: dict[str, float], base: dict[str, float]) -> dict[str, float]:
        def r(a: float, b: float) -> float:
            return float(np.clip(a / b, 0.0, 1.5)) if b > 1e-6 else 1.0

        # Phase-weighted ratios: compare like with like, weighted by the current phase mix.
        w_ip = now["ip_share"]
        sprint = w_ip * r(now["sprints_ip"], base["sprints_ip"]) + (1 - w_ip) * r(
            now["sprints_oop"], base["sprints_oop"]
        )
        hi = w_ip * r(now["hi_ip"], base["hi_ip"]) + (1 - w_ip) * r(now["hi_oop"], base["hi_oop"])
        return {
            "sprint": sprint,
            "hi": hi,
            "speed": r(now["speed"], base["speed"]),
            "press": r(now["press_rate"] * now["press_intensity"], base["press_rate"] * base["press_intensity"]),
        }

    def _update_energy(self) -> None:
        if not self.baseline_done or len(self.window) < 60:
            return
        for k, tid in enumerate(self.team_ids):
            ratios = self.energy_ratios(self.rates(self.window, k), self.rates(self.baseline, k))
            raw = 100.0 * min(
                1.1,
                0.35 * ratios["sprint"] + 0.30 * ratios["hi"] + 0.15 * ratios["speed"] + 0.20 * ratios["press"],
            )
            # Asymmetric smoothing: output can collapse quickly, but "energy" is only
            # credited back slowly, so a two-minute burst after a goal does not read as
            # a full recovery.
            alpha = 0.05 if raw < self.energy_ema[tid] else 0.012
            self.energy_ema[tid] = (1 - alpha) * self.energy_ema[tid] + alpha * raw
            self.energy_history[tid].append(self.energy_ema[tid])

    def team_energy(self, team_id: str) -> TeamEnergy:
        k = self.team_ids.index(team_id)
        if self.baseline_done and len(self.window):
            ratios = self.energy_ratios(self.rates(self.window, k), self.rates(self.baseline, k))
        else:
            ratios = {"sprint": 1.0, "hi": 1.0, "speed": 1.0, "press": 1.0}
        hist = self.energy_history[team_id]
        trend = "stable"
        if len(hist) >= 120:
            delta = hist[-1] - hist[-120]
            trend = "falling" if delta < -4 else "rising" if delta > 4 else "stable"
        return TeamEnergy(
            team_id=team_id,
            index=round(float(np.clip(self.energy_ema[team_id], 0, 100)), 1),
            sprint_ratio=round(ratios["sprint"], 2),
            hi_distance_ratio=round(ratios["hi"], 2),
            speed_ratio=round(ratios["speed"], 2),
            press_ratio=round(ratios["press"], 2),
            trend=trend,
            baseline_ready=self.baseline_done,
        )

    def team_physical(self, team_id: str) -> TeamPhysical:
        k = self.team_ids.index(team_id)
        w = self.window
        g = w.get
        hull_n = max(g("hull_n")[k], 1)
        line_n = max(g("line_n")[k], 1)
        poss_min = g("poss_frames")[1 - k] / self.fps / 60.0
        return TeamPhysical(
            team_id=team_id,
            window_s=float(len(w)),
            distance_m=round(float(g("distance")[k]), 1),
            hi_distance_m=round(float(g("hi_distance")[k]), 1),
            sprint_count=int(g("sprints")[k]),
            mean_speed_mps=round(float(g("speed_sum")[k] / max(w.frames, 1) / 11), 2),
            accel_count=int(g("accels")[k]),
            compactness_m2=round(float(g("hull_sum")[k] / hull_n), 1),
            block_length_m=round(float(g("length_sum")[k] / hull_n), 1),
            defensive_line_m=round(float(g("line_sum")[k] / line_n), 1),
            press_rate=round(float(g("pressures")[k] / max(poss_min, 0.25)), 2),
            high_recoveries=int(g("high_recoveries")[k]),
        )

    def team_passing(self, team_id: str) -> TeamPassing:
        p = self.passes[team_id]
        n = max(p.attempted, 1)
        return TeamPassing(
            team_id=team_id,
            attempted=p.attempted,
            completed=p.completed,
            accuracy=round(p.completed / n, 3),
            mean_distance_m=round(p.distance_sum / n, 1),
            mean_difficulty=round(p.difficulty_sum / n, 3),
            hard_completed=p.hard_completed,
            progressive=p.progressive,
            fastest_pass_kmh=round(p.fastest * 3.6, 1),
        )

    def team_shooting(self, team_id: str) -> TeamShooting:
        s = self.shots[team_id]
        return TeamShooting(
            team_id=team_id,
            shots=s.shots,
            on_target=s.on_target,
            goals=s.goals,
            xg=round(s.xg, 2),
            fastest_shot_kmh=round(s.fastest * 3.6, 1),
        )

    def player_physical(self, pid: str) -> PlayerPhysical:
        info = self.players[pid]
        pt = self.player_totals(pid)
        on = pid in self.slot_pid
        slot = self.slot_pid.index(pid) if on else -1
        cur_speed = float(self.last_frame.speed[slot]) if on and self.last_frame is not None else 0.0
        w_spr = int(self.window.p_sprints[slot]) if on else 0
        w_dist = float(self.window.p_distance[slot]) if on else 0.0
        energy = 100.0
        if on and pt.baseline_sprint_rate is not None and len(self.window) >= 60:
            scale = self.window_s / len(self.window)
            r_d = (w_dist * scale) / pt.baseline_distance_rate if pt.baseline_distance_rate else 1.0
            r_s = (w_spr * scale) / pt.baseline_sprint_rate if pt.baseline_sprint_rate else r_d
            energy = float(np.clip(100 * (0.5 * min(r_d, 1.2) + 0.5 * min(r_s, 1.2)), 0, 100))
        return PlayerPhysical(
            player_id=pid,
            name=info.name,
            team_id=info.team_id,
            number=info.number,
            position=info.position,
            on_pitch=on,
            distance_m=round(pt.distance, 0),
            hi_distance_m=round(pt.hi_distance, 0),
            sprints=pt.sprints,
            top_speed_kmh=round(pt.top_speed * 3.6, 1),
            current_speed_kmh=round(cur_speed * 3.6, 1),
            window_sprints=w_spr,
            window_distance_m=round(w_dist, 0),
            energy=round(energy, 1),
        )

    def _team_snapshot(self, k: int) -> TeamSnapshot:
        tid = self.team_ids[k]
        info = self.meta.home if k == 0 else self.meta.away
        tot = max(self.poss_total.sum(), 1)
        poss_frames = max(self.window.get("poss_frames")[k], 1)
        return TeamSnapshot(
            team_id=tid,
            name=info.name,
            short_name=info.short_name,
            color=info.primary_color,
            score=self.score[k],
            possession_pct=round(100 * float(self.poss_total[k] / tot), 1),
            physical=self.team_physical(tid),
            energy=self.team_energy(tid),
            passing=self.team_passing(tid),
            shooting=self.team_shooting(tid),
            territory=round(float(self.window.get("territory_sum")[k] / poss_frames), 3),
        )

    def current_momentum(self) -> float:
        th = self.window.recent_threat(180)
        tot = th.sum()
        return float((th[0] - th[1]) / tot) if tot > 1e-9 else 0.0

    def snapshot(self) -> MatchSnapshot:
        f = self.last_frame
        assert f is not None, "no frames ingested yet"
        self.momentum = self.current_momentum()
        return MatchSnapshot(
            match_id=self.meta.match_id,
            period=f.period,
            t=f.t,
            clock=clock_str(f.period, f.t),
            minute=round((f.period - 1) * 45 + f.t / 60, 2),
            home=self._team_snapshot(0),
            away=self._team_snapshot(1),
            momentum=round(float(np.clip(self.momentum, -1, 1)), 3),
            ball_xy=(round(float(f.ball_xy[0]), 2), round(float(f.ball_xy[1]), 2)),
            possession_team=f.possession_team,
            players=[self.player_physical(pid) for pid in self.players if pid in self.totals],
        )

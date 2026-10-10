"""Detect moments worth explaining and package the evidence behind them.

Each detector is deliberately simple and threshold-based so that every trigger
is auditable: the ``evidence`` dict contains the exact numbers that fired it,
and those numbers are what the Explain agents are allowed to talk about.
"""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING

import numpy as np

from sisyphus.analytics.contracts import Trigger, TriggerKind
from sisyphus.schema import Event, EventType
from sisyphus.sim import pitch
from sisyphus.sim.engine import Tick

if TYPE_CHECKING:
    from sisyphus.analytics.engine import AnalyticsEngine

ENERGY_THRESHOLDS = (80.0, 65.0, 50.0)
BIG_CHANCE_XG = 0.25
SPEED_BURST_MPS = 8.75  # 31.5 km/h
SUMMARY_MINUTES = (15.0, 30.0, 60.0, 75.0)


class TriggerDetector:
    def __init__(self, engine: AnalyticsEngine):
        self.e = engine
        self.seq = 0
        self.recent: deque[Event] = deque(maxlen=80)
        self.energy_fired: dict[str, set[float]] = {t: set() for t in engine.team_ids}
        self.last_press_drop: dict[str, float] = {t: -1e9 for t in engine.team_ids}
        self.last_swing_minute = -1e9
        self.last_swing_sign = 0
        self.speed_burst_at: dict[str, float] = {}
        self.milestones_done: set[str] = set()
        self.summaries_done: set[float] = set()
        self.possession_id = -1
        self.possession_start_minute = 0.0
        self.possession_start_tfx = 0.0
        self.possession_passes = 0
        self.possession_from_high_recovery = False

    # ------------------------------------------------------------------ helpers
    def _make(self, kind: TriggerKind, tick: Tick, headline: str, severity: float, **kw) -> Trigger:
        f = tick.frame
        snap = self.e.snapshot()
        self.seq += 1
        return Trigger(
            trigger_id=f"{self.e.meta.match_id}-{self.seq:04d}",
            match_id=self.e.meta.match_id,
            kind=kind,
            period=f.period,
            t=f.t,
            clock=snap.clock,
            minute=snap.minute,
            severity=float(np.clip(severity, 0, 1)),
            headline=headline,
            snapshot=snap,
            **kw,
        )

    def _name(self, pid: str | None) -> str:
        return self.e.players[pid].name if pid and pid in self.e.players else "unknown"

    def _team_name(self, tid: str | None) -> str:
        if tid == self.e.meta.home.team_id:
            return self.e.meta.home.short_name
        return self.e.meta.away.short_name

    def _minute(self, tick: Tick) -> float:
        return (tick.frame.period - 1) * 45 + tick.frame.t / 60

    # ------------------------------------------------------------------ events
    def on_events(self, tick: Tick) -> list[Trigger]:
        out: list[Trigger] = []
        for ev in tick.events:
            self._track_possession(ev, tick)
            self.recent.append(ev)
            if ev.type == EventType.GOAL:
                out.append(self._goal(ev, tick))
            elif ev.type == EventType.SHOT and (ev.xg or 0) >= BIG_CHANCE_XG and ev.outcome != "success":
                out.append(self._big_chance(ev, tick))
            elif ev.type == EventType.SUBSTITUTION:
                out.append(self._substitution(ev, tick))
            elif ev.type == EventType.CARD:
                out.append(
                    self._make(
                        "card",
                        tick,
                        f"Yellow card for {self._name(ev.player_id)}",
                        0.4,
                        team_id=ev.team_id,
                        player_id=ev.player_id,
                        evidence={"card": ev.qualifiers.get("card", "yellow"), "player": self._name(ev.player_id)},
                    )
                )
            elif ev.type == EventType.PERIOD_END:
                out.append(self._period_end(ev, tick))
        return out

    def _track_possession(self, ev: Event, tick: Tick) -> None:
        if ev.possession_id != self.possession_id:
            self.possession_id = ev.possession_id or -1
            self.possession_start_minute = self._minute(tick)
            self.possession_passes = 0
            self.possession_from_high_recovery = False
            if ev.start_x is not None and ev.team_id in self.e.team_ids:
                k = self.e.team_ids.index(ev.team_id)
                self.possession_start_tfx = float(pitch.to_team_frame(np.array([ev.start_x, ev.start_y]), k)[0])
        if ev.type == EventType.PASS and ev.outcome == "success":
            self.possession_passes += 1
        if ev.type == EventType.BALL_RECOVERY and ev.qualifiers.get("regained") and ev.team_id in self.e.team_ids:
            k = self.e.team_ids.index(ev.team_id)
            if pitch.to_team_frame(np.array([ev.start_x, ev.start_y]), k)[0] >= 0.6:
                self.possession_from_high_recovery = True

    def _last_shot(self, team_id: str | None) -> Event | None:
        for ev in reversed(self.recent):
            if ev.type == EventType.SHOT and ev.team_id == team_id:
                return ev
        return None

    def _assist(self, scorer: str | None, shot: Event | None) -> str | None:
        if shot is None:
            return None
        for ev in reversed(self.recent):
            if ev.type == EventType.PASS and ev.outcome == "success" and ev.receiver_id == scorer and ev.t <= shot.t:
                return ev.player_id if shot.t - ev.t < 12 else None
        return None

    def _goal(self, ev: Event, tick: Tick) -> Trigger:
        shot = self._last_shot(ev.team_id)
        assist = self._assist(ev.player_id, shot)
        snap_team = self.e.team_energy(ev.team_id) if ev.team_id else None
        opp_id = self.e.team_ids[1 - self.e.team_ids.index(ev.team_id)] if ev.team_id else None
        opp_energy = self.e.team_energy(opp_id) if opp_id else None
        build_s = max(0.0, (self._minute(tick) - self.possession_start_minute) * 60)
        evidence = {
            "scorer": self._name(ev.player_id),
            "team": self._team_name(ev.team_id),
            "score": ev.qualifiers.get("score", ""),
            "xg": round(shot.xg or 0, 2) if shot else None,
            "shot_distance_m": shot.qualifiers.get("distance_m") if shot else None,
            "shot_speed_kmh": round((shot.ball_speed_mps or 0) * 3.6, 1) if shot else None,
            "under_pressure": bool(shot.under_pressure) if shot else None,
            "assist": self._name(assist) if assist else None,
            "passes_in_buildup": self.possession_passes,
            "buildup_seconds": round(build_s, 1),
            "started_from_high_recovery": self.possession_from_high_recovery,
            "possession_start_progress": round(self.possession_start_tfx, 2),
            "scoring_team_energy": snap_team.index if snap_team else None,
            "conceding_team_energy": opp_energy.index if opp_energy else None,
            "conceding_team_trend": opp_energy.trend if opp_energy else None,
        }
        return self._make(
            "goal",
            tick,
            f"GOAL! {self._name(ev.player_id)} scores for {self._team_name(ev.team_id)} ({evidence['score']})",
            1.0,
            team_id=ev.team_id,
            player_id=ev.player_id,
            evidence=evidence,
        )

    def _big_chance(self, ev: Event, tick: Tick) -> Trigger:
        outcome = ev.qualifiers.get("shot_outcome", "missed")
        return self._make(
            "big_chance",
            tick,
            f"Big chance for {self._team_name(ev.team_id)}: {self._name(ev.player_id)} {outcome.replace('_', ' ')}",
            0.6 + 0.4 * min((ev.xg or 0) / 0.6, 1.0),
            team_id=ev.team_id,
            player_id=ev.player_id,
            evidence={
                "shooter": self._name(ev.player_id),
                "team": self._team_name(ev.team_id),
                "xg": round(ev.xg or 0, 2),
                "outcome": outcome,
                "shot_distance_m": ev.qualifiers.get("distance_m"),
                "shot_speed_kmh": round((ev.ball_speed_mps or 0) * 3.6, 1),
                "under_pressure": bool(ev.under_pressure),
                "passes_in_buildup": self.possession_passes,
            },
        )

    def _substitution(self, ev: Event, tick: Tick) -> Trigger:
        out_id, in_id = ev.qualifiers.get("player_out"), ev.qualifiers.get("player_in")
        out_phys = self.e.player_physical(str(out_id)) if out_id in self.e.players else None
        return self._make(
            "substitution",
            tick,
            f"{self._team_name(ev.team_id)} substitution: {self._name(str(in_id))} on for {self._name(str(out_id))}",
            0.5,
            team_id=ev.team_id,
            player_id=str(in_id),
            evidence={
                "player_in": self._name(str(in_id)),
                "player_out": self._name(str(out_id)),
                "team": self._team_name(ev.team_id),
                "out_distance_m": out_phys.distance_m if out_phys else None,
                "out_sprints": out_phys.sprints if out_phys else None,
                "out_energy": out_phys.energy if out_phys else None,
                "team_energy": self.e.team_energy(ev.team_id).index if ev.team_id else None,
            },
        )

    def _period_end(self, ev: Event, tick: Tick) -> Trigger:
        kind: TriggerKind = "full_time" if ev.period == 2 else "period_summary"
        label = "Full time" if kind == "full_time" else "Half time"
        snap = self.e.snapshot()
        return self._make(
            kind,
            tick,
            f"{label}: {snap.home.short_name} {snap.home.score}-{snap.away.score} {snap.away.short_name}",
            0.9 if kind == "full_time" else 0.7,
            evidence={"summary_type": kind, "score": ev.qualifiers.get("score", "")} | self._summary_evidence(snap),
        )

    def _summary_evidence(self, snap) -> dict:
        return {
            "home": snap.home.short_name,
            "away": snap.away.short_name,
            "home_possession": snap.home.possession_pct,
            "away_possession": snap.away.possession_pct,
            "home_xg": snap.home.shooting.xg,
            "away_xg": snap.away.shooting.xg,
            "home_shots": snap.home.shooting.shots,
            "away_shots": snap.away.shooting.shots,
            "home_energy": snap.home.energy.index,
            "away_energy": snap.away.energy.index,
            "home_pass_accuracy": snap.home.passing.accuracy,
            "away_pass_accuracy": snap.away.passing.accuracy,
            "home_sprints_window": snap.home.physical.sprint_count,
            "away_sprints_window": snap.away.physical.sprint_count,
            "momentum": snap.momentum,
        }

    # ------------------------------------------------------------------ per-second checks
    def on_second(self, tick: Tick) -> list[Trigger]:
        out: list[Trigger] = []
        minute = self._minute(tick)
        if tick.frame.ball_state == "dead" and tick.frame.t < 2:
            return out
        for tid in self.e.team_ids:
            out.extend(self._check_energy(tid, tick, minute))
            out.extend(self._check_press(tid, tick, minute))
        out.extend(self._check_momentum(tick, minute))
        out.extend(self._check_players(tick, minute))
        out.extend(self._check_summary(tick, minute))
        return out

    def _check_energy(self, tid: str, tick: Tick, minute: float) -> list[Trigger]:
        if not self.e.baseline_done or minute < 20:
            return []
        en = self.e.team_energy(tid)
        for thr in ENERGY_THRESHOLDS:
            if en.index < thr and thr not in self.energy_fired[tid]:
                self.energy_fired[tid].add(thr)
                phys = self.e.team_physical(tid)
                base = self.e.rates(self.e.baseline, self.e.team_ids.index(tid))
                tired = sorted(
                    (self.e.player_physical(p) for p in self.e.slot_pid if self.e.players[p].team_id == tid),
                    key=lambda p: p.energy,
                )[:3]
                sev = 0.5 + 0.5 * (ENERGY_THRESHOLDS.index(thr) / (len(ENERGY_THRESHOLDS) - 1))
                return [
                    self._make(
                        "fatigue_breakpoint",
                        tick,
                        f"{self._team_name(tid)} energy index drops below {thr:.0f}",
                        sev,
                        team_id=tid,
                        evidence={
                            "team": self._team_name(tid),
                            "energy_index": en.index,
                            "threshold": thr,
                            "sprint_ratio": en.sprint_ratio,
                            "hi_distance_ratio": en.hi_distance_ratio,
                            "press_ratio": en.press_ratio,
                            "sprints_last_5min": phys.sprint_count,
                            "sprints_opening_5min": round(base["sprints"], 1),
                            "hi_distance_last_5min": phys.hi_distance_m,
                            "hi_distance_opening_5min": round(base["hi_distance"], 0),
                            "compactness_m2": phys.compactness_m2,
                            "block_length_m": phys.block_length_m,
                            "defensive_line_m": phys.defensive_line_m,
                            "most_fatigued": ", ".join(f"{p.name} ({p.energy:.0f})" for p in tired),
                            "opponent_energy": self.e.team_energy(
                                self.e.team_ids[1 - self.e.team_ids.index(tid)]
                            ).index,
                            "momentum": round(self.e.momentum, 2),
                        },
                    )
                ]
        return []

    def _check_press(self, tid: str, tick: Tick, minute: float) -> list[Trigger]:
        if not self.e.baseline_done or minute < 20 or minute - self.last_press_drop[tid] < 12:
            return []
        en = self.e.team_energy(tid)
        k = self.e.team_ids.index(tid)
        base = self.e.rates(self.e.baseline, k)
        if base["press_rate"] < 4.0 or en.press_ratio > 0.6:
            return []
        self.last_press_drop[tid] = minute
        phys = self.e.team_physical(tid)
        opp = self.e.team_ids[1 - k]
        return [
            self._make(
                "press_drop",
                tick,
                f"{self._team_name(tid)}'s press is fading: pressure rate down {100 * (1 - en.press_ratio):.0f}%",
                0.6 + 0.4 * (1 - en.press_ratio),
                team_id=tid,
                evidence={
                    "team": self._team_name(tid),
                    "press_rate_now": phys.press_rate,
                    "press_rate_opening": round(base["press_rate"], 1),
                    "press_ratio": en.press_ratio,
                    "high_recoveries_last_5min": phys.high_recoveries,
                    "energy_index": en.index,
                    "opponent": self._team_name(opp),
                    "opponent_pass_accuracy": self.e.team_passing(opp).accuracy,
                    "opponent_territory": self.e.snapshot().team(opp).territory,
                    "defensive_line_m": phys.defensive_line_m,
                },
            )
        ]

    def _check_momentum(self, tick: Tick, minute: float) -> list[Trigger]:
        if len(self.e.window) < 180 or minute - self.last_swing_minute < 6:
            return []
        m = self.e.current_momentum()
        sign = 1 if m > 0.45 else -1 if m < -0.45 else 0
        if sign == 0 or sign == self.last_swing_sign:
            return []
        prev = self.last_swing_sign
        self.last_swing_sign = sign
        self.last_swing_minute = minute
        if prev == 0:
            return []  # first time a side is established is not a swing
        tid = self.e.team_ids[0 if sign > 0 else 1]
        snap = self.e.snapshot()
        return [
            self._make(
                "momentum_swing",
                tick,
                f"Momentum swings to {self._team_name(tid)}",
                0.55 + 0.45 * min(abs(m), 1.0),
                team_id=tid,
                evidence={
                    "team": self._team_name(tid),
                    "momentum": round(m, 2),
                    "territory": snap.team(tid).territory,
                    "opponent_territory": snap.opponent(tid).territory,
                    "energy_index": snap.team(tid).energy.index,
                    "opponent_energy": snap.opponent(tid).energy.index,
                    "opponent_trend": snap.opponent(tid).energy.trend,
                    "shots_last_window": snap.team(tid).shooting.shots,
                },
            )
        ]

    def _check_players(self, tick: Tick, minute: float) -> list[Trigger]:
        out: list[Trigger] = []
        f = tick.frame
        for i, pid in enumerate(f.player_ids):
            sp = float(f.speed[i])
            if sp >= SPEED_BURST_MPS and minute - self.speed_burst_at.get(pid, -1e9) >= 8:
                self.speed_burst_at[pid] = minute
                info = self.e.players[pid]
                tot = self.e.player_totals(pid)
                out.append(
                    self._make(
                        "speed_burst",
                        tick,
                        f"{info.name} hits {sp * 3.6:.1f} km/h",
                        0.3,
                        team_id=info.team_id,
                        player_id=pid,
                        evidence={
                            "player": info.name,
                            "team": self._team_name(info.team_id),
                            "speed_kmh": round(sp * 3.6, 1),
                            "top_speed_kmh": round(max(tot.top_speed, sp) * 3.6, 1),
                            "sprints": tot.sprints,
                        },
                    )
                )
            # Distance milestones are checked cheaply against the live slot accumulator.
            # Only the first player of each team to reach a mark is newsworthy.
            approx = self.e.totals[pid].distance + float(self.e.slot_distance[i])
            for km in (10, 12):
                info = self.e.players[pid]
                key = f"{info.team_id}:dist{km}"
                if approx >= km * 1000 and key not in self.milestones_done:
                    self.milestones_done.add(key)
                    tot = self.e.player_totals(pid)
                    out.append(
                        self._make(
                            "milestone",
                            tick,
                            f"{info.name} is the first {self._team_name(info.team_id)} player past {km} km",
                            0.25 + 0.1 * (km - 10) / 2,
                            team_id=info.team_id,
                            player_id=pid,
                            evidence={
                                "player": info.name,
                                "team": self._team_name(info.team_id),
                                "distance_km": round(tot.distance / 1000, 2),
                                "sprints": tot.sprints,
                                "top_speed_kmh": round(tot.top_speed * 3.6, 1),
                                "energy": self.e.player_physical(pid).energy,
                            },
                        )
                    )
        for tid in self.e.team_ids:
            p = self.e.passes[tid]
            for n in (250, 500):
                key = f"{tid}:pass{n}"
                if p.attempted >= n and key not in self.milestones_done:
                    self.milestones_done.add(key)
                    out.append(
                        self._make(
                            "milestone",
                            tick,
                            f"{self._team_name(tid)} reach {n} passes",
                            0.2,
                            team_id=tid,
                            evidence={
                                "team": self._team_name(tid),
                                "passes": p.attempted,
                                "accuracy": round(p.completed / max(p.attempted, 1), 3),
                                "hard_completed": p.hard_completed,
                                "progressive": p.progressive,
                            },
                        )
                    )
        return out

    def _check_summary(self, tick: Tick, minute: float) -> list[Trigger]:
        for mark in SUMMARY_MINUTES:
            if minute >= mark and mark not in self.summaries_done and minute < mark + 1.5:
                self.summaries_done.add(mark)
                snap = self.e.snapshot()
                return [
                    self._make(
                        "period_summary",
                        tick,
                        f"{int(mark)}' state of play: {snap.home.short_name} {snap.home.score}-{snap.away.score} {snap.away.short_name}",
                        0.35,
                        evidence={"summary_type": "state_of_play", "mark": mark} | self._summary_evidence(snap),
                    )
                ]
        return []

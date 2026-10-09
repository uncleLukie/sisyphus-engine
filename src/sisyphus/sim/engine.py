"""Agent-based synthetic football match simulator.

The simulator advances 22 players and the ball at :data:`pitch.FPS` Hz and
yields a :class:`Tick` per frame containing a tracking :class:`Frame` and any
:class:`~sisyphus.schema.Event` objects resolved during that frame. Consumers
(file writers, the live stream gateway) iterate :meth:`MatchSimulator.run`.

Behaviour model, in brief:

* **Off-ball movement**: each team slides as a block around formation anchors
  (:mod:`sisyphus.sim.tactics`) with Ornstein-Uhlenbeck jitter for realism.
* **Pressing**: the defending team commits up to ``press_players`` chasers to
  the ball carrier, with effort scaled by each player's *willingness*.
* **On-ball decisions**: the carrier periodically scores pass, shot and
  dribble options with interpretable probability models and samples one.
* **Fatigue**: a two-compartment energy model caps speed and acceleration and
  drives willingness, so the press decays over the match. The energy values
  are exposed as ``Frame.truth`` for evaluation only.

Simplifications: 2D ball, no offside, set pieces are handled as restart passes.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np

from sisyphus.schema import Event, EventType, RestartKind
from sisyphus.sim import models, pitch
from sisyphus.sim.roster import PlayerProfile
from sisyphus.sim.scenario import Scenario, SubstitutionPlan
from sisyphus.sim.stamina import StaminaModel
from sisyphus.sim.tactics import FORMATIONS, TeamStyle, kickoff_targets, shape_targets

N = 22
DT = pitch.DT
CONTROL_RADIUS = 2.5  # m: receiver must be this close to the ball on arrival
PICKUP_RADIUS = 1.2  # m: radius to collect a loose ball
TACKLE_RADIUS = 1.5  # m
PRESSURE_RADIUS = 4.0  # m


class BallState(StrEnum):
    OWNED = "owned"
    FLIGHT = "flight"
    LOOSE = "loose"
    DEAD = "dead"


@dataclass
class FrameTruth:
    """Hidden simulator state used only for evaluation (never by analytics)."""

    e_short: np.ndarray
    e_long: np.ndarray
    willingness: np.ndarray


@dataclass
class Frame:
    """Tracking snapshot for one frame. Slot ``i`` maps to ``player_ids[i]``."""

    frame: int
    period: int
    t: float
    player_ids: list[str]
    xy: np.ndarray  # (22, 2) float32 metres
    speed: np.ndarray  # (22,) float32 m/s
    ball_xy: np.ndarray  # (2,)
    ball_speed: float
    ball_state: str
    owner_id: str | None
    possession_team: str | None
    score: tuple[int, int]
    truth: FrameTruth


@dataclass
class Tick:
    frame: Frame
    events: list[Event]


@dataclass
class _Flight:
    kind: str  # "pass" | "shot"
    passer: int
    start: np.ndarray
    target: np.ndarray
    v0: float
    decel: float
    duration: float
    t_start: float
    elapsed: float = 0.0
    intended: int = -1  # receiver (success) or interceptor (fail); -1 none
    success: bool = True
    success_prob: float = 1.0
    restart: RestartKind | None = None
    under_pressure: bool = False
    receiver_hint: int = -1  # the teammate the pass was aimed at
    shot_outcome: str | None = None
    xg: float = 0.0

    @property
    def direction(self) -> np.ndarray:
        d = self.target - self.start
        return d / (np.linalg.norm(d) + 1e-9)

    def position(self) -> np.ndarray:
        s = self.v0 * self.elapsed - 0.5 * self.decel * self.elapsed**2
        return self.start + self.direction * s

    def speed(self) -> float:
        return max(self.v0 - self.decel * self.elapsed, 0.0)


@dataclass
class _DeadBall:
    kind: RestartKind
    team: int
    spot: np.ndarray
    taker: int
    ready_t: float


@dataclass
class _PendingSub:
    plan: SubstitutionPlan
    done: bool = False


@dataclass
class _Ball:
    xy: np.ndarray = field(default_factory=lambda: np.array([pitch.LENGTH / 2, pitch.WIDTH / 2]))
    vel: np.ndarray = field(default_factory=lambda: np.zeros(2))
    state: BallState = BallState.DEAD
    owner: int = -1
    flight: _Flight | None = None
    dead: _DeadBall | None = None
    last_touch_team: int = -1


class MatchSimulator:
    """Seeded, deterministic match simulator for a :class:`Scenario`."""

    def __init__(self, scenario: Scenario, seed: int | None = None, match_id: str | None = None):
        self.scenario = scenario
        self.seed = scenario.default_seed if seed is None else seed
        self.match_id = match_id or f"{scenario.scenario_id}-s{self.seed}"
        self.rng = np.random.default_rng(self.seed)
        self.stamina = StaminaModel()
        self.styles: tuple[TeamStyle, TeamStyle] = (scenario.home_style, scenario.away_style)
        self.teams = (scenario.home, scenario.away)
        self.team_ids = (scenario.home.team_id, scenario.away.team_id)

        self.profiles: list[PlayerProfile] = [*scenario.home.starters, *scenario.away.starters]
        self.team_of = np.repeat([0, 1], 11)
        self.is_gk = np.zeros(N, dtype=bool)
        self.is_gk[[0, 11]] = True
        self.anchors = np.vstack(
            [FORMATIONS[scenario.home.formation], FORMATIONS[scenario.away.formation]]
        )
        self._load_attributes()

        self.pos = np.zeros((N, 2))
        self.vel = np.zeros((N, 2))
        self.noise = np.zeros((N, 2))
        self.e_short = np.ones(N)
        self.e_long = np.ones(N)
        self.tackle_cooldown = np.zeros(N)
        self.last_pressure_t = np.full(N, -99.0)
        self.work_rate = np.repeat([self.styles[0].work_rate, self.styles[1].work_rate], 11)

        self.ball = _Ball()
        self.period = 1
        self.t = 0.0
        self.frame = 0
        self.period_end_t = scenario.period_seconds
        self.period_lengths: list[float] = []
        self.score = [0, 0]
        self.possession_team = -1
        self.possession_id = 0
        self.decision_timer = 0.0
        self.owner_immune = 0.0
        self._restart_kind: RestartKind | None = None
        self.event_seq = 0
        self.pending_subs = [_PendingSub(p) for p in scenario.substitutions]
        self._events: list[Event] = []

    # ------------------------------------------------------------------ setup
    def _load_attributes(self) -> None:
        p = self.profiles
        self.top_speed = np.array([x.top_speed for x in p])
        self.accel = np.array([x.acceleration for x in p])
        self.endurance = np.array([x.endurance for x in p])
        self.passing = np.array([x.passing for x in p])
        self.finishing = np.array([x.finishing for x in p])
        self.tackling = np.array([x.tackling for x in p])
        self.dribbling = np.array([x.dribbling for x in p])

    @property
    def player_ids(self) -> list[str]:
        return [p.player_id for p in self.profiles]

    def _team_slice(self, k: int) -> slice:
        return slice(11 * k, 11 * k + 11)

    def _kickoff_positions(self, kicking_team: int) -> None:
        for k in (0, 1):
            sl = self._team_slice(k)
            tf = kickoff_targets(self.anchors[sl])
            self.pos[sl] = pitch.from_team_frame(tf, k)
        self.vel[:] = 0.0
        # The kicking team's central forward stands at the centre spot.
        striker = 11 * kicking_team + 9
        self.pos[striker] = [pitch.LENGTH / 2 + (-0.6 if kicking_team == 0 else 0.6), pitch.WIDTH / 2]

    # ------------------------------------------------------------------ events
    def _emit(self, etype: EventType, slot: int | None = None, team: int | None = None, **kw) -> Event:
        if team is None and slot is not None:
            team = int(self.team_of[slot])
        ev = Event(
            event_id=self.event_seq,
            match_id=self.match_id,
            frame=self.frame,
            period=self.period,
            t=round(kw.pop("t", self.t), 2),
            type=etype,
            team_id=self.team_ids[team] if team is not None else None,
            player_id=self.profiles[slot].player_id if slot is not None else None,
            possession_id=self.possession_id,
            **kw,
        )
        self.event_seq += 1
        self._events.append(ev)
        return ev

    def _set_possession(self, team: int, cause: str) -> None:
        if team != self.possession_team:
            self.possession_team = team
            self.possession_id += 1
            self._emit(EventType.POSSESSION_CHANGE, team=team, qualifiers={"cause": cause})

    def _give_ball(self, slot: int, cause: str) -> None:
        b = self.ball
        b.state = BallState.OWNED
        b.owner = slot
        b.flight = None
        b.dead = None
        b.vel[:] = 0.0
        b.last_touch_team = int(self.team_of[slot])
        self._set_possession(int(self.team_of[slot]), cause)
        tempo = self.styles[self.team_of[slot]].tempo
        self.decision_timer = float(self.rng.uniform(0.5, 1.9)) / tempo
        self.owner_immune = 0.6

    # ------------------------------------------------------------------ main loop
    def run(self) -> Iterator[Tick]:
        """Simulate the full match, yielding one :class:`Tick` per frame."""
        for period in (1, 2):
            self._start_period(period)
            while True:
                ended = self._period_should_end()
                if ended:
                    self._emit(EventType.PERIOD_END, qualifiers={"score": f"{self.score[0]}-{self.score[1]}"})
                    self.period_lengths.append(round(self.t, 1))
                else:
                    self._step()
                yield Tick(self._snapshot(), self._drain_events())
                self.frame += 1
                if ended:
                    break
                self.t += DT

    def _drain_events(self) -> list[Event]:
        out, self._events = self._events, []
        return out

    def _start_period(self, period: int) -> None:
        self.period = period
        self.t = 0.0
        stoppage = self.rng.uniform(60, 180) if period == 1 else self.rng.uniform(150, 330)
        self.period_end_t = self.scenario.period_seconds + float(stoppage)
        if period == 2:
            self.stamina.halftime(self.e_short, self.e_long)
        kicking = 0 if period == 1 else 1
        self._kickoff_positions(kicking)
        self.possession_team = -1
        self._emit(EventType.PERIOD_START, qualifiers={"period": period})
        self._setup_dead_ball("kickoff", kicking, np.array([pitch.LENGTH / 2, pitch.WIDTH / 2]), delay=1.0)

    def _period_should_end(self) -> bool:
        if self.t < self.period_end_t:
            return False
        # Referees don't blow while the ball is travelling towards goal.
        return self.ball.state != BallState.FLIGHT or self.t > self.period_end_t + 8

    def _snapshot(self) -> Frame:
        b = self.ball
        if b.state == BallState.OWNED:
            ball_speed = float(np.linalg.norm(self.vel[b.owner]))
        elif b.state == BallState.FLIGHT and b.flight is not None:
            ball_speed = b.flight.speed()
        else:
            ball_speed = float(np.linalg.norm(b.vel))
        return Frame(
            frame=self.frame,
            period=self.period,
            t=round(self.t, 2),
            player_ids=self.player_ids,
            xy=self.pos.astype(np.float32),
            speed=np.linalg.norm(self.vel, axis=1).astype(np.float32),
            ball_xy=b.xy.copy(),
            ball_speed=ball_speed,
            ball_state=b.state.value,
            owner_id=self.profiles[b.owner].player_id if b.state == BallState.OWNED else None,
            possession_team=self.team_ids[self.possession_team] if self.possession_team >= 0 else None,
            score=(self.score[0], self.score[1]),
            truth=FrameTruth(
                e_short=self.e_short.astype(np.float32),
                e_long=self.e_long.astype(np.float32),
                willingness=self.stamina.willingness(self.e_short, self.e_long).astype(np.float32),
            ),
        )

    def _step(self) -> None:
        self._maybe_substitute()
        targets, urgency = self._compute_targets()
        self._move_players(targets, urgency)
        self._update_ball()
        if self.ball.state == BallState.OWNED:
            self._on_ball_phase()

    # ------------------------------------------------------------------ substitutions
    def _maybe_substitute(self) -> None:
        minute = (self.period - 1) * self.scenario.period_seconds / 60 + self.t / 60
        stoppage = self.ball.state == BallState.DEAD
        for sub in self.pending_subs:
            if sub.done or minute < sub.plan.minute:
                continue
            # Wait for a stoppage, but don't wait forever.
            if not stoppage and minute < sub.plan.minute + 3:
                continue
            k = self.team_ids.index(sub.plan.team_id)
            team = self.teams[k]
            out_p, in_p = team.player(sub.plan.out_number), team.player(sub.plan.in_number)
            slot = next(i for i in range(11 * k, 11 * k + 11) if self.profiles[i].player_id == out_p.player_id)
            self.profiles[slot] = in_p
            self._load_attributes()
            self.e_short[slot] = 1.0
            self.e_long[slot] = 1.0
            sub.done = True
            self._emit(
                EventType.SUBSTITUTION,
                slot=slot,
                qualifiers={"player_out": out_p.player_id, "player_in": in_p.player_id},
            )

    # ------------------------------------------------------------------ off-ball behaviour
    def _compute_targets(self) -> tuple[np.ndarray, np.ndarray]:
        b = self.ball
        will = self.stamina.willingness(self.e_short, self.e_long)
        targets = np.empty((N, 2))
        urgency = np.full(N, 0.42)

        # Ornstein-Uhlenbeck jitter around tactical targets (GKs jitter less).
        tau, sigma = 6.0, 3.2
        self.noise += -self.noise * DT / tau + sigma * math.sqrt(2 * DT / tau) * self.rng.standard_normal((N, 2))
        noise = self.noise.copy()
        noise[self.is_gk] *= 0.25

        kickoff = b.state == BallState.DEAD and b.dead is not None and b.dead.kind == "kickoff"
        for k in (0, 1):
            sl = self._team_slice(k)
            if kickoff:
                tf = kickoff_targets(self.anchors[sl])
                targets[sl] = pitch.from_team_frame(tf, k)
            else:
                ball_tf = pitch.to_team_frame(b.xy, k)
                in_poss = self.possession_team == k
                tf = shape_targets(self.anchors[sl], in_poss, ball_tf, self.styles[k])
                targets[sl] = pitch.from_team_frame(tf, k) + noise[sl]
                # Recovery runs: defenders far from their slot run harder if willing.
                if not in_poss:
                    far = np.linalg.norm(targets[sl] - self.pos[sl], axis=1) > 8.0
                    urgency[sl] = np.where(far, 0.45 + 0.4 * will[sl], urgency[sl])
                else:
                    far = np.linalg.norm(targets[sl] - self.pos[sl], axis=1) > 10.0
                    urgency[sl] = np.where(far, 0.62, urgency[sl])

        if b.state == BallState.OWNED:
            self._carrier_and_press_targets(targets, urgency, will)
        elif b.state == BallState.FLIGHT and b.flight is not None:
            f = b.flight
            if f.kind == "pass" and f.intended >= 0:
                targets[f.intended] = f.target
                urgency[f.intended] = 1.0
                # Nearest opponent of the intended player closes on the reception point.
                opp = 1 - self.team_of[f.intended]
                cand = self._nearest(f.target, opp, exclude_gk=True, n=1)
                for c in cand:
                    if c != f.intended:
                        targets[c] = f.target
                        urgency[c] = 0.5 + 0.45 * will[c]
        elif b.state == BallState.LOOSE:
            aim = b.xy + b.vel * 0.5
            for k in (0, 1):
                for c in self._nearest(aim, k, exclude_gk=False, n=2):
                    targets[c] = aim
                    urgency[c] = 0.7 + 0.3 * will[c]
        elif b.state == BallState.DEAD and b.dead is not None:
            targets[b.dead.taker] = b.dead.spot
            urgency[b.dead.taker] = 0.75

        return targets, urgency

    def _carrier_and_press_targets(self, targets: np.ndarray, urgency: np.ndarray, will: np.ndarray) -> None:
        b = self.ball
        c = b.owner
        k = int(self.team_of[c])
        goal = pitch.attacking_goal(k)
        to_goal = goal - self.pos[c]
        to_goal /= np.linalg.norm(to_goal) + 1e-9
        # Bias the carry forward but along the current lane rather than straight at goal.
        fwd = np.array([1.0 if k == 0 else -1.0, 0.0])
        direction = 0.65 * fwd + 0.35 * to_goal
        opp = self._nearest(self.pos[c], 1 - k, exclude_gk=False, n=1)[0]
        away = self.pos[c] - self.pos[opp]
        d_opp = float(np.linalg.norm(away))
        if d_opp < 5.0:
            direction += 0.9 * (away / (d_opp + 1e-9)) * (5.0 - d_opp) / 5.0
        direction /= np.linalg.norm(direction) + 1e-9
        targets[c] = np.clip(self.pos[c] + direction * 6.0, [1, 1], [pitch.LENGTH - 1, pitch.WIDTH - 1])
        urgency[c] = 0.55

        # Pressing by the defending team.
        d = 1 - k
        style = self.styles[d]
        ball_tf_x = float(pitch.to_team_frame(b.xy, d)[0])
        lo, hi = style.press_zone
        in_zone = lo <= ball_tf_x <= hi
        n_press = style.press_players if in_zone else 0
        cands = self._nearest(self.pos[c], d, exclude_gk=True, n=max(n_press, 1))
        anticipation = self.pos[c] + self.vel[c] * 0.4
        for i, p in enumerate(cands):
            dist = float(np.linalg.norm(self.pos[p] - self.pos[c]))
            if i < n_press and (i == 0 or (will[p] > 0.5 and dist < 22.0)):
                targets[p] = anticipation
                urgency[p] = 0.55 + 0.45 * will[p]
            elif i == 0:
                # Containment: stand off goal-side of the carrier.
                own_goal = pitch.defending_goal(d)
                v = own_goal - self.pos[c]
                targets[p] = self.pos[c] + v / (np.linalg.norm(v) + 1e-9) * 3.0
                urgency[p] = 0.5

    def _nearest(self, point: np.ndarray, team: int, exclude_gk: bool, n: int) -> list[int]:
        sl = self._team_slice(team)
        idx = np.arange(sl.start, sl.stop)
        if exclude_gk:
            idx = idx[~self.is_gk[idx]]
        d = np.linalg.norm(self.pos[idx] - point, axis=1)
        return [int(i) for i in idx[np.argsort(d)[:n]]]

    def _move_players(self, targets: np.ndarray, urgency: np.ndarray) -> None:
        vmax = self.stamina.speed_cap(self.top_speed, self.e_short, self.e_long)
        amax = self.stamina.accel_cap(self.accel, self.e_short, self.e_long)
        delta = targets - self.pos
        dist = np.linalg.norm(delta, axis=1)
        direction = delta / (dist[:, None] + 1e-9)
        desired_speed = np.minimum(urgency * vmax, 1.25 * dist)
        dv = direction * desired_speed[:, None] - self.vel
        dv_norm = np.linalg.norm(dv, axis=1)
        limit = amax * DT
        scale = np.where(dv_norm > limit, limit / (dv_norm + 1e-9), 1.0)
        dv *= scale[:, None]
        self.vel += dv
        speed = np.linalg.norm(self.vel, axis=1)
        over = speed > vmax
        self.vel[over] *= (vmax[over] / speed[over])[:, None]
        self.pos += self.vel * DT
        np.clip(self.pos, [-2.0, -2.0], [pitch.LENGTH + 2.0, pitch.WIDTH + 2.0], out=self.pos)

        speed = np.linalg.norm(self.vel, axis=1)
        accel = np.linalg.norm(dv, axis=1) / DT
        self.stamina.step(self.e_short, self.e_long, speed, accel, self.work_rate, self.endurance, DT)
        self.tackle_cooldown = np.maximum(self.tackle_cooldown - DT, 0.0)

    # ------------------------------------------------------------------ ball
    def _update_ball(self) -> None:
        b = self.ball
        if b.state == BallState.OWNED:
            v = self.vel[b.owner]
            sp = float(np.linalg.norm(v))
            lead = v / sp * 0.6 if sp > 0.3 else np.zeros(2)
            b.xy = self.pos[b.owner] + lead
            b.vel = v.copy()
        elif b.state == BallState.FLIGHT and b.flight is not None:
            f = b.flight
            f.elapsed = min(f.elapsed + DT, f.duration)
            b.xy = f.position()
            b.vel = f.direction * f.speed()
            if f.kind == "shot":
                if f.elapsed >= f.duration:
                    self._resolve_shot(f)
            else:
                if not pitch.in_bounds(b.xy):
                    self._resolve_pass_out(f)
                elif f.elapsed >= f.duration:
                    self._resolve_pass_arrival(f)
        elif b.state == BallState.LOOSE:
            sp = float(np.linalg.norm(b.vel))
            if sp > 0:
                b.vel *= max(0.0, sp - 3.0 * DT) / sp
            b.xy = b.xy + b.vel * DT
            if not pitch.in_bounds(b.xy):
                self._ball_out(b.xy)
                return
            d = np.linalg.norm(self.pos - b.xy, axis=1)
            closest = int(np.argmin(d))
            if d[closest] < PICKUP_RADIUS:
                team = int(self.team_of[closest])
                self._emit(
                    EventType.BALL_RECOVERY,
                    slot=closest,
                    start_x=round(float(b.xy[0]), 2),
                    start_y=round(float(b.xy[1]), 2),
                    outcome="success",
                    qualifiers={"regained": team != self.possession_team},
                )
                self._give_ball(closest, "recovery")
        elif b.state == BallState.DEAD and b.dead is not None:
            dead = b.dead
            b.xy = dead.spot.copy()
            b.vel[:] = 0.0
            taker_close = np.linalg.norm(self.pos[dead.taker] - dead.spot) < 1.5
            if self.t >= dead.ready_t and (taker_close or self.t > dead.ready_t + 6):
                self.pos[dead.taker] = dead.spot
                kind = dead.kind
                self._give_ball(dead.taker, kind)
                self.decision_timer = 0.0
                self._restart_kind = kind
                self.owner_immune = 2.0

    def _setup_dead_ball(self, kind: RestartKind, team: int, spot: np.ndarray, delay: float) -> None:
        spot = np.clip(spot, [0.0, 0.0], [pitch.LENGTH, pitch.WIDTH])
        if kind == "goal_kick":
            taker = 11 * team  # goalkeeper
        elif kind == "kickoff":
            taker = 11 * team + 9
        else:
            taker = self._nearest(spot, team, exclude_gk=True, n=1)[0]
        b = self.ball
        b.state = BallState.DEAD
        b.owner = -1
        b.flight = None
        b.dead = _DeadBall(kind=kind, team=team, spot=spot.astype(float), taker=taker, ready_t=self.t + delay)
        b.xy = spot.astype(float)
        self._set_possession(team, kind)

    def _ball_out(self, xy: np.ndarray) -> None:
        """Ball crossed a boundary line (not a goal): award the right restart."""
        b = self.ball
        last = b.last_touch_team if b.last_touch_team >= 0 else 0
        x, y = float(xy[0]), float(xy[1])
        self._emit(
            EventType.BALL_OUT,
            team=last,
            start_x=round(min(max(x, 0.0), pitch.LENGTH), 2),
            start_y=round(min(max(y, 0.0), pitch.WIDTH), 2),
        )
        if y < 0 or y > pitch.WIDTH:
            team = 1 - last
            spot = np.array([min(max(x, 0.5), pitch.LENGTH - 0.5), 0.0 if y < 0 else pitch.WIDTH])
            self._setup_dead_ball("throw_in", team, spot, delay=float(self.rng.uniform(4, 8)))
            return
        # Goal line: which team defends this end?
        defending = 0 if x < 0 else 1
        if last == defending:
            corner_y = 0.0 if y < pitch.WIDTH / 2 else pitch.WIDTH
            spot = np.array([0.0 if x < 0 else pitch.LENGTH, corner_y])
            self._setup_dead_ball("corner", 1 - defending, spot, delay=float(self.rng.uniform(12, 18)))
        else:
            gx = 5.5 if defending == 0 else pitch.LENGTH - 5.5
            spot = np.array([gx, pitch.WIDTH / 2 + (9.16 if y > pitch.WIDTH / 2 else -9.16)])
            self._setup_dead_ball("goal_kick", defending, spot, delay=float(self.rng.uniform(7, 12)))

    # ------------------------------------------------------------------ on-ball
    def _pressure_on(self, slot: int) -> tuple[float, float]:
        """Return (pressure 0..1, nearest opponent distance) for a player."""
        opp = self._nearest(self.pos[slot], 1 - self.team_of[slot], exclude_gk=False, n=1)[0]
        d = float(np.linalg.norm(self.pos[opp] - self.pos[slot]))
        return float(np.clip((PRESSURE_RADIUS - d) / 3.0, 0.0, 1.0)), d

    def _on_ball_phase(self) -> None:
        c = self.ball.owner
        k = int(self.team_of[c])
        self.owner_immune = max(self.owner_immune - DT, 0.0)
        self._emit_pressure_events(c, k)
        if self.owner_immune <= 0 and self._maybe_tackle(c, k):
            return
        pressure, _ = self._pressure_on(c)
        self.decision_timer -= DT * (1.0 + 1.6 * pressure) * self.styles[k].tempo
        if self.decision_timer <= 0:
            self._decide(c, k, pressure)

    def _emit_pressure_events(self, c: int, k: int) -> None:
        sl = self._team_slice(1 - k)
        for p in range(sl.start, sl.stop):
            rel = self.pos[c] - self.pos[p]
            d = float(np.linalg.norm(rel))
            if d > PRESSURE_RADIUS or self.t - self.last_pressure_t[p] < 4.0:
                continue
            closing = float(np.dot(self.vel[p] - self.vel[c], rel / (d + 1e-9)))
            if closing > 1.0:
                self.last_pressure_t[p] = self.t
                self._emit(
                    EventType.PRESSURE,
                    slot=p,
                    start_x=round(float(self.pos[p][0]), 2),
                    start_y=round(float(self.pos[p][1]), 2),
                    qualifiers={"target_player": self.profiles[c].player_id, "closing_speed": round(closing, 2)},
                )

    def _maybe_tackle(self, c: int, k: int) -> bool:
        opp = self._nearest(self.pos[c], 1 - k, exclude_gk=True, n=1)[0]
        d = float(np.linalg.norm(self.pos[opp] - self.pos[c]))
        if d > TACKLE_RADIUS or self.tackle_cooldown[opp] > 0:
            return False
        will = float(self.stamina.willingness(self.e_short[opp : opp + 1], self.e_long[opp : opp + 1])[0])
        if self.rng.random() > 0.09 + 0.09 * will:
            return False
        xy = self.pos[c]
        common = dict(start_x=round(float(xy[0]), 2), start_y=round(float(xy[1]), 2))
        foul_p = 0.09 + 0.16 * (1.0 - self.e_long[opp])
        if self.rng.random() < foul_p:
            card = self.rng.random() < 0.12
            self._emit(EventType.FOUL, slot=opp, outcome="fail", qualifiers={"fouled": self.profiles[c].player_id}, **common)
            if card:
                self._emit(EventType.CARD, slot=opp, qualifiers={"card": "yellow"}, **common)
            self.tackle_cooldown[opp] = 3.0
            self._setup_dead_ball("free_kick", k, xy.copy(), delay=float(self.rng.uniform(6, 10)))
            return True
        p_win = (
            0.40
            + 0.3 * (self.tackling[opp] - self.dribbling[c])
            + 0.25 * (self.e_short[opp] - self.e_short[c])
            + 0.1 * (self.e_long[opp] - self.e_long[c])
        )
        won = self.rng.random() < float(np.clip(p_win, 0.1, 0.85))
        self._emit(EventType.TACKLE, slot=opp, outcome="success" if won else "fail",
                   qualifiers={"opponent": self.profiles[c].player_id}, **common)
        self.tackle_cooldown[opp] = 2.0
        if not won:
            self.vel[opp] *= 0.2
            return False
        if self.rng.random() < 0.65:
            self._give_ball(opp, "tackle")
        else:
            b = self.ball
            b.state = BallState.LOOSE
            b.owner = -1
            ang = self.rng.uniform(0, 2 * math.pi)
            b.vel = np.array([math.cos(ang), math.sin(ang)]) * self.rng.uniform(4, 8)
            b.last_touch_team = 1 - k
        return True

    def _decide(self, c: int, k: int, pressure: float) -> None:
        tf_c = pitch.to_team_frame(self.pos[c], k)
        style = self.styles[k]
        restart = getattr(self, "_restart_kind", None)
        self._restart_kind = None

        options: list[tuple[str, int, float, float]] = []  # (kind, target, utility, prob)
        dist_goal = float(np.linalg.norm(pitch.attacking_goal(k) - self.pos[c]))
        if dist_goal < 30 and restart in (None, "free_kick"):
            xg = models.expected_goals(self.pos[c], k, pressure, self.finishing[c])
            if xg > 0.025:
                options.append(("shot", -1, xg * 1.25, xg))

        opp_sl = self._team_slice(1 - k)
        opp_pos = self.pos[opp_sl]
        for j in range(11 * k, 11 * k + 11):
            if j == c:
                continue
            if self.is_gk[j] and (tf_c[0] > 0.45 or pressure < 0.5):
                continue
            d = float(np.linalg.norm(self.pos[j] - self.pos[c]))
            if d < 4.0 or d > 55.0:
                continue
            lane_d, lane_t = pitch.segment_distances(opp_pos, self.pos[c], self.pos[j])
            lane_d = lane_d[lane_t > 0.05]
            lane = float(lane_d.min()) if lane_d.size else 10.0
            space = float(np.linalg.norm(opp_pos - self.pos[j], axis=1).min())
            p = models.pass_success_prob(d, lane, space, pressure, self.passing[c],
                                         self.e_short[c], self.e_long[c])
            tf_j = pitch.to_team_frame(self.pos[j], k)
            gain = models.possession_value(tf_j)
            directness = 1.0 + style.directness * max(0.0, float(tf_j[0] - tf_c[0])) * 1.5
            util = p * gain * directness - (1 - p) * 0.55 * models.turnover_cost(tf_c)
            options.append(("pass", j, util, p))

        if restart is None:
            keep = 0.92 - 0.35 * pressure
            fwd_tf = np.clip(tf_c + np.array([0.05, 0.0]), 0, 1)
            options.append(("dribble", -1, keep * models.possession_value(fwd_tf) * 0.9, keep))

        if not options:
            self.decision_timer = 0.4
            return
        utils = np.array([o[2] for o in options])
        temperature = 0.012 + 0.01 * (1.0 - self.passing[c])
        w = np.exp((utils - utils.max()) / temperature)
        choice = options[int(self.rng.choice(len(options), p=w / w.sum()))]
        kind, target, _, prob = choice
        if kind == "shot":
            self._take_shot(c, k, pressure, prob)
        elif kind == "pass":
            self._play_pass(c, target, k, pressure, prob, restart)
        else:
            self.decision_timer = float(self.rng.uniform(0.6, 1.6)) / style.tempo

    def _play_pass(self, c: int, j: int, k: int, pressure: float, prob: float, restart: RestartKind | None) -> None:
        start = self.pos[c].copy()
        dist = float(np.linalg.norm(self.pos[j] - start))
        lofted = dist > 30
        decel = 0.7 if lofted else 1.6
        v_arr = 6.0 + 0.08 * dist
        v0 = math.sqrt(v_arr**2 + 2 * decel * dist)
        t_est = dist / ((v0 + v_arr) / 2)
        target = np.clip(self.pos[j] + self.vel[j] * t_est * 0.8, [0.5, 0.5], [pitch.LENGTH - 0.5, pitch.WIDTH - 0.5])
        success = self.rng.random() < prob
        intended = j
        if not success:
            opp_sl = self._team_slice(1 - k)
            opp_idx = np.arange(opp_sl.start, opp_sl.stop)
            lane_d, lane_t = pitch.segment_distances(self.pos[opp_idx], start, target)
            score = lane_d + 3.0 * (lane_t < 0.08)
            best = int(np.argmin(score))
            if lane_d[best] < 6.0:
                intended = int(opp_idx[best])
                target = start + (target - start) * float(np.clip(lane_t[best] + self.rng.normal(0, 0.03), 0.15, 1.0))
            else:
                # Misplaced: overhit or off-line, the ball runs loose.
                intended = -1
                ang = self.rng.uniform(0, 2 * math.pi)
                target = target + np.array([math.cos(ang), math.sin(ang)]) * self.rng.uniform(6, 14)
        d_eff = float(np.linalg.norm(target - start))
        v0 = math.sqrt(v_arr**2 + 2 * decel * d_eff)
        disc = max(v0**2 - 2 * decel * d_eff, 0.0)
        duration = (v0 - math.sqrt(disc)) / decel
        self.ball.state = BallState.FLIGHT
        self.ball.owner = -1
        self.ball.last_touch_team = k
        self.ball.flight = _Flight(
            kind="pass", passer=c, start=start, target=target, v0=v0, decel=decel,
            duration=max(duration, DT), t_start=self.t, intended=intended, success=success,
            success_prob=prob, restart=restart, under_pressure=pressure > 0.33, receiver_hint=j,
        )

    def _pass_event(self, f: _Flight, outcome: str, end: np.ndarray, receiver: int | None) -> None:
        self._emit(
            EventType.PASS,
            slot=f.passer,
            t=f.t_start,
            t_end=round(self.t, 2),
            outcome=outcome,
            receiver_id=self.profiles[receiver].player_id if receiver is not None else None,
            start_x=round(float(f.start[0]), 2),
            start_y=round(float(f.start[1]), 2),
            end_x=round(float(end[0]), 2),
            end_y=round(float(end[1]), 2),
            restart=f.restart,
            under_pressure=f.under_pressure,
            pass_distance_m=round(float(np.linalg.norm(end - f.start)), 2),
            ball_speed_mps=round(f.v0, 2),
            success_prob=round(f.success_prob, 3),
            qualifiers={"intended_receiver": self.profiles[f.receiver_hint].player_id,
                        "lofted": f.decel < 1.0},
        )

    def _resolve_pass_arrival(self, f: _Flight) -> None:
        end = self.ball.xy.copy()
        k = int(self.team_of[f.passer])
        if f.intended >= 0 and np.linalg.norm(self.pos[f.intended] - end) < CONTROL_RADIUS:
            same_team = self.team_of[f.intended] == k
            self._pass_event(f, "success" if same_team else "fail", end, f.intended if same_team else None)
            if not same_team:
                self._emit(EventType.INTERCEPTION, slot=f.intended, outcome="success",
                           start_x=round(float(end[0]), 2), start_y=round(float(end[1]), 2))
                self._give_ball(f.intended, "interception")
            else:
                self._give_ball(f.intended, "pass")
            return
        # Nobody controlled it: the pass counts as incomplete and the ball runs loose.
        self._pass_event(f, "fail", end, None)
        b = self.ball
        b.state = BallState.LOOSE
        b.flight = None
        b.vel = f.direction * max(f.speed(), 2.0)

    def _resolve_pass_out(self, f: _Flight) -> None:
        end = np.clip(self.ball.xy, [0.0, 0.0], [pitch.LENGTH, pitch.WIDTH])
        self._pass_event(f, "fail", end, None)
        self._ball_out(self.ball.xy)

    def _take_shot(self, c: int, k: int, pressure: float, xg: float) -> None:
        start = self.pos[c].copy()
        gx = pitch.attacking_goal(k)[0]
        beyond = gx + (1.0 if k == 0 else -1.0)
        # Blocking: defenders close to the shot line near the shooter.
        opp_sl = self._team_slice(1 - k)
        opp_idx = np.array([i for i in range(opp_sl.start, opp_sl.stop) if not self.is_gk[i]])
        lane_d, lane_t = pitch.segment_distances(self.pos[opp_idx], start, pitch.attacking_goal(k))
        blockers = opp_idx[(lane_d < 1.6) & (lane_t < 0.5)]
        p_block = 0.32 if blockers.size else 0.05
        u = self.rng.random()
        blocked_p = p_block * (1 - xg)
        saved_p = (1 - xg - blocked_p) * 0.42
        if u < xg:
            outcome = "goal"
            target = np.array([beyond, self.rng.uniform(pitch.GOAL_Y_MIN + 0.4, pitch.GOAL_Y_MAX - 0.4)])
        elif u < xg + blocked_p:
            outcome = "blocked"
            blocker = int(blockers[0]) if blockers.size else self._nearest(start, 1 - k, True, 1)[0]
            target = self.pos[blocker].copy()
        elif u < xg + blocked_p + saved_p:
            outcome = "saved"
            target = self.pos[11 * (1 - k)].copy()
        else:
            outcome = "off_target"
            side = self.rng.choice([-1.0, 1.0])
            wide = self.rng.uniform(0.5, 6.0)
            y = pitch.WIDTH / 2 + side * (pitch.GOAL_WIDTH / 2 + wide)
            target = np.array([beyond, y])
        speed = float(self.rng.uniform(19, 31) * (0.85 + 0.15 * self.e_short[c]) * (0.9 + 0.1 * self.finishing[c]))
        dist = float(np.linalg.norm(target - start))
        decel = 1.0
        duration = (speed - math.sqrt(max(speed**2 - 2 * decel * dist, 0.0))) / decel
        self.ball.state = BallState.FLIGHT
        self.ball.owner = -1
        self.ball.last_touch_team = k
        self.ball.flight = _Flight(
            kind="shot", passer=c, start=start, target=target, v0=speed, decel=decel,
            duration=max(duration, DT), t_start=self.t,
            under_pressure=pressure > 0.33, shot_outcome=outcome, xg=xg, receiver_hint=c,
        )

    def _resolve_shot(self, f: _Flight) -> None:
        c = f.passer
        k = int(self.team_of[c])
        end = f.target
        self._emit(
            EventType.SHOT,
            slot=c,
            t=f.t_start,
            t_end=round(self.t, 2),
            outcome="success" if f.shot_outcome == "goal" else "fail",
            start_x=round(float(f.start[0]), 2),
            start_y=round(float(f.start[1]), 2),
            end_x=round(float(min(max(end[0], 0.0), pitch.LENGTH)), 2),
            end_y=round(float(end[1]), 2),
            under_pressure=f.under_pressure,
            xg=round(f.xg, 3),
            ball_speed_mps=round(f.v0, 2),
            qualifiers={"shot_outcome": f.shot_outcome or "",
                        "distance_m": round(float(np.linalg.norm(pitch.attacking_goal(k) - f.start)), 1)},
        )
        gk = 11 * (1 - k)
        if f.shot_outcome == "goal":
            self.score[k] += 1
            self._emit(EventType.GOAL, slot=c, qualifiers={"score": f"{self.score[0]}-{self.score[1]}"})
            centre = np.array([pitch.LENGTH / 2, pitch.WIDTH / 2])
            self._setup_dead_ball("kickoff", 1 - k, centre, delay=float(self.rng.uniform(35, 50)))
        elif f.shot_outcome == "saved":
            self._emit(EventType.KEEPER_SAVE, slot=gk, outcome="success",
                       start_x=round(float(self.pos[gk][0]), 2), start_y=round(float(self.pos[gk][1]), 2))
            if self.rng.random() < 0.65:
                self._give_ball(gk, "save")
            else:
                b = self.ball
                b.state = BallState.LOOSE
                b.flight = None
                ang = self.rng.uniform(-1.2, 1.2) + (math.pi if k == 0 else 0.0)
                b.vel = np.array([math.cos(ang), math.sin(ang)]) * self.rng.uniform(5, 10)
                b.last_touch_team = 1 - k
        elif f.shot_outcome == "blocked":
            b = self.ball
            b.state = BallState.LOOSE
            b.flight = None
            ang = self.rng.uniform(0, 2 * math.pi)
            b.vel = np.array([math.cos(ang), math.sin(ang)]) * self.rng.uniform(4, 9)
            b.last_touch_team = 1 - k
        else:
            self._ball_out(end)

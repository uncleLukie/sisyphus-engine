"""Two-compartment player energy model.

* ``e_short`` (anaerobic reserve, 0..1) drains quickly above the aerobic speed
  threshold and recovers within tens of seconds when the player walks or jogs.
  It governs repeated-sprint ability.
* ``e_long`` (match reserve, 0..1) drains slowly with a metabolic-load proxy
  (speed, high-intensity running and accelerations) and barely recovers during
  play. It is what makes a high press collapse late in a match.

Both reserves cap top speed and acceleration, and feed a *willingness* score
that controls how hard a player commits to pressing and recovery runs. This
is the simulator's ground truth for fatigue, which the analytics pipeline
never sees directly and must infer from tracking data.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class StaminaModel:
    aerobic_threshold: float = 4.0  # m/s (~14.4 km/h)
    short_drain: float = 0.0042  # per (m/s above threshold)^2 per second
    short_recovery_tau: float = 22.0  # seconds
    long_drain: float = 1.25e-4  # per unit metabolic load per second
    accel_cost: float = 0.12  # metabolic load per m/s^2 of acceleration
    halftime_long_recovery: float = 0.06
    willingness_midpoint: float = 0.55  # e_long at which willingness halves
    willingness_width: float = 0.055

    def step(
        self,
        e_short: np.ndarray,
        e_long: np.ndarray,
        speed: np.ndarray,
        accel: np.ndarray,
        work_rate: np.ndarray,
        endurance: np.ndarray,
        dt: float,
    ) -> None:
        """Advance both reserves in place by ``dt`` seconds."""
        over = np.maximum(speed - self.aerobic_threshold, 0.0)
        below = speed < self.aerobic_threshold
        recovery = below * (1.0 - e_short) / self.short_recovery_tau
        recovery *= 1.0 - 0.5 * speed / self.aerobic_threshold
        e_short += (recovery - self.short_drain * over**2) * dt
        np.clip(e_short, 0.0, 1.0, out=e_short)

        load = 0.2 + (speed / 7.0) ** 2 + 1.5 * (over / 4.0) ** 2 + self.accel_cost * accel
        e_long -= self.long_drain * load * work_rate / endurance * dt
        np.clip(e_long, 0.0, 1.0, out=e_long)

    def speed_cap(self, top_speed: np.ndarray, e_short: np.ndarray, e_long: np.ndarray) -> np.ndarray:
        return top_speed * (0.62 + 0.38 * e_short) * (0.78 + 0.22 * e_long)

    def accel_cap(self, accel: np.ndarray, e_short: np.ndarray, e_long: np.ndarray) -> np.ndarray:
        return accel * (0.65 + 0.35 * e_short) * (0.8 + 0.2 * e_long)

    def willingness(self, e_short: np.ndarray, e_long: np.ndarray) -> np.ndarray:
        """0..1 commitment to high-effort off-ball actions (pressing, recovery runs)."""
        z = (e_long - self.willingness_midpoint) / self.willingness_width
        long_term = 1.0 / (1.0 + np.exp(-z))
        short_term = np.clip((e_short - 0.1) / 0.4, 0.2, 1.0)
        return long_term * short_term

    def halftime(self, e_short: np.ndarray, e_long: np.ndarray) -> None:
        e_short[:] = 1.0
        np.clip(e_long + self.halftime_long_recovery, 0.0, 1.0, out=e_long)

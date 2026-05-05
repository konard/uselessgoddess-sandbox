"""
Counter-strafing logic for CS2.

Background (cited in report):

  CS2 weapons have an inaccuracy term that is roughly proportional to player
  speed. For rifles the floor begins around v <= 5 u/s; for AWP it is much
  stricter. The bot must therefore zero its lateral velocity before firing.

  CS2 ground physics (Source 2 default):
      sv_friction      = 5.2  (1/s)
      sv_accelerate    = 5.5  (1/s)
      max_walk_speed_AK = 215 u/s
      max_walk_speed_AWP = 200 u/s
  With these constants, releasing the movement key would take v(t) =
  v0 * exp(-friction * dt) — about 130 ms to reach 1 u/s. Counter-strafing
  (pressing the opposite key) accelerates deceleration and is bounded by the
  per-tick velocity update; in practice 50-80 ms is enough.

The control logic exposed below is intentionally simple and deterministic; it
is a control law, not a learned policy.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass


@dataclass
class CSPhysics:
    """Source 2 ground-movement constants verified against
    https://developer.valvesoftware.com/wiki/Sv_friction and the Source-2
    movement reverse engineering by zer0k-z (2023).
    """
    friction: float = 5.2          # sv_friction
    accelerate: float = 5.5        # sv_accelerate
    max_walk_speed: float = 215.0  # AK; M4 = 225; AWP = 200; Deagle = 230
    stop_speed: float = 80.0       # sv_stopspeed
    # 34 % of m_flMaxSpeed is the well-known "running accuracy floor"
    # below which you regain full standing accuracy on rifles.
    accuracy_floor_frac: float = 0.34
    # AWP / scoped weapons require ~0; we expose a hard cap too.
    fire_speed_threshold: float = 5.0  # u/s for AWP-class strictness


class CounterStrafer:
    """State machine that turns a desired (vx, vy) command into A/D/W/S key
    presses with counter-strafing. The bot polls .tick(now, vel) every tick
    and reads .keys."""

    def __init__(self, phys: CSPhysics = CSPhysics()):
        self.phys = phys
        # latched key state
        self.keys = {"a": False, "d": False, "w": False, "s": False}
        self._cs_until = 0.0   # absolute time when counter-strafe key release
        self._cs_axis: str | None = None  # "x" or "y"

    # ---- helpers --------------------------------------------------------

    @staticmethod
    def _release(keys: dict[str, bool], k: str) -> None:
        keys[k] = False

    @staticmethod
    def _press(keys: dict[str, bool], k: str) -> None:
        keys[k] = True

    def _press_only(self, axis: str, sign: int) -> None:
        if axis == "x":
            self.keys["a"] = sign < 0
            self.keys["d"] = sign > 0
        else:
            self.keys["s"] = sign < 0
            self.keys["w"] = sign > 0

    # ---- counter-strafe time estimation --------------------------------

    def cs_duration(self, v: float) -> float:
        """Estimate the duration (s) to press the opposite key to bring
        speed v to ~0. The Source-2 ground move applies Friction() then
        Accelerate(); pressing the opposite movement key adds

            Δv = sv_accelerate * m_flMaxSpeed * dt

        per tick (zer0k-z 2023 RE), so reaching v=0 from v0 takes

            Δt = v0 / (sv_accelerate * m_flMaxSpeed)

        For AK (v0 = 215, sv_accelerate = 5.5, MaxSpeed = 215) this is
        ~ 0.182 s in continuous time, but ground friction simultaneously
        decelerates, so the empirical 1-tick-floor at 64 Hz (15.6 ms) is
        usually enough. The heuristic below uses the conservative bound
        max(1 tick, v / (a*MaxSpeed)).
        """
        cont = abs(v) / max(1e-3, self.phys.accelerate * self.phys.max_walk_speed)
        return max(1 / 64.0, cont)

    # ---- main tick ------------------------------------------------------

    def tick(self, now: float, vx: float, vy: float,
             want_fire: bool, want_dir: tuple[int, int]) -> dict[str, bool]:
        """Update key state.

        vx, vy           : current player velocity (u/s) in player frame
        want_fire        : True if the firing controller wants to shoot now
        want_dir         : desired strafe direction as (dx, dy) in {-1,0,+1}

        Returns the (latched) key dict.
        """
        # Step 1: if we want to fire and current speed > threshold, trigger
        # counter-strafe on the active axis.
        speed = math.hypot(vx, vy)
        if want_fire and speed > self.phys.fire_speed_threshold and now >= self._cs_until:
            # which axis is dominant?
            if abs(vx) >= abs(vy):
                self._press_only("x", -int(math.copysign(1, vx)))
                self._cs_axis = "x"
            else:
                self._press_only("y", -int(math.copysign(1, vy)))
                self._cs_axis = "y"
            self._cs_until = now + self.cs_duration(speed)
            return self.keys

        # Step 2: still inside counter-strafe window? Hold the keys.
        if now < self._cs_until:
            return self.keys

        # Step 3: counter-strafe just expired — release strafe keys for the
        # 1-tick "stop frame" so inaccuracy actually drops, then resume
        # commanded direction next tick.
        if self._cs_axis is not None:
            for k in ("a", "d", "w", "s"):
                self.keys[k] = False
            self._cs_axis = None
            return self.keys

        # Step 4: normal commanded motion.
        dx, dy = want_dir
        self._press_only("x", dx if dx != 0 else 0)
        if dx == 0:
            self.keys["a"] = self.keys["d"] = False
        self._press_only("y", dy if dy != 0 else 0)
        if dy == 0:
            self.keys["s"] = self.keys["w"] = False
        return self.keys

    # ---- diagnostic -----------------------------------------------------

    def is_accurate(self, vx: float, vy: float) -> bool:
        return math.hypot(vx, vy) <= self.phys.fire_speed_threshold

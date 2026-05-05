"""
End-to-end integration: how the pieces fit together inside one tick.

This is a *reference* control loop, not production code — it's deliberately
synchronous and deterministic so the math is auditable. In a real bot you
would split this across two threads (perception + control) communicating
through a lock-free queue at 64+ Hz.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np

from .aim_trajectory import AimParams, synthesize_aim
from .counter_strafe import CSPhysics, CounterStrafer
from .navmesh_follow import NavMesh, PathFollower
from .reaction_time import FlickRT, TrackingRT
from .recoil_compensator import RecoilCompensator, RecoilParams, synthetic_ak47_table
from .sensor_fusion import TargetEstimator


@dataclass
class Perception:
    """One tick's perception bundle."""
    player_pos: np.ndarray
    player_vel: np.ndarray
    cam_yaw: float
    cam_pit: float
    repvit_yaw: float | None = None
    repvit_pit: float | None = None
    yolo_target: tuple[float, float, float, float] | None = None
    # (target_yaw_abs, target_pit_abs, conf, bbox_area_frac)


class CS2Bot:
    def __init__(self, mesh: NavMesh,
                 tick_hz: float = 64.0,
                 sens: float = 2.0):
        self.tick_hz = tick_hz
        self.dt = 1.0 / tick_hz
        self.sens = sens
        # subsystems
        self.estimator = TargetEstimator(dt=self.dt)
        self.follower = PathFollower(mesh)
        self.cs = CounterStrafer(CSPhysics())
        self.recoil = RecoilCompensator(synthetic_ak47_table(),
                                        RecoilParams())
        self.flick_rt = FlickRT()
        self.track_rt = TrackingRT()
        # state
        self.target_visible = False
        self.first_seen_at: float | None = None
        self.fire_armed_at: float | None = None
        self.aim_buffer: list[tuple[float, float]] = []
        self.last_shot_t: float | None = None

    # --------------------------------------------------------------- aim --

    def _aim_command(self, now: float, perc: Perception) -> tuple[float, float]:
        """Decide a (yaw, pitch) target for the renderer. Plans a fresh
        synthesize_aim trajectory only when the discrepancy with the current
        camera exceeds 1 deg, otherwise serves the next sample from the
        buffer."""
        target_yaw, target_pit = self.estimator.lead(latency_ms=30.0)
        if not self.aim_buffer or len(self.aim_buffer) <= 1 or \
                math.hypot(target_yaw - self.aim_buffer[-1][0],
                           target_pit - self.aim_buffer[-1][1]) > 1.0:
            traj = synthesize_aim(np.array([perc.cam_yaw, perc.cam_pit]),
                                  np.array([target_yaw, target_pit]),
                                  AimParams(tick_hz=self.tick_hz))
            self.aim_buffer = [tuple(p) for p in traj]
        next_y, next_p = self.aim_buffer.pop(0) if self.aim_buffer else (target_yaw, target_pit)
        # add recoil compensation
        ry, rp = self.recoil.compensation(now)
        return next_y + ry, next_p + rp

    # ----------------------------------------------------------- firing --

    def _want_fire(self, now: float) -> bool:
        if not self.target_visible:
            return False
        if self.fire_armed_at is None:
            return False
        if now < self.fire_armed_at:
            return False
        if self.last_shot_t is not None and now - self.last_shot_t < 0.10:
            return False  # cycle time
        # only fire when we are accurate
        v = self.estimator.kf_yaw.rate ** 2 + self.estimator.kf_pit.rate ** 2
        return v < 1.0  # squared deg/s — small means ~ on-target

    # -------------------------------------------------------------- step --

    def step(self, now: float, perc: Perception) -> dict:
        # 1. propagate filter
        self.estimator.predict()
        # 2. fold in evidence
        if perc.repvit_yaw is not None and perc.repvit_pit is not None:
            self.estimator.update_repvit(perc.repvit_yaw, perc.repvit_pit)
        if perc.yolo_target is not None:
            ty, tp, conf, area = perc.yolo_target
            self.estimator.update_yolo(ty, tp, conf, area)
            if not self.target_visible:
                self.target_visible = True
                self.first_seen_at = now
                self.fire_armed_at = now + self.flick_rt.sample()
        else:
            self.target_visible = False
            self.first_seen_at = None
            self.fire_armed_at = None
            self.recoil.reset()

        # 3. movement command
        v = self.follower.desired_velocity(perc.player_pos)
        # transform world (vx, vy) to player-frame strafe sign in {-1,0,1}
        cy = math.cos(math.radians(perc.cam_yaw))
        sy = math.sin(math.radians(perc.cam_yaw))
        forward = np.array([cy, sy, 0.0])
        right = np.array([sy, -cy, 0.0])
        f_dot = float(np.dot(v, forward))
        r_dot = float(np.dot(v, right))
        want = (int(np.sign(r_dot)) if abs(r_dot) > 5 else 0,
                int(np.sign(f_dot)) if abs(f_dot) > 5 else 0)

        # 4. counter-strafe + fire decision
        want_fire = self._want_fire(now)
        keys = self.cs.tick(now, perc.player_vel[0], perc.player_vel[1],
                            want_fire, want)

        # 5. aim command
        aim_y, aim_p = self._aim_command(now, perc)

        # 6. fire trigger
        fire = want_fire and self.cs.is_accurate(perc.player_vel[0],
                                                 perc.player_vel[1])
        if fire:
            self.recoil.on_shot(now)
            self.last_shot_t = now

        return {
            "yaw": aim_y, "pit": aim_p,
            "keys": dict(keys),
            "fire": fire,
        }

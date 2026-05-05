"""
Fusing the RepViT M1.5 nav-prior with the YOLO target-detection evidence.

Architecture choice (justified in the report):

  RepViT M1.5  ->  yaw_repvit, pitch_repvit  (low-frequency, robust prior)
                                              60-120 Hz, but laggy and
                                              positionally biased
  YOLO v26     ->  bbox center -> dyaw, dpitch  (high-frequency evidence,
                                              but only when a target is
                                              actually visible)

We need a state estimator that:
  * tracks the *target* yaw/pitch in player frame, not the camera yaw/pitch;
  * downweights YOLO measurements that come from a partial / occluded box;
  * smoothly hands off to RepViT when no target is visible (e.g. during
    pre-aim toward a corner).

A 1-D constant-acceleration Kalman filter per axis works well; the prior
from RepViT becomes an extra pseudo-measurement with its own variance.
"""

from __future__ import annotations

import numpy as np


class AimKalman1D:
    """Constant-acceleration Kalman filter for one angular axis.

    State x = [theta, theta_dot, theta_ddot]^T.
    Measurement z is the angle (degrees).
    """

    def __init__(self, dt: float = 1 / 64.0,
                 process_var: float = 200.0,         # deg^2 / s^4 (high)
                 meas_var_yolo: float = 0.06,        # deg^2 (1px @ 1080p ~0.04 deg)
                 meas_var_repvit: float = 9.0):      # deg^2 (RepViT is rough)
        self.dt = dt
        self.A = np.array([[1, dt, 0.5 * dt * dt],
                           [0, 1, dt],
                           [0, 0, 1]])
        self.H = np.array([[1.0, 0.0, 0.0]])
        # process noise (constant-acc model with random jerk)
        q = process_var
        self.Q = q * np.array([[dt**5/20, dt**4/8,  dt**3/6],
                               [dt**4/8,  dt**3/3,  dt**2/2],
                               [dt**3/6,  dt**2/2,  dt]])
        self.R_yolo = np.array([[meas_var_yolo]])
        self.R_repvit = np.array([[meas_var_repvit]])
        self.x = np.zeros((3, 1))
        self.P = np.eye(3) * 100.0

    def predict(self) -> None:
        self.x = self.A @ self.x
        self.P = self.A @ self.P @ self.A.T + self.Q

    def update(self, z: float, source: str = "yolo",
               extra_var: float = 0.0) -> None:
        R = (self.R_yolo if source == "yolo" else self.R_repvit) + extra_var
        y = np.array([[z]]) - self.H @ self.x
        S = self.H @ self.P @ self.H.T + R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        I_KH = np.eye(3) - K @ self.H
        self.P = I_KH @ self.P

    @property
    def angle(self) -> float:
        return float(self.x[0, 0])

    @property
    def rate(self) -> float:
        return float(self.x[1, 0])


class TargetEstimator:
    """Wraps two AimKalman1D filters (yaw and pitch) and runs them at the
    server tick. The interface mirrors what a CS2 bot mainloop expects:

        est = TargetEstimator()
        est.predict()                   # every tick
        est.update_repvit(yaw, pitch)   # whenever RepViT predicts
        est.update_yolo(dyaw, dpitch, conf, bbox_area_frac)
        target_yaw, target_pitch = est.target()
    """

    def __init__(self, dt: float = 1 / 64.0):
        self.kf_yaw = AimKalman1D(dt=dt)
        self.kf_pit = AimKalman1D(dt=dt)
        self._has_yolo = False
        self._frames_since_yolo = 0

    def predict(self) -> None:
        self.kf_yaw.predict()
        self.kf_pit.predict()
        self._frames_since_yolo += 1

    def update_repvit(self, yaw: float, pitch: float) -> None:
        # only let RepViT pull us when we haven't seen YOLO for a while:
        # hand-off after ~6 ticks (~95 ms at 64 Hz)
        if self._frames_since_yolo > 6:
            self.kf_yaw.update(yaw, "repvit")
            self.kf_pit.update(pitch, "repvit")

    def update_yolo(self, target_yaw_abs: float, target_pitch_abs: float,
                    confidence: float, bbox_area_frac: float) -> None:
        # Larger bbox -> better localization; small/far targets get more
        # measurement variance. Empirical scaling.
        size_var = max(0.0, 0.5 / max(bbox_area_frac, 1e-3) - 0.5)
        conf_var = max(0.0, 0.4 * (1.0 - confidence) ** 2)
        extra = size_var + conf_var
        self.kf_yaw.update(target_yaw_abs, "yolo", extra_var=extra)
        self.kf_pit.update(target_pitch_abs, "yolo", extra_var=extra)
        self._has_yolo = True
        self._frames_since_yolo = 0

    def target(self) -> tuple[float, float]:
        return self.kf_yaw.angle, self.kf_pit.angle

    def lead(self, latency_ms: float = 30.0) -> tuple[float, float]:
        """Predict where the target will be after end-to-end latency.

        Spjut et al. 2019 (NVIDIA) showed motor + display latency dominates
        FPS aim error; we lead by `latency_ms` using the velocity estimate.
        """
        s = latency_ms / 1000.0
        return (self.kf_yaw.angle + self.kf_yaw.rate * s,
                self.kf_pit.angle + self.kf_pit.rate * s)

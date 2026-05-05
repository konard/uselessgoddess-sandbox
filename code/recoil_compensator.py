"""
Recoil compensation.

CS2 uses deterministic per-weapon recoil tables (weapon_recoil_data). For each
weapon the engine stores up to ~30 frames of (yaw_offset, pitch_offset). The
n-th bullet in a continuous burst contributes a known offset; humans
"approximate" the inverse by pulling the mouse along a smoothed curve.

The class below loads a recoil table (a numpy array shape (n_bullets, 2)) and
computes the *delta* compensation needed between bullet n and bullet n+1
along with a small Gaussian noise to humanize. If your bot does not ship the
recoil tables, csgo-skills.com / leetify / many GitHub repos publish them
extracted from `weapons.txt`.

Table conventions:
  table[i] = (yaw_deg, pitch_deg) where pitch_deg is positive when the gun
  rises above the aim point. A perfect bot subtracts table[n] from its aim.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class RecoilParams:
    cycle_time_s: float = 0.10            # AK cycle 0.10 s; M4 0.0857 s
    smoothing_lag_s: float = 0.04         # human delay before pulling
    micro_noise_deg: float = 0.10         # std per shot
    inverse_gain: float = 1.0             # 1.0 = perfect compensation


class RecoilCompensator:
    def __init__(self, table: np.ndarray,
                 params: RecoilParams = RecoilParams()):
        assert table.ndim == 2 and table.shape[1] == 2
        self.table = table.astype(float)
        self.params = params
        self._burst_start: float | None = None
        self._last_shot: float | None = None
        self._bullet_n: int = 0

    # ---- API ------------------------------------------------------------

    def on_shot(self, t: float) -> None:
        if self._burst_start is None or t - (self._last_shot or 0) > 0.4:
            self._burst_start = t
            self._bullet_n = 0
        else:
            self._bullet_n += 1
        self._last_shot = t

    def reset(self) -> None:
        self._burst_start = None
        self._last_shot = None
        self._bullet_n = 0

    def compensation(self, t: float) -> tuple[float, float]:
        """Return (yaw_delta, pitch_delta) to ADD to the aim for time t.

        Pull is delayed by smoothing_lag_s and is the negative of the
        recoil_table value at the *interpolated* bullet index.
        """
        if self._burst_start is None:
            return 0.0, 0.0
        # which bullet "should" we be compensating right now?
        elapsed = t - self._burst_start - self.params.smoothing_lag_s
        if elapsed < 0:
            return 0.0, 0.0
        idx_f = elapsed / self.params.cycle_time_s
        idx_lo = int(np.floor(idx_f))
        if idx_lo >= len(self.table) - 1:
            kick = self.table[-1]
        else:
            frac = idx_f - idx_lo
            kick = (1 - frac) * self.table[idx_lo] + frac * self.table[idx_lo + 1]
        comp = -kick * self.params.inverse_gain
        # tiny gaussian noise so the path isn't a perfect inverse
        n = np.random.normal(0.0, self.params.micro_noise_deg, size=2)
        return float(comp[0] + n[0]), float(comp[1] + n[1])

    # ---- loaders --------------------------------------------------------

    @classmethod
    def from_json(cls, path: str | Path,
                  params: RecoilParams = RecoilParams()) -> "RecoilCompensator":
        data = json.loads(Path(path).read_text())
        return cls(np.array(data, dtype=float), params)


def synthetic_ak47_table(n: int = 30) -> np.ndarray:
    """A *placeholder* AK-47-shaped table for testing only. Replace with the
    real table extracted from CS2 game files for production."""
    yaw = np.zeros(n)
    pit = np.zeros(n)
    for i in range(n):
        if i < 4:
            pit[i] = 0.4 + 0.6 * (i / 3)         # rising
            yaw[i] = 0.0
        elif i < 10:
            pit[i] = 1.6 + 0.05 * (i - 4)        # plateau
            yaw[i] = 0.4 * (i - 4)               # right
        elif i < 18:
            pit[i] = 1.85
            yaw[i] = 2.0 - 0.45 * (i - 10)       # left
        else:
            pit[i] = 1.85
            yaw[i] = -1.6 + 0.25 * (i - 18) * (-1) ** i
    return np.stack([yaw, pit], axis=1)

"""
Reaction-time sampling.

The ex-Gaussian distribution (Hohle 1965; Luce 1986; Ratcliff 1979) is the
empirical workhorse for human RT: it is a convolution of a Gaussian (motor
component) with an exponential (decision component). PDF:

  f(t; mu, sigma, tau) = (1/tau) * exp((mu-t)/tau + sigma^2/(2 tau^2))
                        * Phi((t-mu)/sigma - sigma/tau)

For visual choice RT in trained CS players, typical fits are:
  mu    ~ 0.180 s
  sigma ~ 0.025 s
  tau   ~ 0.080 s
giving a median around 0.220 s and a long right tail.

Beginners shift mu by +50-100 ms.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass


@dataclass
class ExGaussianRT:
    mu: float = 0.180
    sigma: float = 0.025
    tau: float = 0.080
    floor: float = 0.120          # nothing below this in practice

    def sample(self, rng: random.Random | None = None) -> float:
        rng = rng or random
        gauss = rng.gauss(self.mu, self.sigma)
        # exponential via inverse CDF
        u = rng.random()
        expo = -self.tau * math.log(1.0 - u + 1e-12)
        return max(self.floor, gauss + expo)


@dataclass
class FlickRT(ExGaussianRT):
    """Reaction time for a clean flick (target appears, immediate aim)."""
    mu: float = 0.190
    sigma: float = 0.030
    tau: float = 0.090


@dataclass
class TrackingRT(ExGaussianRT):
    """RT for already-tracking adjustments (lower because no decision)."""
    mu: float = 0.080
    sigma: float = 0.015
    tau: float = 0.030
    floor: float = 0.045

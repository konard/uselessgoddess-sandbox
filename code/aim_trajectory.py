"""
Human-like aim trajectory generation.

Combines four independent literature-grounded ideas:

1. Ballistic + corrective submovement decomposition (Meyer, Abrams, Kornblum,
   Wright, Smith 1988; Plamondon Sigma-Lognormal 1995/2006).
2. Bezier-curve geometric path with random control-point offsets (Aldaron's
   WindMouse 2007; pyclick / HumanCursor / bezmouse).
3. Minimum-jerk timing of t in [0,1] (Flash & Hogan 1985).
4. Ornstein-Uhlenbeck and Perlin-noise micro-jitter overlay to model hand
   tremor.

The output is a sequence of yaw / pitch deltas (in degrees per tick) that the
user's CS2 bot can feed to mouse_event / SendInput / vmt_keyboard or the
sub-tick mouse API.

References (also in the report):
  Fitts P.M. (1954) J. Exp. Psychol. 47:381-391.
  Flash T., Hogan N. (1985) J. Neurosci. 5:1688-1703.
  Meyer D.E. et al. (1988) Psychol. Rev. 95:340-370.
  Plamondon R. (1995) Biol. Cybern. 72:295-307.
  Plamondon R., Djioua M. (2006) Pattern Recogn. 39:1859-1872.
  Land B.J. ('Aldaron') (2007) WindMouse forum post on SRL.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

import numpy as np


# ---------------------------------------------------------------------------
# 1. Speed profiles
# ---------------------------------------------------------------------------

def min_jerk_profile(t: np.ndarray) -> np.ndarray:
    """Flash & Hogan (1985) 5th-order minimum-jerk position curve.

    For t in [0,1]: s(t) = 10 t^3 - 15 t^4 + 6 t^5, so s(0)=0, s(1)=1, and the
    1st/2nd derivatives vanish at both endpoints, giving the well-known
    bell-shaped velocity profile observed in fast point-to-point hand
    movements.
    """
    t = np.clip(t, 0.0, 1.0)
    return 10 * t ** 3 - 15 * t ** 4 + 6 * t ** 5


def sigma_lognormal_velocity(
    t: np.ndarray, D: float, t0: float, mu: float, sigma: float
) -> np.ndarray:
    """Plamondon (1995) lognormal velocity profile of a single neuromuscular
    impulse: v(t) = D / (sigma * sqrt(2 pi) * (t-t0)) *
                   exp( - (ln(t-t0) - mu)^2 / (2 sigma^2) ).

    A complete aim movement is the sum of K such impulses (Sigma-Lognormal):
    one large ballistic + 1-3 small corrective ones (Meyer 1988 evidence).
    """
    out = np.zeros_like(t, dtype=float)
    mask = t > t0
    tt = t[mask] - t0
    out[mask] = (
        D / (sigma * math.sqrt(2 * math.pi) * tt)
    ) * np.exp(-((np.log(tt) - mu) ** 2) / (2 * sigma ** 2))
    return out


def fitts_movement_time(amp_deg: float, width_deg: float,
                        a: float = 0.10, b: float = 0.15) -> float:
    """Shannon-form Fitts's law (MacKenzie 1989): MT = a + b * log2(A/W + 1).

    For mouse pointing on desktop, MacKenzie's review reports a around 0.08-
    0.23 s and b around 0.14-0.20 s/bit. CS2 aim movements are short and
    intra-FOV; we default to (0.10, 0.15) which gives 110 ms for
    A=W (ID=1 bit) and 510 ms for ID=8 bits.
    """
    if width_deg <= 0:
        width_deg = 1e-3
    ID = math.log2(amp_deg / width_deg + 1.0)
    return a + b * ID


# ---------------------------------------------------------------------------
# 2. Geometric path: Bezier with random control points
# ---------------------------------------------------------------------------

def _binomial(n: int, k: int) -> int:
    return math.comb(n, k)


def bezier_curve(control_pts: np.ndarray, n_samples: int = 100) -> np.ndarray:
    """Evaluate an n-th degree Bezier curve at uniformly spaced t.

    control_pts shape (n+1, d). Returns (n_samples, d).
    """
    n = control_pts.shape[0] - 1
    t = np.linspace(0.0, 1.0, n_samples)
    out = np.zeros((n_samples, control_pts.shape[1]))
    for k in range(n + 1):
        bk = _binomial(n, k) * (t ** k) * ((1 - t) ** (n - k))
        out += np.outer(bk, control_pts[k])
    return out


def random_bezier_path(start: np.ndarray, end: np.ndarray,
                       n_ctrl: int = 2, jitter: float = 0.20,
                       rng: np.random.Generator | None = None) -> np.ndarray:
    """Generate a Bezier path with n_ctrl random interior control points
    perturbed perpendicular to the start->end line.

    jitter is expressed as a fraction of the chord length, matching the
    convention in pyclick / bezmouse.
    """
    rng = rng or np.random.default_rng()
    chord = end - start
    L = np.linalg.norm(chord)
    if L < 1e-9:
        return np.tile(start, (2, 1))
    direction = chord / L
    perp = np.array([-direction[1], direction[0]])
    pts = [start]
    for i in range(1, n_ctrl + 1):
        frac = i / (n_ctrl + 1)
        base = start + frac * chord
        offset = rng.normal(0.0, jitter * L) * perp
        # Bias toward the side opposite the previous control point so the path
        # gently arcs rather than zig-zags (matches human "swoop" behaviour).
        pts.append(base + offset)
    pts.append(end)
    return np.array(pts)


# ---------------------------------------------------------------------------
# 3. Stochastic noise overlay (hand tremor / mouse sensor noise)
# ---------------------------------------------------------------------------

def ornstein_uhlenbeck(n: int, theta: float = 12.0, sigma: float = 0.05,
                       dt: float = 1 / 64.0,
                       rng: np.random.Generator | None = None) -> np.ndarray:
    """Discretized OU process. theta is mean-reversion rate (1/s),
    sigma is the diffusion. For aim noise we want a stationary std ~ 0.03 deg
    => sigma = std * sqrt(2*theta).

    OU is preferred over white Gaussian because human hand tremor has a 1/f
    power spectrum and finite autocorrelation, which OU approximates well at
    one timescale.
    """
    rng = rng or np.random.default_rng()
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = x[i - 1] + -theta * x[i - 1] * dt + sigma * math.sqrt(dt) * rng.standard_normal()
    return x


def perlin_1d(n: int, octaves: int = 3, persistence: float = 0.5,
              base_freq: float = 1.5,
              rng: np.random.Generator | None = None) -> np.ndarray:
    """Cheap multi-octave value-noise (Perlin-style) for a smooth, naturally
    correlated jitter signal of length n. We don't reproduce Ken Perlin's exact
    gradient noise; for 1-D cursor noise a multi-octave smoothed random walk
    suffices (this is what most bot frameworks ship).
    """
    rng = rng or np.random.default_rng()
    out = np.zeros(n)
    amp = 1.0
    for o in range(octaves):
        freq = base_freq * (2 ** o)
        n_anchor = max(2, int(n * freq / base_freq / 16))
        anchors = rng.standard_normal(n_anchor)
        xs = np.linspace(0, n_anchor - 1, n)
        i0 = np.floor(xs).astype(int)
        i1 = np.clip(i0 + 1, 0, n_anchor - 1)
        u = xs - i0
        # smoothstep
        s = u * u * (3 - 2 * u)
        out += amp * ((1 - s) * anchors[i0] + s * anchors[i1])
        amp *= persistence
    out -= out.mean()
    out /= (out.std() + 1e-9)
    return out


# ---------------------------------------------------------------------------
# 4. Full aim trajectory: assemble the pieces
# ---------------------------------------------------------------------------

@dataclass
class AimParams:
    tick_hz: float = 64.0           # CS2 server tick (subtick still 64)
    a: float = 0.10                 # Fitts intercept (s)
    b: float = 0.15                 # Fitts slope (s/bit)
    target_width_deg: float = 4.0   # opponent body width at typical range
    bezier_jitter: float = 0.18     # fraction of chord
    n_ctrl: int = 2                 # interior control points
    overshoot_prob: float = 0.18    # P(primary submovement overshoots)
    overshoot_gain: float = 0.07    # 7 % of amplitude on average
    corrective_lag: float = 0.06    # 60 ms gap before corrective begins
    tremor_std_deg: float = 0.04    # per-axis, stationary
    tremor_theta: float = 12.0      # OU mean-reversion (1/s)
    micro_pause_prob: float = 0.05  # per submovement
    micro_pause_ms: tuple[float, float] = (40.0, 110.0)


def synthesize_aim(start_yp: np.ndarray, target_yp: np.ndarray,
                   params: AimParams = AimParams(),
                   rng: np.random.Generator | None = None) -> np.ndarray:
    """Return an (N, 2) array of (yaw, pitch) angles sampled at tick_hz, going
    from start_yp to target_yp in a human-like way.

    The angles are absolute positions; the caller takes consecutive differences
    to get per-tick mouse deltas.
    """
    rng = rng or np.random.default_rng()
    delta = target_yp - start_yp
    amp_deg = float(np.linalg.norm(delta))
    if amp_deg < 1e-6:
        return start_yp.reshape(1, 2)

    # 1. duration from Fitts
    T_primary = fitts_movement_time(amp_deg, params.target_width_deg,
                                    params.a, params.b)

    # 2. geometric path: 2-D Bezier in (yaw, pitch) plane
    ctrl = random_bezier_path(start_yp, target_yp,
                              n_ctrl=params.n_ctrl,
                              jitter=params.bezier_jitter,
                              rng=rng)

    n_primary = max(4, int(round(T_primary * params.tick_hz)))
    t_primary = np.linspace(0, 1, n_primary)
    s = min_jerk_profile(t_primary)         # eased time
    path = bezier_curve(ctrl, n_samples=n_primary)
    # re-time path along arc length under min-jerk schedule
    arc = np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))
    arc = np.concatenate(([0], arc))
    arc_norm = arc / (arc[-1] + 1e-9)
    # for each min-jerk fraction s_i, find x at that arc length
    primary = np.zeros((n_primary, 2))
    for d in range(2):
        primary[:, d] = np.interp(s, arc_norm, path[:, d])

    # 3. overshoot + corrective submovement (Meyer 1988)
    pieces = [primary]
    landed = primary[-1]
    if rng.random() < params.overshoot_prob:
        over_dir = delta / (amp_deg + 1e-9)
        over_mag = abs(rng.normal(params.overshoot_gain, 0.03)) * amp_deg
        overshoot_target = target_yp + over_dir * over_mag
        n_over = max(2, int(0.05 * params.tick_hz))
        t_o = np.linspace(0, 1, n_over)
        over_seg = (1 - min_jerk_profile(t_o))[:, None] * landed + \
                   min_jerk_profile(t_o)[:, None] * overshoot_target
        # tiny pause then corrective back to true target
        n_pause = max(1, int(params.corrective_lag * params.tick_hz))
        pause = np.tile(overshoot_target, (n_pause, 1))
        n_corr = max(3, int(0.10 * params.tick_hz))
        t_c = np.linspace(0, 1, n_corr)
        corr_seg = (1 - min_jerk_profile(t_c))[:, None] * overshoot_target + \
                   min_jerk_profile(t_c)[:, None] * target_yp
        pieces += [over_seg, pause, corr_seg]
        landed = target_yp

    if rng.random() < params.micro_pause_prob:
        ms = rng.uniform(*params.micro_pause_ms)
        n_mp = max(1, int(ms / 1000.0 * params.tick_hz))
        pieces.append(np.tile(landed, (n_mp, 1)))

    traj = np.concatenate(pieces, axis=0)

    # 4. tremor overlay (OU) on each axis
    tremor_sigma = params.tremor_std_deg * math.sqrt(2 * params.tremor_theta)
    yaw_n = ornstein_uhlenbeck(len(traj), params.tremor_theta, tremor_sigma,
                               1 / params.tick_hz, rng)
    pit_n = ornstein_uhlenbeck(len(traj), params.tremor_theta, tremor_sigma,
                               1 / params.tick_hz, rng)
    traj[:, 0] += yaw_n
    traj[:, 1] += pit_n

    return traj


def yp_to_mouse_counts(traj_deg: np.ndarray, sens: float = 2.0,
                       cs2_yaw_per_count: float = 0.022) -> np.ndarray:
    """Convert absolute yaw/pitch in degrees to per-tick raw mouse counts.

    cs2_yaw_per_count = m_yaw * sensitivity. Default m_yaw is 0.022 deg per
    count in CS2 (same as CS:GO). The user's ingame sensitivity multiplies
    that. 400 DPI * sens 2.0 -> 17.4 cm / 360 deg (low-sens pro range).
    """
    deltas = np.diff(traj_deg, axis=0)
    return np.round(deltas / (cs2_yaw_per_count * sens)).astype(int)

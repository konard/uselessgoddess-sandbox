"""Generate every figure used in CS2_Human_Like_Bot_Report.md.

Run:  python scripts/make_figures.py
Outputs go to figures/.
"""

from __future__ import annotations

import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Allow running from the repo root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from code.aim_trajectory import (  # noqa: E402
    AimParams, bezier_curve, fitts_movement_time, min_jerk_profile,
    ornstein_uhlenbeck, perlin_1d, random_bezier_path,
    sigma_lognormal_velocity, synthesize_aim,
)
from code.recoil_compensator import synthetic_ak47_table  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "figures")
os.makedirs(OUT, exist_ok=True)


def save(fig, name: str) -> None:
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


# ---------------------------------------------------------------------------
# 1. Fitts's law family
# ---------------------------------------------------------------------------

def fig_fitts() -> None:
    A = np.linspace(0.5, 30.0, 200)
    W = 4.0
    ID_fitts = np.log2(2 * A / W)
    ID_shannon = np.log2(A / W + 1)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(A, 0.10 + 0.15 * ID_fitts, label="MT  (Fitts 1954)")
    ax.plot(A, 0.10 + 0.15 * ID_shannon, label="MT  (Shannon, MacKenzie 1989)")
    ax.set_xlabel("Movement amplitude A (deg)")
    ax.set_ylabel("Predicted MT (s)")
    ax.set_title("Fitts's law for a CS2 aim flick   (W = 4 deg, a = 0.10, b = 0.15)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    save(fig, "fig_fitts.png")


# ---------------------------------------------------------------------------
# 2. Min-jerk vs Sigma-Lognormal velocity
# ---------------------------------------------------------------------------

def fig_velocity_profiles() -> None:
    T = 0.25
    t = np.linspace(0, T, 400)
    tau = t / T
    v_minjerk = (30 * tau ** 2 - 60 * tau ** 3 + 30 * tau ** 4) / T
    # Sigma-Lognormal: 1 ballistic + 1 small corrective
    v_sl = (sigma_lognormal_velocity(t, D=1.0, t0=0.0, mu=-2.0, sigma=0.30) +
            0.18 * sigma_lognormal_velocity(t, D=1.0, t0=0.16,
                                            mu=-2.6, sigma=0.20))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(t * 1000, v_minjerk / np.max(v_minjerk),
            label="Minimum jerk (Flash & Hogan 1985)")
    ax.plot(t * 1000, v_sl / np.max(v_sl),
            label="Sigma-Lognormal (Plamondon)\n1 ballistic + 1 corrective")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Tangential speed (normalised)")
    ax.set_title("Two human-like aim velocity profiles for a 250 ms flick")
    ax.legend()
    ax.grid(True, alpha=0.3)
    save(fig, "fig_velocity_profiles.png")


# ---------------------------------------------------------------------------
# 3. Bezier path with random control points + min-jerk re-timed
# ---------------------------------------------------------------------------

def fig_bezier_path() -> None:
    rng = np.random.default_rng(42)
    start = np.array([0.0, 0.0])
    end = np.array([20.0, 4.0])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    ax = axes[0]
    for k in range(6):
        ctrl = random_bezier_path(start, end, n_ctrl=2, jitter=0.18, rng=rng)
        path = bezier_curve(ctrl, n_samples=80)
        ax.plot(path[:, 0], path[:, 1], alpha=0.7, lw=1.5)
    ax.scatter([start[0], end[0]], [start[1], end[1]], c="k", zorder=5)
    ax.set_title("6 random Bezier paths\n(2 interior knots, jitter = 18 % chord)")
    ax.set_xlabel("yaw delta (deg)")
    ax.set_ylabel("pitch delta (deg)")
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal", adjustable="box")

    ax = axes[1]
    ctrl = random_bezier_path(start, end, n_ctrl=2, jitter=0.18, rng=rng)
    path = bezier_curve(ctrl, n_samples=200)
    arc = np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))
    arc = np.concatenate(([0], arc))
    arc /= arc[-1]
    n = 64
    t = np.linspace(0, 1, n)
    s = min_jerk_profile(t)
    retimed = np.zeros((n, 2))
    for d in range(2):
        retimed[:, d] = np.interp(s, arc, path[:, d])
    speed = np.linalg.norm(np.diff(retimed, axis=0), axis=1) * 64
    ax2 = ax.twinx()
    ax.plot(t[1:] * 1000, speed, color="tab:blue", label="speed (deg/s)")
    ax2.plot(t * 1000, s, color="tab:orange", lw=1, ls="--",
             label="min-jerk position")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Speed deg/s", color="tab:blue")
    ax2.set_ylabel("Position fraction", color="tab:orange")
    ax.set_title("Min-jerk re-timed Bezier path\n(64 ticks, T = 1 s for visibility)")
    ax.grid(True, alpha=0.3)

    save(fig, "fig_bezier_path.png")


# ---------------------------------------------------------------------------
# 4. OU and Perlin noise
# ---------------------------------------------------------------------------

def fig_noise() -> None:
    rng = np.random.default_rng(7)
    n = 400
    t = np.arange(n) / 64
    ou = ornstein_uhlenbeck(n, theta=12.0, sigma=0.04 * math.sqrt(24),
                            dt=1 / 64, rng=rng)
    pn = 0.04 * perlin_1d(n, octaves=4, persistence=0.55,
                          base_freq=2.0, rng=rng)
    fig, ax = plt.subplots(figsize=(7, 3.4))
    ax.plot(t * 1000, ou, label="Ornstein-Uhlenbeck (theta=12)")
    ax.plot(t * 1000, pn, label="Multi-octave value noise")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Aim noise (deg)")
    ax.set_title("Two equivalent micro-jitter overlays  (sigma ~ 0.04 deg)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    save(fig, "fig_noise.png")


# ---------------------------------------------------------------------------
# 5. End-to-end synthesised aim trajectory
# ---------------------------------------------------------------------------

def fig_synth_aim() -> None:
    rng = np.random.default_rng(2)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    ax = axes[0]
    for k in range(8):
        traj = synthesize_aim(np.array([0.0, 0.0]),
                              np.array([18.0, 2.0]),
                              params=AimParams(),
                              rng=rng)
        ax.plot(traj[:, 0], traj[:, 1], alpha=0.7, lw=1.2)
    ax.scatter([0, 18], [0, 2], c="k", zorder=5)
    ax.set_title("8 synthesised flicks (yaw, pitch)\nincluding overshoot + corrective")
    ax.set_xlabel("yaw (deg)"); ax.set_ylabel("pitch (deg)")
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal", adjustable="box")

    ax = axes[1]
    traj = synthesize_aim(np.array([0.0, 0.0]),
                          np.array([18.0, 2.0]),
                          params=AimParams(),
                          rng=rng)
    speed = np.linalg.norm(np.diff(traj, axis=0), axis=1) * 64
    t = np.arange(len(speed)) / 64 * 1000
    ax.plot(t, speed, color="tab:purple")
    ax.set_xlabel("Time (ms)"); ax.set_ylabel("Tangential speed (deg/s)")
    ax.set_title("Speed profile of one full flick")
    ax.grid(True, alpha=0.3)

    save(fig, "fig_synth_aim.png")


# ---------------------------------------------------------------------------
# 6. CS2 inaccuracy decay vs movement
# ---------------------------------------------------------------------------

def fig_cs2_inaccuracy() -> None:
    """Reproduce the verified weapons.vdata model:

       inaccuracy(t) = InaccuracyFire * exp(-t * ln(10) / RecoveryTime)
       moveInaccuracy = InaccuracyMove * (clamp(|v| / (MaxSpeed*0.34), 0, 1))^p
    """
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    ax = axes[0]
    t = np.linspace(0, 1.0, 500)
    for name, IF, RT, color in [
        ("AK-47", 0.0078, 0.368, "tab:red"),
        ("M4A4",  0.0070, 0.339, "tab:blue"),
        ("AWP (uns)", 0.05385, 0.345, "tab:green"),
    ]:
        ax.plot(t * 1000,
                IF * np.exp(-t * math.log(10) / RT) * 1000,
                color=color, label=name)
    ax.set_yscale("log")
    ax.set_xlabel("Time since last shot (ms)")
    ax.set_ylabel("Fire-induced inaccuracy (mrad)")
    ax.set_title("Recovery decay (verified weapons.vdata)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()

    ax = axes[1]
    v = np.linspace(0, 250, 500)
    for name, IM, MaxSpeed, color in [
        ("AK-47",  0.17506, 215.0, "tab:red"),
        ("M4A4",   0.13788, 225.0, "tab:blue"),
        ("AWP (uns)", 0.17648, 200.0, "tab:green"),
    ]:
        ratio = np.clip(v / (MaxSpeed * 0.34), 0, 1)
        ax.plot(v, IM * ratio * 1000, color=color, label=name)
        ax.axvline(MaxSpeed * 0.34, color=color, ls="--", alpha=0.4)
    ax.set_xlabel("Player speed (u/s)")
    ax.set_ylabel("Move inaccuracy (mrad)")
    ax.set_title("Movement inaccuracy vs speed\n(dashed = 34 % MaxSpeed accuracy floor)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    save(fig, "fig_cs2_inaccuracy.png")


# ---------------------------------------------------------------------------
# 7. Recoil pattern & compensation
# ---------------------------------------------------------------------------

def fig_recoil() -> None:
    table = synthetic_ak47_table(30)
    cum = np.cumsum(table, axis=0)
    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    ax.plot(cum[:, 0], -cum[:, 1], "o-", ms=4)
    for i in range(len(cum)):
        ax.annotate(str(i + 1), (cum[i, 0], -cum[i, 1]),
                    fontsize=7, alpha=0.7)
    ax.set_title("AK-47 recoil-shape illustration\n(synthetic — replace with extracted table)")
    ax.set_xlabel("yaw (deg, +right)"); ax.set_ylabel("pitch (deg, +up)")
    ax.invert_yaxis()
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    save(fig, "fig_recoil_pattern.png")


# ---------------------------------------------------------------------------
# 8. ex-Gaussian RT distribution
# ---------------------------------------------------------------------------

def fig_rt() -> None:
    from scipy.special import erf  # only here, avoid hard dep elsewhere

    def exg_pdf(x, mu, sigma, tau):
        a = (mu - x) / tau + sigma ** 2 / (2 * tau ** 2)
        b = (x - mu) / sigma - sigma / tau
        Phi = 0.5 * (1 + erf(b / math.sqrt(2)))
        return (1 / tau) * np.exp(a) * Phi

    x = np.linspace(0.05, 0.7, 500)
    fig, ax = plt.subplots(figsize=(7, 3.6))
    for label, mu, sigma, tau, color in [
        ("Tracking RT (μ=80, σ=15, τ=30)",  0.080, 0.015, 0.030, "tab:blue"),
        ("Flick RT  (μ=190, σ=30, τ=90)",   0.190, 0.030, 0.090, "tab:orange"),
        ("Choice RT (μ=300, σ=40, τ=120)",  0.300, 0.040, 0.120, "tab:red"),
    ]:
        ax.plot(x * 1000, exg_pdf(x, mu, sigma, tau) / 1000,
                label=label, color=color)
    ax.set_xlabel("RT (ms)")
    ax.set_ylabel("density (1/ms)")
    ax.set_title("ex-Gaussian reaction-time distributions used by the bot")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    save(fig, "fig_rt_distributions.png")


def main():
    fig_fitts()
    fig_velocity_profiles()
    fig_bezier_path()
    fig_noise()
    fig_synth_aim()
    fig_cs2_inaccuracy()
    fig_recoil()
    fig_rt()
    print("done.")


if __name__ == "__main__":
    main()

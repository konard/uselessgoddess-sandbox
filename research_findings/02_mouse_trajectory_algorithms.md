# Mouse trajectory generation — grounded findings

(All citations verified by sub-agent.)

## WindMouse (BenLand100 / Aldaron, 2007)
- Author retrospective with full Python: https://ben.land/post/2021/04/25/windmouse-human-mouse-movement/
- Original Java in SMART project: https://github.com/BenLand100/SMART/blob/master/src/EventNazi.java (`windMouseImpl` ~lines 341–395)
- Ports: https://github.com/AsfhtgkDavid/windmouse (Python), https://github.com/MostlyFinished/windmouse-rs (Rust)

Model: cursor as particle under constant-magnitude **gravity** toward target plus stochastic **wind**.

```
G(x) = G₀ · (x_f − x) / |x_f − x|

if dist >= D₀:
    W_x = W_x/√3 + (2·rand()−1) · W₀/√5
    W_y = W_y/√3 + (2·rand()−1) · W₀/√5
else:
    W_x /= √3; W_y /= √3
    M_eff = max(3, M₀/√5)
v_x += W_x + G₀·(x_f − x)/dist
v_y += W_y + G₀·(y_f − y)/dist
v_mag = sqrt(v_x² + v_y²)
if v_mag > M_eff:
    v_clip = M_eff/2 + rand()·M_eff/2
    v_x *= v_clip / v_mag
    v_y *= v_clip / v_mag
x += v_x; y += v_y
emit_pixel(round(x), round(y))
```

Defaults: `G₀=9, W₀=3, M₀=15, D₀=12 px`.

## Open-source Bezier mouse libs
Common pattern: knots between start/end, cubic/N-degree Bezier, easing on t, optional Gaussian distortion + overshoot.

- **pyclick**: https://github.com/patrikoss/pyclick
  - `generateInternalKnots(left,right,down,up, knotsCount=2)`; defaults offset=100.
  - `distortPoints(distortionMean=1, distortionStdev=1, distortionFrequency=0.5)`.
  - `tweenPoints(tween=pytweening.easeOutQuad, targetPoints=100)` for non-uniform t.

- **ghost-cursor (TS)**: https://github.com/Xetera/ghost-cursor
  - `bezierCurve(start,end,spreadOverride)`; `getLUT(steps)`.
  - Fitts: `time = a + b·log2(distance/width + 1)` with `a=0, b=2`.
  - `steps = ceil((log2(fitts(length,width)+1) + baseTime)·3)`, MIN_STEPS=25.
  - Overshoot: when distance>500 px, OVERSHOOT_RADIUS=120 px, spread=10 px; then correct.

- **HumanCursor**: https://github.com/riflosnake/HumanCursor (PyAutoGUI/Selenium variants).
- **bezmouse**: https://github.com/vincentbavitz/bezmouse — `connected_bez(points, deviation)`; deviation 0–100% of segment length, default 30.
- **Python ghost-cursor**: https://github.com/mcolella14/python_ghost_cursor

## Minimum-jerk re-timing (Flash & Hogan 1985)
- *J. Neurosci.* 5(7):1688–1703 — full text PDF: https://www.jneurosci.org/content/jneuro/5/7/1688.full.pdf
- UCSD MPLab tutorial: https://inc.ucsd.edu/mplab/75/media//minimumJerk.pdf
- s(τ) = 10τ³ − 15τ⁴ + 6τ⁵ ; v(τ) ∝ 30τ²(1−τ)²
- Constant-jerk simpler S-curve: s(τ) = 3τ² − 2τ³.

Re-timing recipe: build spatial Bezier P(u); arc-length parameterize to Q(s); sample Q(s(τ)) at uniform τ. Result: bell-shaped tangential velocity matching human empirics.

## GAN/Diffusion mouse synthesis
- BeCAPTCHA-Mouse (Acien, Morales et al. 2020/2022, *Pattern Recognition*): https://arxiv.org/abs/2005.00890 ; https://github.com/BiDAlab/BeCAPTCHA-Mouse
  - Uses Plamondon's Sigma-Lognormal kinematic theory features.
  - Synthesis methods: heuristic (constant / log / Gaussian velocity profiles) and a GAN.
  - Detectors: SVM-RBF, KNN, RF, MLP, LSTM, GRU. ~93% from a single trajectory; +36% relative gain when neuromotor features added.
- DMTG (Liu 2024): https://arxiv.org/abs/2410.18233 — entropy-controlled diffusion; reduces commercial-detector accuracy by 4.75–9.73%.
- Sigma-Lognormal review: https://link.springer.com/article/10.1007/s12559-020-09803-8

## Bot-discriminating features (consensus)
- Tangential velocity / accel / jerk stats (mean, std, max, skew, kurt)
- Angular velocity, curvature, path-vs-displacement ratio
- Initial acceleration + final corrective sub-movement (Sigma-Lognormal "fine correction")
- Inter-event timing: micro-pauses, non-uniform Δt
- 1/f spectral content; pure Bezier paths too smooth → detectable by jerk RMS / spectral
- Integer pixel deltas with HID-driven jitter (sub-pixel float without rounding is a giveaway)
- Image-based representation (CNN over heat-map): ~99.78% bot detection (Expert Systems 2023)

## Aimbot tuning parameters (engineering vocabulary)
- Witschel & Wressnegger 2020 (EuroSec): https://arxiv.org/abs/2004.12183
  - Smoothing/lerp factor s∈[0,1]: `aim ← aim + s·(target − aim)`. s=1 snap; humans ≈ 0.05–0.25 per 100–250 ms.
  - FOV cone clamp θ: only engage when angle ≤ θ; anti-cheats flag fixed θ.
  - Prediction lead Δt: `target ← target + v_target · Δt`; humans lead inconsistently.

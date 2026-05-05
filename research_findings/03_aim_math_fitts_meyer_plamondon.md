# Aim mathematics — Fitts, Meyer, Plamondon, Flash & Hogan, ex-Gaussian

## Fitts's law (1954) and Shannon variant
- Fitts 1954 J. Exp. Psychol. 47(6):381–391 — http://www2.psychology.uiowa.edu/faculty/mordkoff/infoproc/pdfs/Fitts%201954.pdf
- MacKenzie 1989 J. Motor Behav. 21(3):323–330 — https://www.yorku.ca/mack/JMB89.html
- Soukoreff & MacKenzie 2004 IJHCS 61:751–789 — https://www.yorku.ca/mack/ijhcs2004.pdf

Equations:
```
MT = a + b · log2(2A/W)            (Fitts 1954)
MT = a + b · log2(A/W + 1)         (Shannon variant; MacKenzie 1989; ISO 9241-9)
```
Mouse throughput (Soukoreff & MacKenzie 2004 Table 5): 3.7–4.9 bits/s (ISO).
Practical defaults: a ∈ 0–230 ms, b ∈ 100–250 ms/bit. Throughput TP ≈ 4 bits/s.

## Crossman & Goodeve (1963/1983)
- QJEP A 35(2):251–278 — https://journals.sagepub.com/doi/abs/10.1080/14640748308402133
- Deterministic iterative-correction; reproduces MT ∝ log2(A/W).

## Meyer, Abrams, Kornblum, Wright & Smith (1988) — WHo model
- Psychol. Rev. 95(3):340–370. PubMed 3406245. UMich PDF: https://public.websites.umich.edu/~kornblum/files/psychological_rev_95-3.pdf
- "Stochastic optimized-submovement model": primary submovement + optional secondary corrective submovement, programmed to minimise expected MT subject to a target-hit frequency constraint.
- Implies: square-root speed–accuracy at short distances; asymmetric velocity profile (fast launch + slower correction).

## Plamondon's kinematic theory (1995); Sigma-Lognormal (2006/2009)
- Plamondon 1995 Biol. Cybern. 72:295–307 (Part I) and 309–320 (Part II) — https://link.springer.com/article/10.1007/BF00202785
- Plamondon & Djioua 2006 Hum. Mov. Sci. 25:586–607.
- O'Reilly & Plamondon 2009 Pattern Recognit. 42:3324–3337.
- Review: PMC3867641 — https://pmc.ncbi.nlm.nih.gov/articles/PMC3867641/

Single-impulse lognormal velocity:
```
|v_i(t)| = D_i · Λ(t; t0_i, μ_i, σ_i)
        = D_i / (σ_i · √(2π) · (t − t0_i)) · exp[ −(ln(t−t0_i) − μ_i)² / (2 σ_i²) ]
```
Sigma-Lognormal: trajectory = vector sum of N lognormal strokes with stroke-specific (D, t0, μ, σ, θ_s, θ_e).

Practical fit ranges for short flicks: σ ≈ 0.1–0.5; μ ≈ −1.5 to −2.5; t0 50–150 ms after command; 1–3 strokes.

## Flash & Hogan (1985) — minimum jerk
- J. Neurosci. 5(7):1688–1703 — https://www.jneurosci.org/content/5/7/1688
- Cost J = ∫₀ᵀ (d³x/dt³)² dt. Solution:
```
x(t) = x0 + (x1 − x0) · (10 τ³ − 15 τ⁴ + 6 τ⁵),   τ = t/T
v(t) = (x1−x0)/T · (30 τ² − 60 τ³ + 30 τ⁴)
v_max = 1.875 · (x1−x0)/T at τ = 0.5
```
Typical mouse-flick T = 150–400 ms.

## Two-thirds power law (Lacquaniti, Terzuolo, Viviani 1983)
- Acta Psychol. 54(1-3):115–130. Review PMC4416550, Zago et al. 2018 (Springer EBR).
- ω(t) = K · κ(t)^(2/3); equivalently v(t) = γ · κ(t)^(−1/3).
- Applies to continuous curved drawing/scribbling, smooth-pursuit eye movements; *not* to ballistic straight aim.
- For a CS2 bot: use it on tracking arcs around moving targets; speed must drop on tight curls.

## Reaction time — ex-Gaussian
- Hohle 1965 J. Exp. Psychol. 69:382–386. Luce 1986 *Response Times* (Oxford).
- Modern: Matzke & Wagenmakers 2009 Psychon. Bull. Rev. 16:798–817 — https://www.ejwagenmakers.com/2009/MatzkeWagenmakers2009.pdf

PDF:
```
f(x; μ, σ, τ) = (1/τ) · exp[(μ−x)/τ + σ²/(2τ²)] · Φ((x−μ)/σ − σ/τ)
```
Mean = μ + τ; variance = σ² + τ².

Empirical (visual choice RT):
- Simple visual: μ ≈ 200–300, σ ≈ 20–40, τ ≈ 40–80 ms (median 220–340).
- 2-/4-choice: μ ≈ 300–500, σ ≈ 30–60, τ ≈ 60–150 ms.
- For a competitive CS2 bot: ex-Gaussian(μ=200, σ=25, τ=50) ms (median ~250 ms, occasional long tails).

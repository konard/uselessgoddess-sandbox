# CS2 movement / accuracy / firing — grounded findings

(All numbers verified against SteamDatabase GameTracking-CS2 weapons.vdata + community guides.)

## Inaccuracy fields (from `weapons.vdata`, half-cone radians except `MaxSpeed` u/s)
| Weapon | Stand | Crouch | Move | Fire | JumpInitial | RecoveryTimeStand | RecoveryTimeCrouch | MaxSpeed |
|---|---|---|---|---|---|---|---|---|
| AK-47 | 0.00641 | 0.00481 | 0.17506 | 0.0078 | 0.10094 | 0.368 → 0.506 | 0.305 → 0.420 | 215 |
| M4A4 | 0.0049 | 0.00410 | 0.13788 | 0.0070 | 0.09441 | 0.339 → 0.466 | 0.242 → 0.333 | 225 |
| M4A1-S | 0.0049 | 0.00410 | 0.09288 | 0.0120 | 0.09677 | 0.339 → 0.466 | 0.242 → 0.333 | 225 |
| AWP (un/scoped) | 0.0808 / 0.002 | 0.0606 / 0.0015 | 0.17648 | 0.05385 | 0.17286 | 0.345 | 0.247 | 200 / 100 |
| Deagle | 0.0042 | 0.00218 | 0.0481 | 0.07223 | 0.54882 | 0.8112 | 0.4499 | 230 |

Source: https://github.com/SteamDatabase/GameTracking-CS2/blob/master/game/csgo/pak01_dir/scripts/weapons.vdata
Cross-checks: https://counterstrike.fandom.com/wiki/Inaccuracy ; https://csmarket.gg/blog/cs2-weapon-accuracy-what-you-need-to-know/

## Decay model
```
inaccuracy(t) = InaccuracyFire · exp(−t · ln(10) / RecoveryTime)
```
i.e. RecoveryTime is the time for inaccuracy to drop by a factor of 10.
Recovery time is bullet-indexed: uses RecoveryTimeStand until `m_nRecoveryTransitionStartBullet` (=2 rifles), linearly interpolates to `RecoveryTimeStandFinal` by `m_nRecoveryTransitionEndBullet` (=5 rifles).

## Movement accuracy floor
```
ratio = clamp(|v| / (MaxSpeed · 0.34), 0, 1)
moveInaccuracy = InaccuracyMove · ratio^p     (p ≈ 1.0 in CS:GO, same in CS2)
```
Below 34% MaxSpeed (≈73 u/s for AK; 76.5 for M4) you regain full standing accuracy.

## Counter-strafe math
- `sv_friction = 5.2`, `sv_accelerate = 5.5`, `sv_stopspeed = 80` (Source 2 defaults).
- Passive friction per tick (when |v| > stopspeed): `v ← v · (1 − sv_friction · dt)`. At 64 Hz dt = 1/64, decay ~8%/tick → ~155 ms to fall under the 34% threshold.
- Counter-strafe: pressing opposite key applies `sv_accelerate · MaxSpeed · dt = 5.5 · 215 / 64 ≈ 18.5 u/tick` for AK → crosses 0 in 1–2 ticks (15.6–31 ms). Real human window 40–80 ms due to input/system latency.
- Refs: https://developer.valvesoftware.com/wiki/Sv_friction ; https://gist.github.com/zer0k-z/808bc8bfc494e0bbb5a423c2b1ca6685

## AWP-specific
Unscoped `InaccuracyMove = 0.176` vs scoped `InaccuracyStand = 0.002`. Velocity must be effectively 0 (≤5 u/s practical). `cl_weapon_debug_show_accuracy` shows live cone.

## Recoil
Deterministic from `m_nRecoilSeed` + `m_flRecoilAngle` + `m_flRecoilMagnitude` (and variances). Same seed → same sequence. AK seed = 223; M4A4 = 38965; M4A1-S = 38965; AWP = 4100; Deagle = 1454.

AK-47 pattern shape:
- B1–B4: nearly straight up
- B5–B7: up + slight right
- B8–B14: hard pull left
- B15–B25: oscillation right/left, decaying

Compensation = negative cumulative offset table by bullet n. Standard production approach: extract the table by running `weapon_accuracy_nospread 1` and `cl_weapon_debug_show_accuracy 3`, store as JSON.

## Subtick movement (March 2023)
- Server still 64 tick, but inputs carry sub-tick fractional timestamps `when_pressed`, `when_released`, `when_fired`.
- Firing accuracy is computed using player velocity at the *exact sub-tick instant* of the click.
- Implication: predict velocity at the queued fire time, not at the most recent tick snapshot.
- Refs: https://www.counter-strike.net/news/movingbeyondticks ; https://www.dexerto.com/counter-strike-2/counter-strike-2-sub-tick-updates-explained-2094004/

## Latency / reaction
- Pro CS reaction (go/no-go): 180–230 ms; <150 ms is "elite".
- Spjut et al., NVIDIA, "Latency of 30 ms benefits FPS targeting" (SIGGRAPH Asia 2019) — https://research.nvidia.com/sites/default/files/pubs/2019-11_Latency-of-30/FPS_Tasks_sa2019_AuthorVersion.pdf : ~30% faster acquisition going from 50→30 ms; hit rate monotonically improves under 30 ms.
- Practical end-to-end target for a competitive bot: ≤30 ms.

## Tap vs burst vs spray ranges
- Tap (>25–30 m): wait full RecoveryTimeStand (AK 0.37 s) per shot. AK first-shot Stand inaccuracy 0.00641 rad ≈ 16 cm circle at 25 m — head-shot reliable.
- Burst (12–25 m): 2–4 shots @ cycle time 0.1 s for AK. Compound inaccuracy ≈ Stand + Fire·exp(−ΔT/Recovery).
- Spray (≤12 m): pull on the recoil pattern; TTK > per-shot accuracy.

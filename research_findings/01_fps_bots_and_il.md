# FPS bots / IL / RL — grounded findings

(All citations verified by sub-agent.)

## Pearce & Zhu 2022 — Counter-Strike Deathmatch with Large-Scale Behavioural Cloning
- arXiv: https://arxiv.org/abs/2104.04258 (IEEE CoG 2022).
- Architecture: EfficientNetB0 first six residual stages → 18×10×112 feature map from 280×150×3 frames; ConvLSTM; linear head; ImageNet pre-training.
- Action tokenization (directly transferable):
  - mouse-x: 19 unequal bins {-300,-200,-100,-50,-20,-10,-4,-2,0,2,4,10,20,50,100,200,300} (plus 2 extra)
  - mouse-y: 13 unequal bins {-50,…,50}
  - keys: WASD, fire, jump, reload, weapon-switch (binary)
- Data: ~4M frames scraped + small expert set. KPM 3.72 vs medium bot 2.41; K/D ≈ 2.09.

## Pearce et al. 2023 — Imitating Human Behaviour with Diffusion Models
- arXiv: https://arxiv.org/abs/2301.10677 (ICLR 2023).
- Diffusion policy as drop-in replacement for softmax head; better Hellinger distance to human action distribution.
- Captures multi-modal joint correlations (e.g., aim + strafe).

## Jaderberg et al. 2019 — Quake III CTF (Science 364:859–865)
- DOI: https://doi.org/10.1126/science.aau6249 ; preprint https://arxiv.org/abs/1807.01281
- Two-timescale LSTM + DNC-like memory; PBT of ~30 agents. Beats strong humans even at throttled 257 ms reaction time.
- Methodology of throttling RT to keep agents fair is the standard for "fairness."

## OpenAI Five (2019) and VPT (2022)
- Five: https://arxiv.org/abs/1912.06680 — 4096-LSTM, ~159M params, PPO self-play.
- VPT: https://arxiv.org/abs/2206.11795 — Inverse Dynamics Model trained on labelled play, then pseudo-labels ~70k h of YouTube; first agent to craft diamond pickaxe. **The IDM-then-pseudolabel pipeline is the canonical scaling recipe** for unlabelled CS2 streamer footage.

## ViZDoom and FPS-RL line
- Kempka et al. 2016: https://arxiv.org/abs/1605.02097
- Lample & Chaplot 2017 (Arnold): https://arxiv.org/abs/1609.05521 — auxiliary "is enemy visible?" head during training only; cheap inductive bias that helps perception.

## Anti-cheat / aim-bot detection ML
- Galli et al., 1D-CNN over mouse/keyboard time series, 99.2/98.9% accuracy on triggerbot/aimbot. https://link.springer.com/article/10.1007/s10994-021-06055-x
- HAWK (Zhang et al. 2024): https://arxiv.org/abs/2409.14830 — server-side ML detector for CS:GO.
- Liu et al. 2021 vision-based: https://arxiv.org/abs/2103.10031
- Witschel & Wressnegger 2020 "Aim Low, Shoot High": https://arxiv.org/abs/2004.12183 — adaptive aimbot evading VAC/VACnet/Overwatch.
- These define the empirical specification of "human-like."

## IRL of aim style
- Ziebart 2008 MaxEnt-IRL: https://cdn.aaai.org/AAAI/2008/AAAI08-227.pdf
- For aim: Pearce 2023 diffusion is the de facto "human-kinematics-aware" route; can also train a discriminator and use as reward bonus.

## RepViT / YOLOv8/v10 (brief)
- RepViT 2023: https://arxiv.org/abs/2307.09283 — >80% top-1 ImageNet at ~1.0 ms on iPhone 12; good lightweight encoder.
- YOLOv8: https://docs.ultralytics.com/models/yolov8/
- YOLOv10: https://docs.ultralytics.com/models/yolov10/ (NMS-free end-to-end head, lower latency)

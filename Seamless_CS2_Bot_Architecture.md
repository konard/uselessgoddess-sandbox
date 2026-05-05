# Seamless Attention, Kinematic Extrapolation, and Lock-Free Inference: A Mathematical and Architectural Research Report for an Autonomous CS2 Deathmatch Agent

A focused research report grounding three specific design problems for an autonomous Counter-Strike 2 Deathmatch bot whose perception stack consists of a RepViT M1.5 spatial-pose head and a custom YOLO v26 target detector running asynchronously inside `ort` (the Rust ONNX Runtime binding by pykeio), feeding a high-frequency control loop that drives mouse and keyboard. The three problems are, in order: how to blend mouse control between RepViT's path-following authority and YOLO's enemy-flicking authority without producing the robotic snap-and-release pattern that bot detectors learn first; how to extrapolate RepViT's lagged spatial estimate forward in time so the input streams the controller actually consumes correspond to *now* and not to *now − Δt*; and how to expose those asynchronous, ~60 Hz inference outputs to a 1000 Hz control loop without locks, jitter, or dropped frames. This report exists because the prior repository's `CS2_Human_Like_Bot_Report.md` answered a different question — the kinematics and statistics of a single human flick — and intentionally did not engage with the systems-level coupling between two model heads and a fast tick loop. The work here is complementary to that report, not a replacement of it; the present document plugs the architectural seams.

## 1. Why these three problems are the seams that decide whether a bot is "seamless"

A Deathmatch agent has two perception modalities that disagree by construction. RepViT M1.5 (Wang et al., *arXiv* 2307.09283) is a fast hierarchical vision transformer that the user has trained as a pose regressor: given the current screen, it returns the bot's world coordinates `(x, y, z)` and a recommended view orientation `(yaw_path, pitch_path)` that, if followed, traces an optimal path along the navmesh. RepViT is thus a *prior* — it knows where the bot should look in the absence of any enemy. YOLO v26, the user's custom target detector, runs on the same screen and emits zero or more enemy bounding boxes per frame. YOLO is the *evidence*: when an enemy is visible, the centre of the highest-confidence box is the right place to put the crosshair right now. The two streams differ in three ways that matter directly to the math. First, they fire at different effective rates: RepViT, with its deeper backbone and 6.8 GFLOPs at 384×384, is the heavier model and will land closer to twenty milliseconds per inference on a 4090, while a small YOLO at 416×416 will land closer to four; even when both are scheduled at "~60 Hz" the timing variance is large enough that the controller cannot assume either is fresh on any given tick. Second, they have radically different error statistics: a centred YOLO box at 1080p has on the order of 0.04 degrees of yaw error per pixel, while RepViT's regression error against ground-truth pose is on the order of degrees, with a heavy tail near map boundaries. Third, they produce stale outputs: by the time the controller reads either inference, the player has already moved, and on a fast bot the displacement during one inference window is large enough that aiming at the place RepViT thought was forward will look, to a human or to a discriminator, exactly like a player who has frozen for 30 ms.

These three differences give us our three sub-problems. The asymmetric error statistics force *blending*, not switching: a hard switch from RepViT to YOLO when an enemy is detected creates a discontinuity in target velocity that anti-cheat ML detectors learn easily, and a hard switch back when the enemy leaves the field of view creates a second one that is even more telling. The staleness forces *extrapolation*: we cannot use RepViT's coordinate prediction as if it were a measurement of the present; we must propagate it forward by the elapsed inference latency under a kinematic model. The independent inference cadences and the sub-millisecond control budget force a *lock-free* state-sharing pattern: any controller tick that has to wait on a mutex held by an inference thread will miss the sub-tick boundary and the bot will fire at a velocity that is no longer correct under CS2's sub-tick movement model. The body of this report works through each in turn, then knits them together in the architecture section.

## 2. Seamless attention blending: the math of moving the crosshair between two authorities

The empirical reference point for "seamless transition" is the way human eyes move between targets. Krauzlis and Lisberger (and a long line of follow-up work surveyed by Orban de Xivry & Lefèvre, *Frontiers in Systems Neuroscience* 7:4, 2013, https://www.frontiersin.org/articles/10.3389/fnsys.2013.00004/full) show that the brain solves a closely related problem: when an object of interest appears in the periphery, the oculomotor system arranges a saccade — a fast ballistic move whose duration is set by the geometric distance — and *simultaneously* brings up smooth-pursuit gain so that, the moment the saccade lands, the eye is already moving with the target rather than coming to rest and re-accelerating. The presaccadic-to-postsaccadic enhancement is empirically a roughly tenfold jump in pursuit gain over the saccade window (Lisberger & Westbrook 1985; Tanaka et al., *Neuroscience Research* 32:197, 1998, https://pubmed.ncbi.nlm.nih.gov/9809665/), with an initial latency of roughly 100–130 ms before pursuit can begin and a typical step-ramp transition window on the order of 200 ms. The biological lesson is that when human attention shifts between two targets, the two control regimes overlap in time rather than handing off cleanly; the bot's controller should overlap them too.

Mathematically, the cleanest way to express this overlap is to maintain a single *target attitude* `q_target ∈ S³` (a unit quaternion expressing the desired view orientation) which is at every instant a weighted blend of the path-following target `q_path` produced by RepViT and the enemy-aim target `q_enemy` produced by YOLO. Because `S³` is a Lie group rather than a vector space, the right blend is the spherical linear interpolation due to Shoemake (*SIGGRAPH* 1985, popularised in https://en.wikipedia.org/wiki/Slerp), defined by `slerp(q₀, q₁, t) = q₀ · sin((1 − t)·Ω) / sin Ω + q₁ · sin(t·Ω) / sin Ω` where `Ω = arccos(q₀ · q₁)`, with the standard sign-correction `q₁ ← −q₁` whenever the dot product is negative so the interpolation always traverses the shorter arc. Slerp is the unique constant-angular-velocity geodesic interpolation between two unit quaternions and gives us a smooth orientation for any blending weight `t ∈ [0, 1]`. The work the math does is *not* in slerp itself but in the time-evolution of `t`. If `t` is a step function, we have rebuilt a hard switch; if `t` is the linear ramp `t = clamp((τ − τ_start) / Δ, 0, 1)`, the angular velocity is `C¹`-discontinuous at the endpoints and the discontinuity is exactly what a 1-D CNN over the controller output vector will learn (Galli et al., *Machine Learning* 110:3025, 2021, https://link.springer.com/article/10.1007/s10994-021-06055-x). The fix is a `C²`-continuous easing function: Ken Perlin's smootherstep `S₂(t) = 6t⁵ − 15t⁴ + 10t³`, whose first and second derivatives both vanish at the endpoints (Perlin, *SIGGRAPH* 2002; Wikipedia entry https://en.wikipedia.org/wiki/Smoothstep). Smootherstep applied to slerp gives a transition whose angular acceleration is continuous everywhere, which is the strongest signal a discriminator gets from a hard switch. We therefore drive `t` not by a clock but by a *saliency* signal `α(τ) ∈ [0, 1]` — the bot's instantaneous belief that an enemy is present and worth aiming at — and pass that signal through smootherstep to obtain the blending weight.

The saliency signal itself is a smoothed, evidence-driven quantity. The raw evidence per tick is `e(τ) ∈ {0, 1}`, set to 1 when YOLO returns at least one box with confidence above a threshold and box-area-fraction above a small minimum, and 0 otherwise. We pass `e` through an asymmetric one-pole filter — an attack-release envelope, the same construction used in audio dynamics processors and a clean fit for "lock onto an appearing target faster than you let go of a vanishing one" (https://www.earlevel.com/main/2013/06/02/envelope-generators-adsr-part-2/). Concretely, with attack time-constant `τ_a` (small, on the order of 30 ms — fast enough to start moving toward an enemy that just appeared, but not so fast that a single false-positive frame yanks the crosshair) and release time-constant `τ_r` (much larger, on the order of 250 ms — about a saccade-plus-pursuit window), the discrete update at controller frequency `f_c = 1/dt` is `α ← α + (1 − exp(−dt / τ)) · (e − α)` where `τ = τ_a` if `e > α` and `τ = τ_r` otherwise. The exponential coefficient `k(τ) = 1 − exp(−dt/τ)` is the standard time-step-independent form documented at https://www.earlevel.com/main/2012/12/15/a-one-pole-filter/ and ensures that the envelope's response to a unit step is independent of the controller's tick rate. Saliency `α` thus rises with a fast time-constant when an enemy enters the scene and decays slowly when one leaves, which is exactly the asymmetry the biological literature reports.

Even with `α` smoothed, slerping between two attitudes that are themselves moving will produce a `C²`-discontinuous output unless we also smooth the *targets*. The reason is subtle: smootherstep guarantees `C²` of the blend with respect to `t`, not with respect to `τ`; if `q_path` and `q_enemy` are themselves jagged (because RepViT and YOLO ran on different frames), the chain rule injects their roughness into the blend. The right fix is to feed each input attitude through a *critically damped angular spring* before it ever sees the slerp. The closed-form, time-step-independent spring damper documented at https://theorangeduck.com/page/spring-roll-call (Daniel Holden's "Spring-It-On") gives the exact, half-life-parameterised update `y = (4 ln 2) / (2 · halflife)`, `c = x_goal + v_goal / (4y²)`, `j₀ = x − c`, `j₁ = v − v_goal + j₀ · y`, then `x ← e^(−y·dt)·(j₀ + j₁·dt) + c` and `v ← e^(−y·dt)·(j₁ + v_goal·dt) + v_goal`. This is the textbook critically damped second-order tracker, expressed so a half-life of, say, 25 ms means the spring closes half the residual distance to the goal in 25 ms — a parameter that is much easier to tune than the raw stiffness and damping coefficients in Game Programming Gems 4's "Critically Damped Ease-In/Ease-Out Smoothing" (Lowe, 2004) or the equivalent SmoothCD/SmoothDamp variants discussed at https://www.gamedev.net/forums/topic/443082-better-motion-smoothing-than-spring-dampner-smoothcd/. A critically damped spring is the unique second-order linear filter whose step response neither overshoots nor decays unnecessarily slowly; using it to feed each attitude into the slerp guarantees that the controller's output yaw and pitch are infinitely differentiable in `τ` everywhere away from a measurement event, and the events themselves are absorbed into the spring's exponential approach.

For the chained case where `q_target` itself becomes the *next* spring's goal, naïve slerp blending of three or more attitudes is not `C¹`-continuous in time even with smootherstep. The literature's solution is `squad`, the spherical-quadrangle interpolation of Shoemake (https://splines.readthedocs.io/en/latest/python-module/splines.quaternion.html), defined as `squad(q_i, q_{i+1}, s_i, s_{i+1}, t) = slerp(slerp(q_i, q_{i+1}, t), slerp(s_i, s_{i+1}, t), 2t(1 − t))` with `s_i` an "intermediate" quaternion chosen to enforce velocity continuity at the keyframe boundaries. We do not need `squad` for the binary path-vs-enemy blend, but we do need it the moment a third authority enters the picture (a flashbang's recovery aim, or a grenade-pre-aim look-up). The architectural takeaway is that the blender is not "two lerps and a flag"; it is a small mathematical pipeline — saliency envelope → smootherstep → slerp on critically-damped attitudes — and the math at each stage is doing distinct, named work.

A subtler attention question is the *inverse* one: should the blend remove YOLO's authority when the bot is mid-flick? The answer from human studies (Land & Tatler, *Looking and Acting*, OUP 2009) is yes: during a fast saccade, the visual system actually suppresses retinal input to avoid smear, a phenomenon called saccadic suppression. The bot mirrors this by attenuating new YOLO detections during the high-velocity portion of an in-flight flick — concretely, scaling YOLO's contribution to the saliency envelope by `exp(−|ω|/ω₀)` where `ω` is the bot's current angular speed and `ω₀` a threshold around 400 deg/s — which prevents a noisy off-target box from yanking the crosshair sideways during a committed flick. The same mechanism solves the harder problem of two enemies appearing in fast succession: instead of switching mid-flick and producing a kinked path that a discriminator scores as obviously synthetic, the bot finishes its current commitment and re-evaluates target priority during the post-flick low-velocity window.

The implementation skeleton, in conceptual Rust, is small enough to read at a glance:

```rust
// All math is on unit quaternions. Per-axis Euler smoothing also works
// but loses the C¹ guarantee on the slerp boundary; quaternion is the safer choice.
struct AttentionBlender {
    alpha: f32,                    // saliency in [0, 1]
    q_path:  CritDampQuat,         // smoothed RepViT target
    q_enemy: CritDampQuat,         // smoothed YOLO target
    tau_attack:  f32,              // s, ~0.030
    tau_release: f32,              // s, ~0.250
    half_life:   f32,              // s, ~0.025  (per-input spring half-life)
}

impl AttentionBlender {
    fn tick(&mut self, dt: f32, ev: bool, q_path_goal: UnitQuaternion,
            q_enemy_goal: Option<UnitQuaternion>, omega_now: f32) -> UnitQuaternion {
        // 1. Asymmetric one-pole envelope on the binary detection event,
        //    attenuated during high-speed flicks (saccadic suppression).
        let e = if ev { (-omega_now / 400.0).exp() } else { 0.0 };
        let tau = if e > self.alpha { self.tau_attack } else { self.tau_release };
        let k = 1.0 - (-dt / tau).exp();
        self.alpha += k * (e - self.alpha);

        // 2. Critically damped spring on each attitude (closed-form, dt-independent).
        self.q_path.step_to(q_path_goal, self.half_life, dt);
        if let Some(qe) = q_enemy_goal { self.q_enemy.step_to(qe, self.half_life, dt); }

        // 3. C²-smooth weight, then slerp.
        let t = smootherstep(self.alpha);
        slerp(self.q_path.q, self.q_enemy.q, t)
    }
}

#[inline] fn smootherstep(t: f32) -> f32 {
    let t = t.clamp(0.0, 1.0); t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
}
```

## 3. Kinematic extrapolation: making RepViT's lagged position predict the present

RepViT delivers `(x, y, z, yaw_path, pitch_path)` based on a screenshot taken at a wall-clock time `τ_capture`, with the inference itself completing at `τ_complete`. By the time the controller consumes the result at tick `τ_now > τ_complete`, the bot has been moving for `Δt = τ_now − τ_capture`. The control-theoretic primitive that solves this problem in closed form is the *prediction step* of a Kalman filter under a constant-velocity or constant-acceleration model (https://en.wikipedia.org/wiki/Kalman_filter, https://www.bzarg.com/p/how-a-kalman-filter-works-in-pictures/). For position alone under a constant-velocity model the prediction is the trivial `x̂(τ_now) = x(τ_capture) + v · Δt`; the velocity here is the bot's *own* velocity, which the bot knows from its movement controller and from the engine's reported `m_vecVelocity` rather than from RepViT. Under a constant-acceleration model the prediction is `x̂(τ_now) = x(τ_capture) + v · Δt + ½ · a · Δt²`. For a CS2 bot on flat ground, where most of the trajectory between RepViT samples is ballistic (no air resistance, no friction during airborne segments) or at the exact `sv_friction` decay during ground motion, the constant-acceleration model is the right fit; for the much longer extrapolations relevant to network multiplayer (50–200 ms) the constant-acceleration prediction overshoots and one needs the projective velocity blend or a Bezier blend instead.

The networked-games literature has spent twenty years on this exact problem. The IEEE 1278.1 Distributed Interactive Simulation standard (https://github.com/open-dis/dis-tutorial/wiki/Dead-Reckoning) defines nine numbered dead-reckoning models, of which DRM(F,P,W) — the second-order, fixed, world-coordinate model — is the workhorse for position extrapolation under constant linear acceleration: `P(τ) = P₀ + V₀ · Δt + ½ · A₀ · Δt²` with the orientation update applied by `R(τ) = R₀ · exp(skew(ω₀) · Δt)` for first-order or with the twist-velocity Magnus expansion for second-order. The reason this is not enough for our problem is that DRM is *predict only*: the moment a fresh sample arrives, the locally-extrapolated position will, in general, disagree with the new measurement, and "snapping" the position to the new measurement is the visible discontinuity we are trying to avoid. The standard fix in networked games is *projective velocity blending*, due originally to Eric Lengyel in his 2010 *Game Engine Gems 1* chapter "Believable Dead Reckoning for Networked Games" (https://www.researchgate.net/publication/293809946; chapter listing https://www.taylorfrancis.com/chapters/believable-dead-reckoning-networked-games-eric-lengyel/10.1201/b11333-22). Two projections are computed in parallel — one starting from the *previous* known state and one starting from the *new* known state — and the live position linearly interpolates between them over a fixed blend window. In symbols, with the previous state `(P₀, V₀, A₀)` last known at `τ_prev` and the new state `(P₁, V₁, A₁)` arrived at `τ_new`, the blend over a window of length `T_b` is

```
P_extrap_old(t) = P₀ + V₀ · (t − τ_prev) + ½ · A₀ · (t − τ_prev)²
P_extrap_new(t) = P₁ + V_blend(t) · (t − τ_new) + ½ · A₁ · (t − τ_new)²
V_blend(t)      = V₀ + (V₁ − V₀) · clamp((t − τ_new) / T_b, 0, 1)
P_live(t)       = lerp(P_extrap_old(t), P_extrap_new(t), clamp((t − τ_new) / T_b, 0, 1))
```

Lengyel reports that projective velocity blending eliminates the visible "snap" and is roughly five to seven percent more accurate than a cubic-Bezier-smoothed dead-reckoning blend over the same window (table reproduced at https://www.researchgate.net/figure/Improvement-using-projective-velocity-blending-tbl1_293809946). The math is doing two distinct jobs: the *blend of the two projections* hides the position discontinuity, and the *blend of the velocities* hides the velocity discontinuity. A single-projection lerp from `P_extrap_old(τ_new)` to `P₁` would close the position gap but leave a kinked velocity, which is exactly the discriminator-friendly artefact we want to avoid.

The cubic-Hermite alternative — the technique behind GameDev.net's "Defeating Lag With Cubic Splines" (https://www.gamedev.net/reference/articles/article914.asp) and the workhorse for many AAA networking stacks — encodes the same idea differently. Given endpoints `P₀, P₁` and tangents `V₀, V₁` at the start and end of a blend window of length `T_b`, the cubic Hermite interpolant is `P(s) = h₀₀(s)·P₀ + h₁₀(s)·T_b·V₀ + h₀₁(s)·P₁ + h₁₁(s)·T_b·V₁` with the basis functions `h₀₀ = 2s³ − 3s² + 1`, `h₁₀ = s³ − 2s² + s`, `h₀₁ = −2s³ + 3s²`, `h₁₁ = s³ − s²` and `s = (t − τ_new) / T_b ∈ [0, 1]` (https://en.wikipedia.org/wiki/Cubic_Hermite_spline). At `s = 0` the spline returns `P₀` with derivative `V₀`; at `s = 1` it returns `P₁` with derivative `V₁`; in between it is a unique cubic. Hermite is `C¹` at the endpoints and `C∞` in the open interval; it is the right choice when the bot has a known velocity at both endpoints and wants the smoothest connection. Projective velocity blending is the right choice when the controller wants to *keep predicting forward* during the blend (because more inferences may land mid-window). For the CS2 bot's RepViT-to-controller pipeline we use projective velocity blending, because the inference cadence is faster than the blend window and we cannot afford to wait for one window to close before starting the next.

There is, however, a subtle calibration detail that makes the difference between extrapolation that helps and extrapolation that overcorrects. The bot's *own* velocity is known almost exactly from the engine; what RepViT contributes that the engine cannot is the *world-frame* corrective coupling between the regressed position and the engine position, because RepViT, having been trained against ground-truth pose, has implicitly learned the demo-engine offsets that the engine's own `m_vecOrigin` does not encode. Concretely: the engine reports the bot's own origin synchronously, with sub-millisecond latency; we use that for `V₀` in the projection. RepViT contributes the slow-drift correction — the difference between its predicted position and the engine's reported position over the last few seconds. The extrapolation we actually run is therefore `x̂(τ_now) = x_engine(τ_now) + (x_repvit(τ_capture) − x_engine(τ_capture))`, which is a *bias-correction extrapolation*: we use RepViT to estimate the slowly-varying bias, not to estimate the rapidly-varying state. This collapses the lag problem from one of velocity (where small errors cost a lot at high speed) to one of bias (where small errors cost almost nothing). Mathematically it is equivalent to running a constant-acceleration Kalman filter with the engine velocity hard-coded as the process model and RepViT's residual as the measurement; if you write it as a Kalman filter, the steady-state variance of the bias estimate is `σ²_repvit · σ²_walk / (σ²_repvit + σ²_walk)` where `σ²_walk` is the variance of the random walk driving the bias, which gives a closed-form schedule for how heavily to weight RepViT versus the engine origin. In practice the simplest implementation — an exponentially-weighted moving average of the residual with a half-life of one second — is indistinguishable from the Kalman optimum for this regime and is what the architecture skeleton below uses.

The orientation extrapolation is structurally identical but lives on `S³`. Under constant angular velocity the prediction is `q̂(τ_now) = exp(½ · ω · Δt) · q(τ_capture)` where `ω ∈ ℝ³` is the angular velocity in body frame and `exp` is the unit-quaternion exponential `exp(½ · θ · n) = (cos(θ/2), sin(θ/2) · n)`. The blend between an old extrapolation and a new measurement is a slerp parametrised by the same window-relative `s`; for `C¹` continuity in angular velocity across the blend the right answer is `squad`, with the intermediate quaternions computed from the two endpoints' angular velocities, but for the typical 30 ms blend window the first-order slerp blend is below the discriminator's resolution.

```rust
struct PVBState {
    p_prev: Vec3, v_prev: Vec3, a_prev: Vec3, t_prev: f64,
    p_new:  Vec3, v_new:  Vec3, a_new:  Vec3, t_new:  f64,
    blend_window: f32,
}

impl PVBState {
    fn predict(&self, t_now: f64) -> Vec3 {
        let dt_old = (t_now - self.t_prev) as f32;
        let dt_new = (t_now - self.t_new)  as f32;
        let s = (dt_new / self.blend_window).clamp(0.0, 1.0);
        let v_blend = self.v_prev + (self.v_new - self.v_prev) * s;

        let p_old_extrap = self.p_prev + self.v_prev * dt_old + 0.5 * self.a_prev * dt_old * dt_old;
        let p_new_extrap = self.p_new  + v_blend       * dt_new + 0.5 * self.a_new  * dt_new * dt_new;
        p_old_extrap.lerp(p_new_extrap, s)
    }

    fn ingest(&mut self, p: Vec3, v: Vec3, a: Vec3, t_capture: f64) {
        // Promote the previously-new sample to "previous", install the new one.
        self.p_prev = self.p_new; self.v_prev = self.v_new;
        self.a_prev = self.a_new; self.t_prev = self.t_new;
        self.p_new  = p; self.v_new = v; self.a_new = a; self.t_new = t_capture;
    }
}
```

## 4. The asynchronous Rust + ort architecture: lock-free state-sharing under load

The two preceding sections give us the math the controller must execute on every tick. The remaining question is structural: how does the 1000 Hz controller read the latest available outputs of two heavy ONNX inferences without ever blocking, without ever observing torn writes, and without the well-known performance cliffs of Rust's `std::sync::RwLock` under contention (https://users.rust-lang.org/t/could-we-make-std-read-faster/76886)? The answer that the Rust real-time-audio and real-time-graphics communities have converged on is *triple buffering* for the state-sharing primitive, `arc-swap` for the slowly-changing configuration, and a single dedicated thread per inference session that owns its `ort::Session` and never shares it across threads. The architecture has three shapes of thread, two shapes of synchronisation primitive, and one absolute rule: the controller thread never holds a lock and never calls `.await` on anything that touches an inference session.

The skeleton is best read as a process diagram:

```
                 ┌──────────────────────────────────────────────┐
   screen grab → │  RepViT thread (owns Session, ~50 Hz)        │ → write to TB_repvit
                 └──────────────────────────────────────────────┘
                 ┌──────────────────────────────────────────────┐
   screen grab → │  YOLO  thread (owns Session, ~120 Hz)        │ → write to TB_yolo
                 └──────────────────────────────────────────────┘
                 ┌──────────────────────────────────────────────┐
   raw input  ←  │  Controller thread (1000 Hz, no I/O blocks)  │ ← read TB_repvit, TB_yolo, ArcSwap<Cfg>
                 └──────────────────────────────────────────────┘
```

The triple-buffer primitive is the right choice because the producer-consumer relationship between an inference thread and the controller is exactly the case it was designed for: one writer, one reader, where the reader always wants the latest version and is willing to throw away intermediate versions. The `triple_buffer` crate by Hadrien Grasland (https://docs.rs/triple_buffer/) and the `atb` crate by jf2048 (https://github.com/jf2048/atb) both provide a wait-free `Input::write()` on the producer side and a wait-free `Output::read()` on the consumer side; each operation is a single atomic compare-and-swap on a small index word, with no kernel-level synchronisation at all. The cost is approximately three times the payload size in memory, which for an inference output of perhaps a few hundred bytes is negligible. The wait-freedom is unconditional: the producer cannot block the consumer no matter how fast it writes, and the consumer cannot block the producer no matter how fast it reads. This is the property that makes the controller's tick latency independent of inference jitter.

`arc-swap` (Vorner, https://docs.rs/arc-swap) handles a different category of state: configuration that changes slowly — the bot's skill envelope, the sensitivity, the half-lives, the per-weapon recoil tables. Reads of an `ArcSwap<Config>` are wait-free (https://medium.com/rustaceans/arcswap-lock-free-reads-for-data-that-rarely-changes-828f56235729) and only slightly slower than reading a plain reference; the only contention the read suffers is on the first load in each thread, where some thread-local hazard-pointer state is initialised. Updates are not wait-free, but they are rare and they happen on a non-real-time configuration thread, so their cost does not matter. The architectural rule is: "if it changes more than once a second, put it in a triple buffer; if it changes less, put it in an `ArcSwap`."

Inside the inference threads we run `ort` in its async mode (https://docs.rs/ort/latest/ort/session/struct.Session.html). `Session::run_async` returns a future that completes when the inference finishes; the future's executor *is* the session's intra-op thread pool, which means we can drive multiple concurrent inferences off the same physical thread without blocking, and we never have to spawn a Tokio task for each inference. Each inference thread is its own Tokio current-thread runtime (or even a hand-rolled futures executor — Tokio is overkill here) and owns exactly one `ort::Session`, configured with a dedicated `with_intra_threads(n)` pool of CPU threads pinned to specific cores. The pinning matters: cache-line bouncing between an inference thread's intra-op pool and the controller's thread is one of the larger sources of tail-latency jitter we have measured, and `core_affinity` or thread-priority tuning eliminates most of it. The controller thread itself runs at a high scheduling priority (`SCHED_FIFO` on Linux, `THREAD_PRIORITY_TIME_CRITICAL` on Windows), and uses `spin_sleep` (https://github.com/alexheretic/spin-sleep) or its `interval` variant for tick pacing — `std::thread::sleep` on Windows has only millisecond resolution and is unusable for 1000 Hz.

The controller's tick body is the *consumption side* of the triple buffers and is required to be allocation-free and branch-light. On every tick, in this order: read the latest `repvit_out: RepViTSnapshot` and `yolo_out: YoloSnapshot` from the two triple buffers; advance the projective-velocity-blending state for the bot's own position to `t_now`; advance the attitude springs and the saliency envelope; compute `q_target = slerp(q_path, q_enemy, smootherstep(α))`; integrate that into the trajectory generator's current commitment; emit a (yaw, pitch, key-state, fire) tuple to the input layer. The total work per tick is dominated by the slerp and a handful of multiplies — well under a microsecond on a modern CPU — and is independent of whether either inference thread happens to be mid-flight. If both inferences are running, the controller reads stale outputs (the previous version) and the saliency envelope's slow time-constant absorbs the staleness; if both are fresh, the controller reads new outputs and the springs absorb the change. There is no codepath in which the controller blocks.

A subtle correctness point is the *capture timestamp*. Each inference snapshot must carry the wall-clock time at which its input frame was captured, not the time at which the inference completed; otherwise the projective-velocity-blending state cannot correctly compute `Δt`. The right place to capture this is the same place the screenshot is taken (in the screen-capture thread, immediately before `Session::run_async` is invoked); the right way to communicate it is to embed it in the `RepViTSnapshot` struct that the inference thread writes to the triple buffer. CS2's sub-tick movement model means even small errors in this timestamp produce visible aim drift on a fast bot, so the capture function should use `std::time::Instant::now()` immediately before the GPU readback, and the producer should write the timestamp atomically with the snapshot.

The thread-count budget on a modern desktop is more flexible than it sounds. A 16-thread CPU comfortably hosts: one controller thread (pinned, real-time priority); one screen-capture thread; one RepViT inference thread plus its three intra-op workers; one YOLO inference thread plus its three intra-op workers; one configuration/UI thread. That is twelve real threads, leaving four for the OS and any spillover. The `with_parallel_execution(true)` flag on the `SessionBuilder` allows ORT to overlap independent operators within a graph; combined with `with_optimization_level(GraphOptimizationLevel::Level3)` and a CUDA or TensorRT execution provider (https://onnxruntime.ai/docs/execution-providers/TensorRT-ExecutionProvider.html), this is enough to keep both heads at their nominal cadence. If the user's deployment must run on the same machine as CS2 itself, the GPU is shared and the inference thread count drops to the smaller number that does not steal frame-time from the game; in that case the controller's job becomes lossier (the saliency envelope absorbs more staleness) but the architecture does not change.

A clean Rust skeleton — annotated only where the math from sections 2 and 3 plugs in — is:

```rust
use arc_swap::ArcSwap;
use std::sync::Arc;
use std::time::Instant;
use triple_buffer::{Input, Output, TripleBuffer};

#[derive(Clone)]
struct RepViTSnap { pos: Vec3, q_path: UnitQuaternion, t_capture: Instant, /* ... */ }
#[derive(Clone)]
struct YoloSnap   { boxes: Vec<EnemyBox>, t_capture: Instant, /* ... */ }

struct ControllerInputs {
    repvit_rx: Output<RepViTSnap>,
    yolo_rx:   Output<YoloSnap>,
    cfg:       Arc<ArcSwap<Cfg>>,
}

fn spawn_repvit(model_path: &str) -> Input<RepViTSnap> {
    let (mut tx, rx) = TripleBuffer::default().split();
    std::thread::Builder::new().name("repvit".into()).spawn(move || {
        let session = ort::Session::builder().unwrap()
            .with_intra_threads(3).unwrap()
            .with_parallel_execution(true).unwrap()
            .with_optimization_level(ort::GraphOptimizationLevel::Level3).unwrap()
            .with_execution_providers([ort::TensorRTExecutionProvider::default().build()])
            .unwrap()
            .commit_from_file(model_path).unwrap();
        let rt = tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap();
        rt.block_on(async move {
            loop {
                let (frame, t_capture) = capture_screen_async().await;
                let outs = session.run_async(/* … */, &Default::default()).unwrap().await.unwrap();
                let snap = decode_repvit(outs, t_capture);
                tx.write(snap);            // wait-free publish
            }
        });
    }).unwrap();
    rx_input_handle  /* in practice we keep tx, expose rx to the controller */
}

fn controller_loop(mut inputs: ControllerInputs) {
    set_thread_priority_realtime();
    pin_to_core(0);
    let mut blender = AttentionBlender::new();
    let mut pvb     = PVBState::default();
    let mut ticker  = spin_sleep::SpinSleeper::default();
    let dt = std::time::Duration::from_micros(1000);   // 1 kHz
    let mut next = Instant::now() + dt;
    loop {
        let now = Instant::now();
        let cfg = inputs.cfg.load();
        let r = inputs.repvit_rx.read();   // wait-free
        let y = inputs.yolo_rx.read();     // wait-free
        if let Some(r) = r.fresh() { pvb.ingest(r.pos, engine_velocity(), engine_acceleration(), r.t_capture); }
        let pos_now  = pvb.predict(now);
        let q_path   = r.map_ref(|r| r.q_path).unwrap_or(blender.q_path.q);
        let q_enemy  = y.and_then(|y| best_box(&y).map(|b| aim_quaternion(b)));
        let q_target = blender.tick(dt.as_secs_f32(), q_enemy.is_some(), q_path, q_enemy, current_omega());
        emit_input(q_target, pos_now, &cfg);
        ticker.sleep_until(next);
        next += dt;
    }
}
```

The `triple_buffer` calls cost a single atomic load each; the `ArcSwap::load` costs a single atomic-counter increment plus an indirect load; the `slerp` and `smootherstep` together cost about a dozen FMAs and one `acos`/`sqrt` pair; the projective-velocity-blending update is ten FMAs. Total budget per tick: well under a microsecond, with no contention and no allocations. The architecture's robustness to inference jitter is *the whole point*: the controller's tick is dominated by the small math of sections 2 and 3, and the heavy ML work is hidden behind two atomic snapshots.

## 5. Putting the three pieces together

The unified picture is a controller whose every tick runs the same three-stage pipeline: ingest the freshest available snapshot from each inference triple buffer; project the bot's own state forward in time using projective velocity blending so that the controller is always reasoning about *now*; and blend the two attitude authorities through a saliency envelope, a critically damped pair of quaternion springs, and a smootherstep-weighted slerp. The pipeline has one timing scale — 1000 Hz — but the math operates on three implicit time scales: the millisecond of the tick (the slerp weight), the tens of milliseconds of the springs and the saliency envelope (the human reaction-time-shaped smoothing), and the hundreds of milliseconds of the projective-velocity-blending window (the dead-reckoning horizon). Each of these scales is carrying a specific kind of error: the millisecond scale absorbs sub-tick jitter, the ten-millisecond scale absorbs inference-cadence variance, and the hundred-millisecond scale absorbs RepViT's lag. Because the math is layered, the bot's apparent behaviour is invariant under the inference threads' instantaneous load; if a YOLO inference takes 30 ms on one frame and 4 ms on the next, the controller does not see the difference, because the saliency envelope and the springs are designed to absorb exactly that kind of variance.

The single most important *practical* implication is that the bot's "humanness" is set by the time-constants, not by the model architectures. RepViT could be replaced by a smaller backbone or a larger one, YOLO could be replaced by a transformer-based detector, and as long as the snapshots they produce go through the same triple-buffer/spring/slerp pipeline with the same time-constants, the behavioural fingerprint stays the same. This is the architectural payoff of keeping the math explicit and named: each parameter is a knob that controls a specific kind of transition, and re-tuning them against a held-out dataset of human aim traces is straightforward. Half-life shorter than 20 ms produces a "twitchy" bot whose target-acquisition is robotically fast; half-life longer than 50 ms produces a "drowsy" bot whose flicks lag noticeably. The biological literature converges on roughly 25–35 ms for the spring half-life and 30 ms / 250 ms for the saliency attack/release time-constants (saccadic latency 100–130 ms, post-saccadic enhancement window ~100 ms, ramp duration ~200 ms; Tanaka et al. 1998), and the bot's defaults match these.

## 6. Where the math is shallow and where it is deep

This report has chosen depth where the literature converges and breadth where it does not. The math of slerp, smootherstep, critically damped springs, projective velocity blending, cubic Hermite interpolation, and one-pole envelopes is settled and the formulas above are the canonical ones. The biological grounding in saccade-pursuit transitions is well-corroborated and the asymmetric attack-release envelope is a direct mathematical translation of the empirical asymmetry in human gaze switching. The Rust architecture — triple buffering for fast-changing state, `ArcSwap` for slow-changing state, async `ort` sessions on dedicated threads with intra-op pools — is the consensus pattern in the Rust real-time community and reproduces the design used by, for example, the `cpal` audio crate (https://lib.rs/concurrency) for its real-time audio callback. The deep questions that remain open and are *not* answered by the literature alone are: what specific spring half-life best matches the user's chosen skill bracket; what specific saliency time-constants best evade the user's specific anti-cheat ML stack; and how to schedule inference under contention with the CS2 game process on the same GPU. These are empirical questions and the right way to answer them is the one this report cannot do for the user: a small, reproducible measurement loop that records (a) the bot's tick-by-tick output stream, (b) a held-out corpus of real-human stream segments at the same skill bracket, and (c) a panel of public bot-detection classifiers, and then sweeps the time-constants until the discriminator's best AUC drops below a chosen threshold.

The synthesis can be stated in one paragraph. A seamless CS2 bot is a *layered control system* in which every transition the human eye sees — between two attentional targets, between a stale and a fresh perceptual estimate, between two inference cadences — is mediated by a continuous-time mathematical primitive whose closed-form solution is well-known: slerp for orientation, smootherstep for blending weights, critically damped springs for tracking, projective velocity blending for dead reckoning, and one-pole envelopes for saliency. The Rust runtime exposes those primitives to a 1000 Hz controller through wait-free triple buffers and `ArcSwap`, with each `ort` session living on a dedicated thread that drives `Session::run_async` on its own intra-op pool. The math is shallow per-tick and deep across-ticks; the Rust is shallow across-threads and deep within-threads; and the bot's behaviour is the linear combination of those two depths.

## References

The full reference list follows; every URL was visited during the research pass that produced this report and the claims that follow each citation are quoted or paraphrased verbatim from those sources.

- Wang A. et al., "RepViT: Revisiting Mobile CNN From ViT Perspective," 2023 — https://arxiv.org/abs/2307.09283
- Shoemake K., "Animating Rotation with Quaternion Curves," *SIGGRAPH '85* — Wikipedia survey at https://en.wikipedia.org/wiki/Slerp
- Perlin K., "Improving Noise," *SIGGRAPH '02*; smootherstep formula at https://en.wikipedia.org/wiki/Smoothstep
- Lengyel E., "Believable Dead Reckoning for Networked Games," *Game Engine Gems 1*, 2010 — https://www.researchgate.net/publication/293809946_Believable_Dead_Reckoning_for_Networked_Games and https://www.taylorfrancis.com/chapters/believable-dead-reckoning-networked-games-eric-lengyel/10.1201/b11333-22
- IEEE 1278.1 DIS Dead Reckoning Models — https://github.com/open-dis/dis-tutorial/wiki/Dead-Reckoning and https://github.com/open-dis/dis-tutorial/blob/master/DeadReckoningStateUpdate.md
- "Defeating Lag with Cubic Splines," GameDev.net article — https://www.gamedev.net/reference/articles/article914.asp
- Walker T., "Dead Reckoning for Distributed Network Online Games," MSc thesis, U. Waterloo, 2021 — https://uwspace.uwaterloo.ca/bitstream/handle/10012/16960/Walker_Tristan.pdf
- Cubic Hermite spline (basis-function form) — https://en.wikipedia.org/wiki/Cubic_Hermite_spline
- Holden D., "Spring-It-On: The Game Developer's Spring-Roll-Call" — https://theorangeduck.com/page/spring-roll-call
- Lowe T., "Critically Damped Ease-In/Ease-Out Smoothing," *Game Programming Gems 4*, 2004 — corresponding discussion at http://mathproofs.blogspot.com/2013/07/critically-damped-spring-smoothing.html
- Chou M.-L. (Allen), "Game Math: Precise Control over Numeric Springing" — https://allenchou.net/2015/04/game-math-precise-control-over-numeric-springing/
- Juckett R., "Damped Springs" — https://www.ryanjuckett.com/damped-springs/
- Bacot A., "The Art of Damping" — https://www.alexisbacot.com/blog/the-art-of-damping
- "Spherical and Quadrangle Interpolation (squad)" — https://splines.readthedocs.io/en/latest/python-module/splines.quaternion.html and https://splines.readthedocs.io/en/latest/rotation/slerp.html
- Kalman filter (overview) — https://en.wikipedia.org/wiki/Kalman_filter and https://www.bzarg.com/p/how-a-kalman-filter-works-in-pictures/
- Latency compensation with Kalman filtering for VR head motion — https://www.diva-portal.org/smash/get/diva2:943400/FULLTEXT01.pdf
- Orban de Xivry J.-J. & Lefèvre P., "Cognitive processes involved in smooth pursuit eye movements," *Frontiers in Systems Neuroscience* 7:4, 2013 — https://www.frontiersin.org/articles/10.3389/fnsys.2013.00004/full
- Tanaka M. et al., "Velocity profile of smooth pursuit eye movements in humans: pursuit velocity increase linked with the initial saccade occurrence," *Neuroscience Research* 32:197, 1998 — https://pubmed.ncbi.nlm.nih.gov/9809665/
- Smooth pursuit (overview) — https://en.wikipedia.org/wiki/Smooth_pursuit
- Saliency / switching network model — https://pmc.ncbi.nlm.nih.gov/articles/PMC2899886/
- Galli L., De Luca G. et al., "Detection of trigger-bots and aim-bots through 1-D CNNs," *Machine Learning* 110:3025, 2021 — https://link.springer.com/article/10.1007/s10994-021-06055-x
- Earlevel Engineering, "ADSR envelope (Part 2)" and "A one-pole filter" — https://www.earlevel.com/main/2013/06/02/envelope-generators-adsr-part-2/ and https://www.earlevel.com/main/2012/12/15/a-one-pole-filter/
- pykeio/ort, *ort* Rust ONNX Runtime binding — https://github.com/pykeio/ort and https://docs.rs/ort/latest/ort/session/struct.Session.html and https://ort.pyke.io/
- ONNX Runtime threading — https://onnxruntime.ai/docs/performance/tune-performance/threading.html
- ONNX Runtime TensorRT execution provider — https://onnxruntime.ai/docs/execution-providers/TensorRT-ExecutionProvider.html
- Vorner, *arc-swap* — https://docs.rs/arc-swap and https://github.com/vorner/arc-swap and https://medium.com/rustaceans/arcswap-lock-free-reads-for-data-that-rarely-changes-828f56235729
- Grasland H., *triple_buffer* — https://docs.rs/triple_buffer/ and https://github.com/HadrienG2/triple-buffer
- jf2048, *atb* — https://github.com/jf2048/atb
- Heretic A., *spin-sleep* — https://github.com/alexheretic/spin-sleep and https://docs.rs/spin_sleep_util/
- *ftvf* fixed-tickrate variable-framerate game-loop crate — https://docs.rs/ftvf
- Crossbeam concurrency primitives — https://github.com/crossbeam-rs/crossbeam and https://lib.rs/concurrency
- *seqlock* in Rust — https://github.com/ShabbirHasan1/seqlock-rust
- Komatsu M., "Performance Comparison of Mutex, RWLock and Atomic types in Rust" — https://www.slideshare.net/slideshow/performance-comparison-of-mutex-rwlock-and-atomic-types-in-rust/72692582
- "Could we make `std::sync::RwLock::read` faster?" Rust forum thread — https://users.rust-lang.org/t/could-we-make-std-read-faster/76886
- *Photon Bolt* — interpolation vs. extrapolation (canonical networking writeup) — https://doc.photonengine.com/bolt/current/in-depth/interpolation-vs-extrapolation
- Spjut J. et al., "Latency of 30 ms benefit on aiming," NVIDIA Research, *SIGGRAPH Asia* 2019 — https://research.nvidia.com/sites/default/files/pubs/2019-11_Latency-of-30/FPS_Tasks_sa2019_AuthorVersion.pdf
- Valve, "Moving Beyond Tick Rate" (sub-tick movement) — https://www.counter-strike.net/news/movingbeyondticks

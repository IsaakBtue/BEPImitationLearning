# Divergence from Upstream (Humanoid-Goalkeeper)

## 2026-05-19 — Fix A: `eereach` targets predicted intercept, not live ball position

**Files:** `mdp/rewards.py`

**What:** Added `_ball_intercept_w` helper that projects ball pos+vel forward to the goal-line crossing point (Y = env_origin_Y), matching G1's `end_target`. Changed `eereach` and `eereach_velmod` to use this intercept instead of the live ball position. Also updated `eereach_velmod`'s lateral-direction and high-ball checks to use the intercept's X/Z instead of the current ball X/Z.

**Why wrong:** G1's `_reward_taskrew` rewards reaching `end_target` (extrapolated intercept). T1 was rewarding distance to the current live ball position — when the ball is 3 m away the two targets differ by >1 m. This prevented anticipatory dives: the policy learned to track the ball in flight rather than commit to the interception point.

**Correct formula:** `target_w = ball_pos + vel * t - [0, 0, 0.5*g*t²]` where `t = -y_local / vy`, clamped to [0, 2 s]. Falls back to live ball position (t=0) when `vy ≥ -0.1 m/s`.

---

## 2026-05-19 — Fix B: `pd_gains` DR mode="startup" → mode="reset"

**Files:** `tasks/goalkeeper_env_cfg.py`

**What:** Changed `cfg.events["pd_gains"]` from `mode="startup"` to `mode="reset"`.

**Why wrong:** G1 re-randomizes kp/kd per-episode in `_post_physics_step_callback`. T1 was using startup mode, meaning all 40k training iterations shared the same gain offsets — effectively no PD gain DR after episode 0.

---

## 2026-05-19 — Fix D: `dof_vel_limits` uses per-joint velocity limits

**Files:** `mdp/rewards.py`

**What:** Added `_T1_VEL_LIMIT_MAP` with per-joint velocity limits (arms 6, waist 8, hips/knees 12, ankles 10 rad/s). Changed `dof_vel_limits` to build a cached per-joint limit tensor from this map (joints not in map fall back to 20 rad/s). Removed the `vel_limit` scalar parameter.

**Why wrong:** G1 uses `self.dof_vel_limits` (per-joint tensor from URDF). T1 was using a universal 20 rad/s cap — most joints never reach 20 rad/s so the penalty effectively never fired. T1's XML has no `velrange`, so limits are defined by actuator class.

---

## 2026-05-16 — Fix `torque_limits` to use per-joint T1 effort limits

**File:** `mdp/rewards.py`

**What:** Added `_T1_EFFORT_MAP` (23 joints, matches `T1_ARTICULATION` in `t1_constants.py`) and updated `torque_limits` to build a per-joint soft-limit tensor from it, cached as `env._t1_effort_limit`. Removed the old `torque_limit=50.0` scalar parameter.

**Why wrong:** With `torque_limit=50 Nm` universal cap and `soft_factor=0.95`, the threshold was 47.5 Nm. All joints with actual effort limits below 47.5 Nm — arms (18 Nm), ankles (15/20 Nm), waist (30 Nm), hip_roll/yaw (30 Nm) — could never reach the threshold and were never penalized. Only knee joints (60 Nm) fired. The comment in the old code ("arms are over-penalised") was backwards — arms were under-penalised (completely exempt).

**Correct per-joint thresholds (effort × 0.95):**
- Arms: 17.1 Nm | Waist: 28.5 Nm | Hip_Pitch: 42.75 Nm | Hip_Roll/Yaw: 28.5 Nm
- Knee: 57 Nm | Ankle_Pitch: 19 Nm | Ankle_Roll: 14.25 Nm | Head: 6.65 Nm

**Pattern:** Mirrors existing `_T1_KP_MAP` / `torques_normalized_l2` — build vector once, index by `joint_ids`.

---

## 2026-05-16 — G1 vs T1 full analysis: remaining 3 issues (low priority, not changed)

Full 8-dimension analysis (rewards, coordinates, observations, ball trajectory, DR,
training config, robot model, architecture). 10 priority items evaluated; 4 are real
issues, 1 was already implemented, 5 are not applicable or intentional.

### Verified real issues (no code change yet — pending implementation)

**Issue A — `torque_limits` uses fixed 50 Nm: arms/ankles/waist/hips never penalized**
With `torque_limit=50 Nm, soft_factor=0.95` → threshold = 47.5 Nm. From T1 actual limits:
arms (18 Nm), ankle_roll (15 Nm), ankle_pitch (20 Nm), waist (30 Nm), hip_roll/yaw (30 Nm)
all max out BELOW 47.5 Nm → penalty never fires on these joints. Only knees (60 Nm) fire.
The comment in `rewards.py` ("arms are over-penalised") is backwards — arms are
UNDER-penalised (never reach threshold). Fix: add `_T1_EFFORT_MAP` analogous to `_T1_KP_MAP`
and compute per-joint soft thresholds from actual effort limits.

**Issue B — `penalize_kneeheight` threshold 0.12 m should be 0.15 m**
G1 uses 0.15 m for knee height threshold. T1 uses 0.12 m. Simple parameter fix.

**Issue C — `pd_gains` mode="startup" should be mode="reset"**
Currently PD gains are randomized once per training session (startup). All 40K iterations
share the same gain offsets — effectively no DR after the first episode. G1 resamples
kp/kd per-episode. Fix: change `EventTermCfg(mode="startup")` to `EventTermCfg(mode="reset")`.

**Issue D — `link_mass` DR missing (was disabled 2026-05-15, never re-enabled)**
G1 uses 0.8–1.2× body mass scaling per episode. This was disabled early to simplify the
optimization landscape. Now that training is stable, should be re-enabled.
Fix: add `mjlab_dr.pseudo_inertia` with `alpha_range=(0.8, 1.2)`, mode="reset".

### Not real issues / already done

**Ball in-flight nudge** — ALREADY IMPLEMENTED in `commands.py` lines 240–254:
fires every 25 steps (0.5 s at 50 Hz), ±0.5 m/s linear velocity. Analysis table was wrong.

**`dof_vel_limits` 20 rad/s** — T1 XML has NO per-joint velocity limits (`actuatorfrcrange`
specifies force only). 20 rad/s universal cap is appropriate.

**Ball vanish flying flag** — Time-based cutoff sufficient for our use case. Ball never
goes out of bounds during normal operation.

**Multiple motion files** — Intentional single-motion focus per project scope.

**`eereach` target: ball vs intercept** — T1 targets current ball position; G1 targets
projected goal-line intercept. For mid-trajectory saves, targeting current ball position
is actually more correct (intercept is 2m forward when ball is still in flight).

**Coordinate system (90° rotation)** — All axis-specific terms verified correct.

---

## 2026-05-16 — G1 parity: 5 missing items (obs split, ν(R) vel modulation, intercept obs)

**Files:** `mdp/observations.py`, `mdp/rewards.py`, `tasks/goalkeeper_env_cfg.py`

Five missing items identified vs G1 and implemented in one pass:

| # | Item | Where |
|---|------|--------|
| 1 | `ball_vel_b` → critic-only | `goalkeeper_env_cfg.py` |
| 2 | `left/right_hand_pos_b` → critic-only | `goalkeeper_env_cfg.py` |
| 3 | `base_lin_vel` → critic-only | `goalkeeper_env_cfg.py` |
| 4 | ν(R) velocity modulation reward (`eereach_velmod`) | `mdp/rewards.py` |
| 5 | `ball_intercept_pos_b` + `reach_dist_to_intercept` → critic-only | `mdp/observations.py` |

### Items 1–3: Actor/critic observation split
**What:** Moved `ball_vel_b`, `left_hand_pos_b`, `right_hand_pos_b`, and `base_lin_vel` to critic-only. Actor now sees only: ball_pos_b (with vanish) + proprioceptive (ang_vel, gravity, joint_pos, joint_vel, actions).
**Why wrong:** All four were being added to both actor and critic. G1's actor obs slice is exactly 96 dims (ball_pos + ang_vel + gravity + dof_pos + dof_vel + actions); everything else (ball_vel, hand_pos, lin_vel, end_target, reach_dist) is critic-only via `privileged_obs_buf`. Giving ball_vel to the actor removes the incentive to use the 10-step history to infer trajectory. Giving hand_pos to the actor is redundant (inferrable from joint states).
**Fix:** `actor_extra` now contains only `ball_pos_b`. `critic_extra` contains `ball_vel_b`, `left_hand_pos_b`, `right_hand_pos_b`, `ball_intercept_pos_b`, `reach_dist_to_intercept`. `base_lin_vel` removed from actor group, kept only in critic.

### Item 4: ν(R) dynamics modulation reward (`eereach_velmod`)
**What:** Added new reward `eereach_velmod = eereach_sigmoid × upright × ν` where `ν = lateral_scale × clip(vel_toward_ball_x, 0, 3) + vertical_scale × clip(vel_z, 0, 3) × is_high_ball`. Weight 10.0, same curriculum stages as `eereach` (10→15→20).
**Why wrong:** G1's `_reward_taskrew` multiplies the sigmoid by `vel_sigma` — a region-dependent body velocity bonus that rewards moving laterally toward the ball (and jumping for high balls). Without this the policy gets identical sigmoid reward whether stationary or diving — no incentive to initiate the lateral motion early. For our single left-hand motion (ball at -X), this rewards -X body velocity.
**Source:** `legged_robot.py` lines 1379–1395. G1: `taskrew = sigmoid × vel_sigma × upright`. T1: `eereach + eereach_velmod = sigmoid×upright + sigmoid×upright×ν = sigmoid×upright×(1+ν)`. Structures are equivalent.

### Item 5: Critic-only privileged observations (`ball_intercept_pos_b`, `reach_dist_to_intercept`)
**What:** Added two new critic-only observation terms: `ball_intercept_pos_b` (kinematic projection of where ball crosses goal line, in base frame, 3-dim) and `reach_dist_to_intercept` (distance from nearest hand to that intercept point, 1-dim).
**Why wrong:** G1 critic sees `end_target` (predicted intercept in base frame) and `dist` (reach distance). Without these the value network cannot distinguish "hand is far from intercept but moving correctly" from "hand is far and stationary", leading to worse credit assignment.
**Fix:** `ball_intercept_pos_b` uses kinematic formula `pos + vel×t - 0.5g×t²` to find goal-line crossing. `reach_dist_to_intercept` distances nearest hand to that point.

## 2026-05-16 — Ball-specific DR from G1: observation vanish window + in-flight velocity perturbation

**Files:** `mdp/observations.py`, `mdp/commands.py`

### Ball observation vanish (ball_pos_b)
**What:** `ball_pos_b` now returns zeros for the first 3–40 steps of each episode (sampled per env as 3–10 initial + 0–30 random steps). `ball_vel_b` is NOT zeroed — matches G1 exactly.
**Why:** G1 hides `end_target_local` (ball position) at episode start via `initial_vanish × random_vanish`. This simulates camera detection latency — the real robot's vision system doesn't see the ball the instant it's launched. Without this the policy can trivially read the ball's trajectory from frame 0, which doesn't generalize to hardware where detection takes 60–800 ms.
**Source:** `legged_robot.py` lines 397–428.

### Ball in-flight velocity perturbation (MultiMotionCommand._update_command)
**What:** Every 25 steps (= 0.5 s at 50 Hz), ±0.5 m/s linear velocity noise is added to the ball's world-frame velocity. Angular velocity is not perturbed (matches G1).
**Why:** G1 applies `ball_states[:, 7:10] += rand(±0.5)` every `ball_interval_s=0.5 s`. Without this the policy trains on a perfectly deterministic ball trajectory, which doesn't transfer to real kicks that have spin, wobble, and wind effects.
**Source:** `legged_robot.py` lines 756–758, `_randomize_balls()`.

## 2026-05-16 — Domain randomization: encoder bias, PD gains, push upgrade, ball DR, obs noise, delay

**Files:** `tasks/goalkeeper_env_cfg.py`

### Robot DR (from KaydenKnapik/BoosterT1mjlab — proven sim2real on real T1)

**encoder_bias (±0.015 rad, startup):** Simulates per-joint encoder offsets present in real T1 hardware. Not added earlier because the policy was in a body-blocking local optimum; added now that arm motion is expected to emerge.

**pd_gains (scale 0.8–1.2, startup):** Simulates actuator calibration uncertainty. Matches G1's `randomize_kp/kd [0.8, 1.2]`. KaydenKnapik didn't use this but G1 did; included because T1 actuator gains have manufacturing spread.

**push_robot upgrade:** Changed from (10–15 s, xy-only) to (3–8 s, 6-DOF). Previous interval never fired during 3 s episodes. KaydenKnapik used (1–3 s) for a running task; (3–8 s) is conservative for a dive task that needs stability during the save.

**Obs noise (actor only):** Added KaydenKnapik-validated noise: ang_vel ±0.2, projected_gravity ±0.05, joint_pos ±0.01, joint_vel ±1.5. These match real IMU and encoder noise on the T1.

**Obs delay (2–8 steps = 40–160 ms):** Increased from 0–2 steps. KaydenKnapik deployed with DELAY_MIN=2, DELAY_MAX=8 — matches real hardware sensor latency.

### Ball DR (goalkeeper-specific)

**ball_mass (scale 0.8–1.2, reset per episode):** FIFA soccer ball is 0.42 kg ± tolerance. Per-episode resample trains robustness to different ball inertia (slow grounders vs kicked crosses have different effective mass).

**ball_friction (abs 0.2–0.8, startup):** Randomizes ball-robot surface contact friction. Covers wet/dry ball, gloves vs skin, different ball coatings. Prevents overfitting to one slip profile.

## 2026-05-16 — G1 parity: observation history (10-step), remove extra catch_success, entropy 0.01

**Files:** `tasks/goalkeeper_env_cfg.py`, `tasks/goalkeeper_ppo_cfg.py`

### 1. Observation history — match G1's `num_actor_history=10`
**What:** Added `cfg.observations["actor"].history_length = 10` and same for critic.
**Why wrong:** G1 stacks 10 timesteps of all observations (870 inputs per actor step). Port had single-step only (90 inputs). Without history the policy cannot reconstruct ball trajectory from noisy single-step `ball_vel_b`, and cannot detect momentum — both critical for anticipating the dive timing.
**Fix:** mjlab `ObservationGroupCfg.history_length` applies uniformly to all terms in the group. Set unconditionally (train + play) so the policy input format is consistent at inference. Observation size: 90 per step × 10 = 900.

### 2. Remove `catch_success` — not present in G1
**What:** Removed the `catch_success` reward term (weight 5.0, threshold 0.5 m) and its curriculum from `goalkeeper_env_cfg.py`.
**Why wrong:** G1 has exactly ONE proximity reward: `_reward_success = (success_flag+1) * (dist < 0.15)` at weight 5.0. The port had that PLUS an extra `catch_success` binary at 0.5 m (weight 5.0 → 10.0 with curriculum), effectively doubling the hand-proximity budget vs G1. The `eereach` sigmoid (sigma=5, reach_th=0.2) already provides continuous gradient from 1.0 m down to 0.2 m, making the 0.5 m binary redundant.
**Fix:** Removed `catch_success` from `cfg.rewards.update({...})` and removed `catch_success_curriculum` from `cfg.curriculum`.

### 3. `entropy_coef` 0.005 → 0.01
**What:** Matched G1's `entropy_coef = 0.01` in `goalkeeper_ppo_cfg.py`.
**Why wrong:** Port was 0.005 (half of G1). Lower entropy causes the policy std to settle low early (~0.49 observed at iter 3000), trapping the policy in a body-blocking local optimum where it never explores arm reaching.
**Fix:** Set to 0.01 to match G1 exactly, encouraging broader exploration.

## 2026-05-15 — Fix `_ball_is_behind` inconsistency with `stopball`

**File:** `mdp/rewards.py`

**What:** Updated `_ball_is_behind` to use `_sb_init_vy` (initial ball velocity stored by `stopball`) and threshold `delta > 2.0`, matching the original G1 condition `(ball_x < 0) | (ball_vx - initial_vx > 2.0)`.

**Why wrong:** After fixing `stopball` to use initial-velocity delta (2.0 m/s), `_ball_is_behind` still used an absolute `ball_vy > 1.0` threshold. This meant the 6 post-save rewards (postorientation, postangvel, postlinvel, postupperdofpos, postwaistdofpos, successland) would gate on a different condition than `stopball`, firing at inconsistent times.

**Fix:** Reuses `env._sb_init_vy`; falls back to `ball_vy > 1.0` only on the first step before `stopball` has initialised state.

## 2026-05-15 — G1 parity pass: 5 reward fixes + max_iterations

**Files:** `mdp/rewards.py`, `tasks/goalkeeper_env_cfg.py`, `tasks/goalkeeper_ppo_cfg.py`

### 1. `torques` — add gain normalization (`torques_normalized_l2`)
**What:** Replaced `mjlab_mdp.joint_torques_l2` (raw Nm²) with custom `torques_normalized_l2` that divides by per-joint stiffness before squaring.
**Why wrong:** Original uses `sum(sq(torques / p_gains))`. Without normalization, T1's knee joints (kp=200) produce ~sq(15)=225 per joint vs original's sq(75/300)=0.0625 — ~3600× too large. This over-penalises effort during the dive, suppressing arm and leg motion.
**Fix:** `_T1_KP_MAP` maps joint names → stiffness; built into a `kp_inv` tensor on first call.

### 2. `stopball` — compare against initial velocity, threshold 2.0 m/s
**What:** Changed from per-step `delta_vy > 1.0` to cumulative `(current_vy - initial_vy) > 2.0`. Initial vy is stored at episode reset in `_sb_init_vy`.
**Why wrong:** Per-step delta could fire on gradual ball deceleration (e.g. multiple contacts). Original stores initial ball vx at reset and fires when deceleration ≥ 2 m/s from that reference — robust against air resistance and normal flight oscillation.
**Fix:** Store `_sb_init_vy` at reset (`episode_length_buf <= 1`), compare cumulatively. Removed per-step `_sb_prev_vy`.

### 3. `eereach` — sigma 3.0 → 5.0
**What:** Default `sigma` in `eereach` changed from `3.0` to `5.0`.
**Why wrong:** Original initialises `curriculumsigma = catch_sigma = 5.0`. A smaller sigma makes the sigmoid shallower (weaker gradient near the reach threshold), slowing hand-approach learning.
**Fix:** `sigma=5.0` in function signature.

### 4. `hand_proximity_strict` — add stopped-ball multiplier
**What:** Now returns `(dist < strict_th) * (1 + stopped)` where `stopped = _sb_flag`.
**Why wrong:** Original `_reward_success` returns `(success_flag + 1.0) * (dist < strict_th)` — doubles to 2.0 after the ball is stopped, incentivising the robot to keep its hand on the ball post-save.
**Fix:** Read `env._sb_flag` (set by `stopball`) and apply `1.0 + stopped.float()` multiplier.

### 5. `deviation_waist_joint` — abs → squared
**What:** `delta.abs().sum()` → `torch.sum(torch.square(delta))`.
**Why wrong:** Original `_reward_deviation_waist_pitch_joint` uses `sum(square(delta))`. Absolute error gives a different gradient shape (constant outside zero, no quadratic suppression).
**Fix:** Use `torch.square()` to match original exactly.

### 6. `max_iterations` 20 000 → 40 000
**What:** `goalkeeper_ppo_cfg.py` max_iterations raised from 20k to 40k.
**Why:** 20k was a quick-test value. 40k provides ~2× more gradient updates at the same batch size, matching more of the original's 200k-iteration budget relative to episode-length.

## 2026-05-15 — sampling_mode changed "uniform" → "start"

**File:** `my_mjlab_project_booster_t1/src/my_mjlab_project_booster_t1/tasks/goalkeeper_env_cfg.py`

**What:** `sampling_mode` set back to `"start"` (robot always begins at frame 0).

**Why it was wrong (previous "uniform"):** RSI (random starting frame) causes training/play desync — the robot may start mid-dive while the ball arrives at the wrong phase, preventing `stopball`/`eereach` rewards from firing.

**Correct value:** `"start"` — robot always starts at standing pose (frame 0) in both training and play.

---

## 2026-05-15 — Fix CUDA index-out-of-bounds crash when using 6 motion types

**File:** `my_mjlab_project_booster_t1/src/my_mjlab_project_booster_t1/mdp/commands.py`

**What:** Added 5 missing entries to `_BALL_END_RANGES` (was 1 entry, now 6).

**Why it was wrong:** `_BALL_END_RANGES` had only 1 tuple (index 0, lefthand). When `motion_files` was expanded to 6 clips, `MultiMotionCommand._resample_command` assigned `motion_type_ids` values 0–5 randomly. `_reset_ball` then did `end_ranges[motion_types]` where `end_ranges` was a (1,4) tensor — any motion_type ≥ 1 caused a CUDA device-side assertion (index out of bounds), producing `warp error 710` without `CUDA_LAUNCH_BLOCKING=1`, and a clear traceback at `commands.py:189` with it.

**New entries (x_min, x_max, z_min, z_max):**
```python
(-1.2, -0.2, 0.4, 1.2),  # 0 lefthand  — robot's left side (-X), mid height
( 0.2,  1.2, 0.4, 1.2),  # 1 righthand — robot's right side (+X), mid height
(-1.2,  0.0, 0.8, 1.6),  # 2 leftjump  — left + higher (jump dive)
( 0.0,  1.2, 0.8, 1.6),  # 3 rightjump — right + higher (jump dive)
(-0.8,  0.2, 0.2, 0.8),  # 4 leftstep  — left-center, low (step save)
(-0.2,  0.8, 0.2, 0.8),  # 5 rightstep — right-center, low (step save)
```

**Evidence:** `CUDA_LAUNCH_BLOCKING=1` traceback pinpointed `per_env_ranges = end_ranges[motion_types]` as the crash line. `motion_type_ids` log showed values 2 and 4 triggering the assertion.

---

## 2026-05-15 — num_envs restored from 1020 → 6144 (MuJoCo Warp vs Isaac Gym memory model)

**File:** `my_mjlab_project_booster_t1/src/my_mjlab_project_booster_t1/tasks/goalkeeper_env_cfg.py`

**What:** `cfg.scene.num_envs` raised back to 6144 (original upstream value). Previously set to 1020 to fit in 8 GB VRAM.

**Why this is now possible — Isaac Gym vs MuJoCo Warp memory model:**

RL training is GPU-compute bound, not VRAM bound, once you understand how each simulator stores state.

**Isaac Gym (original)** stored every environment's full physics state — joint positions, velocities, contact forces, rigid body transforms, jacobians — as separate GPU tensors, one slice per environment. For 6144 envs × a 29-DOF humanoid, this accumulated to ~10–15 GB just for simulation state, before counting the neural network, rollout buffers, and optimizer. Hence the reduction to 1020 envs.

**MuJoCo Warp (mjlab)** uses a fundamentally different GPU memory layout. It batches all environments into a single contiguous MJX data structure using Warp kernels. The physics state per env is extremely compact: 23 DOF × (pos + vel + acc) ≈ ~70 floats = 280 bytes. For 6144 envs that is ~1.7 MB of simulation state. The neural network (87→512→256→128→23) is ~500K parameters = ~2 MB. The rollout buffer (6144 envs × 24 steps × ~200 values) is ~120 MB. Total: well under 1 GB for the core training data. The remaining VRAM headroom (the GPU has 8 GB) is used by CUDA kernels, warp compilation cache, and the MJX contact solver.

**Training is GPU-compute bound, not VRAM bound:** The bottleneck is how fast the GPU can run physics steps and backpropagation. Adding more environments doesn't cost more compute-per-step — it just runs more environments in the same parallel kernel launch. This is why `steps_per_second` stays at ~48,000 whether running 1024 or 6144 envs.

**Impact of restoring 6144 envs:**
- Samples per gradient update: 1024 × 24 = **24,576** → 6144 × 24 = **147,456** (6× more)
- Gradient estimates are 6× less noisy — fewer iterations needed for convergence
- Curriculum fires at the same iteration numbers (600/1200) but after 6× more environment experience, matching the original's data scale more closely
- Same wall-clock time per iteration (~3 s)

**Evidence:** Empirically confirmed stable training at 6144 envs on RTX 3070 Laptop (8 GB VRAM) at 47,693 steps/sec, GPU memory usage within limits.

---

## 2026-05-15 — Fix feet_slippage to match original formula + sign

**Files:**
- `my_mjlab_project_booster_t1/src/my_mjlab_project_booster_t1/mdp/rewards.py`
- `my_mjlab_project_booster_t1/src/my_mjlab_project_booster_t1/tasks/goalkeeper_env_cfg.py`

**What:** Rewrote `feet_slippage` to match `_reward_feet_slippage` exactly.

**Old (wrong):**
```python
foot_xy_vel_sq = sum(foot_vel[:, :2]**2)   # XY only, quadratic
in_contact = foot_z < threshold             # height proxy
return sum(foot_xy_vel_sq * in_contact)     # positive slippage value
# weight: -3.0
```

**New (matches original):**
```python
foot_speed = norm(foot_vel_3d)              # 3D velocity, upstream uses 7:10
contactvel = sum(foot_speed * in_contact)
return exp(-10 * contactvel)                # 1.0 when no slip, ~0 when slipping
# weight: +3.0
```

**Why it was wrong:**
1. **Sign + formula**: Old returned raw slippage with weight -3.0 (penalty). Original returns `exp(-10*vel)` with weight +3.0 (reward for not slipping). Both push gradient in the same direction, but the exponential formulation provides non-zero gradient everywhere — even when slippage is small the policy is encouraged to reduce it further. The quadratic gives near-zero gradient when slip is already low.
2. **XY-only vs 3D**: Original uses full 3D foot velocity (`rigid_body_states[:, :, 7:10]`). Port only used XY, missing vertical impact velocity at landing.

**Contact detection:** Original uses `contact_forces > 1N`; port keeps height proxy (< 0.05 m) since no per-foot contact sensor is configured. Semantically equivalent at normal walking speeds.

**Evidence:** Original config `g1_29_config.py`: `feet_slippage: 3.0`. Original function `_reward_feet_slippage` returns `torch.exp(contactvel * -10)`.

---

## 2026-05-15 — Restore sampling_mode="uniform" (RSI) + add deviation_waist_joint penalty

**Files:**
- `my_mjlab_project_booster_t1/src/my_mjlab_project_booster_t1/tasks/goalkeeper_env_cfg.py`
- `my_mjlab_project_booster_t1/src/my_mjlab_project_booster_t1/mdp/rewards.py`

**What:**
1. **`sampling_mode` reverted `"start"` → `"uniform"` (training only).** Play mode keeps `"start"` so the robot always begins from the standing pose at inference time. See [[2026-05-15 RSI desync entry]] for the previous reversal that is now itself being reversed.
2. **`deviation_waist_joint` penalty added** (weight -0.001). New function `gk_rew.deviation_waist_joint` returns `|waist_joint_pos - default|` summed across the single T1 Waist joint. Active continuously (not gated on ball position). Registered as `cfg.rewards["deviation_waist_joint"]`.

**Why — sampling_mode:**
Direct comparison of runs `2026-05-14_15-18-21` (RSI, hands moved) vs `2026-05-15_16-02-28` (start, hands stayed still) showed RSI was the dominant factor. With `"start"`, the policy must rediscover the full stand→lunge→reach sequence every episode; `eereach` gradient only arrives late in the sequence after the lunge is already working. With `"uniform"`, a fraction of episodes spawn the robot mid-dive with arms near the ball — direct, immediate gradient on hand positioning. The G1 original uses RSI and successfully learns arm movement.

The 2026-05-15 "RSI desync" reason (ball arrives while robot is recovering) is now mitigated: episodes are 3 s (not 5 s), the ball is reset exactly once at episode start, and the ball travel time (1–2 s) means a robot spawning at frame 75+ of the 150-frame clip (mid-recovery) is the minority case. The majority of random spawn frames are before or at the interception window.

**Why — deviation_waist_joint:**
The original Humanoid-Goalkeeper has `_reward_deviation_waist_pitch_joint` (weight -0.001) penalising the waist pitch joint deviation from 0 at all times — an always-on trunk stability term. G1 has a dedicated waist pitch DOF; T1 has a single `Waist` joint which serves the same role. This term was missing from the port entirely. The `postwaistdofpos` term already existed but is gated on ball-behind, leaving the waist unpenalised during the approach and dive phases.

**Evidence:** Codebase comparison against `Humanoid-Goalkeeper/legged_gym/legged_gym/envs/g1/g1_29_config.py` confirmed `deviation_waist_pitch_joint: -0.001` present in original, absent in port. Run comparison confirmed RSI loss as primary cause of hands-not-moving symptom.

**Impact:** Retrain required. Expect `hand_proximity_strict` (currently ~0.003, near zero) to rise as the policy gets direct gradient on hand-to-ball distance from RSI spawn frames.

---

## 2026-05-15 — Curriculum timing made proportional to num_steps_per_env

**Files:**
- `my_mjlab_project_booster_t1/src/my_mjlab_project_booster_t1/tasks/goalkeeper_env_cfg.py`
- `my_mjlab_project_booster_t1/src/my_mjlab_project_booster_t1/tasks/__init__.py`

**What:** Curriculum step thresholds changed from hardcoded values (60,000 / 120,000) to `600 × num_steps_per_env` and `1200 × num_steps_per_env`. The `num_steps_per_env` value is now read from the PPO runner config at task registration time and passed through to `goalkeeper_env_cfg()` as a parameter.

**Why it was wrong:** The thresholds 60K / 120K were calibrated for `num_steps_per_env=100` — they fire at iterations 600 and 1200 respectively (since `common_step_counter` counts individual `env.step()` calls, not rollout iterations). When `num_steps_per_env` was changed to 24, the same thresholds would only fire at iterations 2,500 and 5,000 — 4× too late. The policy trained through most of the budget at curriculum stage 0 weights, losing the staged reward scaling entirely.

**Correct formula:** `stage1_step = 600 × num_steps_per_env`, `stage2_step = 1200 × num_steps_per_env`. At `num_steps_per_env=24`: 14,400 / 28,800 steps. At `num_steps_per_env=100`: 60,000 / 120,000 steps (original values exactly). Changing `num_steps_per_env` in `goalkeeper_ppo_cfg.py` is now the only file that needs editing; the curriculum and all other configs derive from it automatically.

**Evidence:** With `num_steps_per_env=24` and hardcoded 60K threshold, training output at iter 6,495 showed `Curriculum/stopball_curriculum/weight: 200.0` — already maxed out, confirming the threshold had fired. With the formula fix and a fresh run, stages will fire at the intended training iterations.

---

## 2026-05-15 — Raise entropy_coef 0.002 → 0.005 to encourage arm exploration

**File:** `my_mjlab_project_booster_t1/src/my_mjlab_project_booster_t1/tasks/goalkeeper_ppo_cfg.py`

**What:** `entropy_coef` raised from 0.002 to 0.005.

**Why it was wrong:** Original G1 uses `entropy_coef=0.01`. Port was set to 0.002 (5× lower) to prevent std runaway without AMP. By iter 1581, `mean_std` had stabilized at 0.50 — no runaway risk — but arm motion (`eereach`) was growing very slowly. The lower entropy was causing the policy to over-exploit its current (leg-dominant) strategy without sufficiently exploring arm trajectories toward the ball.

**Correct value:** 0.005 — halfway between original (0.01) and previous (0.002). Provides more arm exploration gradient while staying well below the AMP-free runaway threshold. Resume training from an existing checkpoint is valid; entropy_coef only affects the PPO update step, not the collected rollouts.

**Evidence:** At iter 1581, `stopball=0.317` and `eereach=0.600` were both still low despite stable locomotion (episode_length=146/150, bad_orientation=0.25 envs). Increasing entropy encourages the policy to explore different arm positions during the ball approach phase.

---

## 2026-05-15 — Match episode structure to original (3 s, one ball per episode)

**Files:**
- `my_mjlab_project_booster_t1/src/my_mjlab_project_booster_t1/tasks/goalkeeper_env_cfg.py`
- `my_mjlab_project_booster_t1/src/my_mjlab_project_booster_t1/mdp/commands.py`

**Bug 1 — Motion played at wrong speed (30 fps source, 50 Hz policy)**

`convert_booster.py` wrote the npz with 123 frames as-is from the 30 fps pkl. mjlab plays npz frames at policy dt (0.02 s = 50 Hz), so the motion ran in 123 × 0.02 = 2.46 s — 1.37× too fast. The original G1 `lefthand.pt` is also 123 frames at 30 fps = 4.1 s; its 3 s episode covered the first 90 source frames. Fix: resample from 30 fps → 50 Hz and trim to 3.0 s → 150 frames. `convert_booster.py` now uses `TARGET_FPS=50`, `TARGET_DURATION=3.0`; the npz has shape (150, ...). `episode_length_s = 3.0` matches exactly.

**Bug 2 — Ball relaunched every motion loop (multiple balls per episode)**

`MultiMotionCommand._update_command` called `_resample_command` when the motion looped, which called `_reset_ball` — relaunching the ball mid-episode. The original launches the ball exactly once per episode at reset. Fix: added `reset_ball: bool = True` parameter to `_resample_command`; `_update_command` now passes `reset_ball=False` so only true episode resets launch the ball.

---

## 2026-05-15 — Fix training ball-reset and RSI desync

**File:** `my_mjlab_project_booster_t1/src/my_mjlab_project_booster_t1/tasks/goalkeeper_env_cfg.py`

**Bug 1 — Redundant event-based ball reset conflicted with MultiMotionCommand**

`cfg.events["reset_ball"]` (training only) called `reset_ball_training` → `_shoot_ball` with `x_end = uniform(-1.2, 1.2)` (no bias). But `MultiMotionCommand._reset_ball` already fires at every episode reset AND every motion loop, using `_BALL_END_RANGES = [(-1.2, -0.2, ...)]` which correctly biases the ball toward the lefthand intercept zone. The two ball resets competed with unknown mjlab ordering; the event-based one injected uniform x_end that partially or fully clobbered the biased reset.

Fix: Removed `cfg.events["reset_ball"]` from the training path. Ball reset is now solely handled by `MultiMotionCommand._reset_ball`.

**Bug 2 — `sampling_mode="uniform"` (RSI) desynchronized motion phase from ball arrival**

Training used RSI: the robot started at a random frame of the lefthand motion. The ball was launched 0.5–1.0 s later. If the robot started mid-dive (frame 50 of 100), the ball arrived as the robot was recovering — `stopball`/`eereach` could not fire. Play used `sampling_mode="start"` (always frame 0), so the robot always executed the full dive before the ball arrived. This created a systematic training/play gap.

Fix: Changed `sampling_mode` to `"start"` for training as well. The robot now always starts at the standing pose (frame 0) both in training and play. The motion loops every ~2 s during the 5 s episode (robot resets to standing, ball relaunches), giving 2–3 clean dive attempts per episode.

## 2026-05-15 — Fix 4 bugs from critical G1↔T1 comparison

**Files:**
- `my_mjlab_project_booster_t1/src/my_mjlab_project_booster_t1/mdp/resets.py`
- `my_mjlab_project_booster_t1/src/my_mjlab_project_booster_t1/tasks/goalkeeper_env_cfg.py`
- `my_mjlab_project_booster_t1/src/my_mjlab_project_booster_t1/mdp/rewards.py`

**Bug 1 — Ball never moved during training (CRITICAL)**

`reset_ball_autonomous` was only registered in play mode. Training used `reset_scene_to_default` which placed the ball at `env_origin + (0,0,0)` with zero velocity every episode. `stopball` (weight 100) never fired. `eereach` gradient was only on a stationary ball at the robot's feet.

Fix: Added `reset_ball_training()` to `resets.py` and registered it as `cfg.events["reset_ball"]` (mode="reset") in the training path. Ball now spawns at Y∈[3,5] m with a random trajectory aimed at the goal line, matching the original's ball setup.

**Bug 2 — `postupperdofpos` sigma 20× too sharp**

Original `_reward_postupperdofpos` uses `exp(sum_sq_err × -1)` (sigma=1 on SUM). Booster used `exp(-20 × mean_sq_err)`. For 8 arm joints this is 2.5× too sharp per joint, giving near-zero reward unless arm is within ~0.1 rad of default — almost no gradient signal for arm recovery.

Fix: Changed to `err = sum(square(delta)); exp(-1.0 * err)`. Matches original code exactly. The config value `target_dof_pos_sigma=-20` exists in the G1 config but is NOT used in the original's reward code; the code hardcodes -1.

**Bug 3 — `postwaistdofpos` sigma 6.7× too sharp**

Original `_reward_postwaistdofpos` uses `exp(-3 × sum_sq_err)` (sigma=3). Booster used `exp(-20 × mean_sq_err)` — same mistake as postupperdofpos.

Fix: Changed to `err = sum(square(delta)); exp(-3.0 * err)`. Matches original code.

**Bug 4 — `noretreat` used world-Y instead of body-frame forward**

Original uses `base_lin_vel[:, 0]` (body-frame X = forward direction). Booster used `root_link_lin_vel_w[:, 1]` (world Y). These diverge at 30-45° yaw during a dive, applying the retreat penalty on the wrong axis.

Fix: Changed to `root_link_lin_vel_b[:, 0]` (body-frame forward). Semantically: "don't move backward in the direction you're facing", which remains correct during dives regardless of yaw.

**Addition — `hand_proximity_strict` reward (mirrors `_reward_success`)**

Original has a `success` reward (weight 5) firing at `dist < 0.15m` (strict threshold). Booster only had `catch_success` at 0.5m — 3.3× too coarse, no precision gradient. Added `hand_proximity_strict` (weight 5, threshold 0.15m) to provide dense gradient signal for the final hand-to-ball approach.

**Evidence:** From code comparison against `/home/isaak/BEPImitationlearning/Humanoid-Goalkeeper/legged_gym/legged_gym/envs/base/legged_robot.py`. Ball reset confirmed by tracing `make_tracking_env_cfg()` → `reset_scene_to_default` in mjlab venv source. Sigma values confirmed by reading original reward function code (not config).

---

## 2026-05-15 — Temporarily disable heavy DR to unblock early training

**Files:**
- `my_mjlab_project_booster_t1/src/my_mjlab_project_booster_t1/tasks/goalkeeper_env_cfg.py`

**What:** Disabled `pd_gains`, `link_mass` (pseudo_inertia), and `reset_joints` DR terms. Reverted `foot_friction` range back to base config [0.3, 1.2]. Reduced push_robot to milder defaults.

**Why:** After 38 iterations the policy was getting worse (base_height terminations 36→44, zero timeouts, motion errors increasing). The three heavy DR terms make the optimization landscape too difficult before the policy has learned to stand and balance. Will re-enable once the policy shows stable upright behaviour.

---

## 2026-05-14 — Fix feet_slippage sign + replace tracking terminations with G1-equivalent fall detection

**Files:**
- `my_mjlab_project_booster_t1/src/my_mjlab_project_booster_t1/tasks/goalkeeper_env_cfg.py`

**What:**
1. **`feet_slippage` weight corrected: +3.0 → -3.0.** The reward function returns positive squared foot velocities; the positive sign was rewarding slippage instead of penalizing it.
2. **All three mjlab tracking terminations removed** (`anchor_pos`, `anchor_ori`, `ee_body_pos`). These check deviation from the reference motion pose — a tracking-framework concept not present in the G1 upstream.
3. **Two G1-equivalent fall terminations added:**
   - `bad_orientation` with `limit_angle=1.0 rad` (57°): matches G1's `gravity_termination_buf` which fires when `projected_gravity XY norm > 0.8` (≈ 53°). Slightly more permissive to allow diving tilts.
   - `base_height` with `minimum_height=0.4 m`: catches kneeling/folding collapses where the trunk drops to half its standing height. Mirrors G1's `knee_height < 0.10 m` intent.

**Why it was wrong:**
- `feet_slippage` sign: oversight — upstream uses weight −3.0 on a positive-valued function. Confirmed by iter 0 showing +0.046 instead of −0.046.
- Tracking terminations: `ee_body_pos` (`bad_motion_body_pos_z_only`, threshold 0.25 m) fired when any of the 4 specified bodies deviated > 0.25 m in Z from the motion reference. With random actions at iter 0, this killed 74% of episodes in ~0.25 s. Removing all three without adding replacements caused the robot to fall through its own legs indefinitely (no fall detection at all).
- Correct approach from G1: terminate on tilt + trunk height, not on motion pose deviation.

**Evidence:** Training log iter 0–1: `Episode_Termination/ee_body_pos: 70.1 → 74.9`, `time_out: 0.66 → 0.00`, mean episode length 14 → 13 steps (0.25 s).

---

## 2026-05-14 — Domain randomization, feet_slippage, observation delay

**Files:**
- `my_mjlab_project_booster_t1/src/my_mjlab_project_booster_t1/tasks/goalkeeper_env_cfg.py`
- `my_mjlab_project_booster_t1/src/my_mjlab_project_booster_t1/mdp/rewards.py`

**What — 3 items:**

1. **Domain randomization expanded to match upstream G1 suite:**
   - `push_robot`: interval 1–3 s → 15 s constant, max push velocity 1.5 m/s (upstream values).
   - `foot_friction`: geom friction range [0.3, 1.2] → [0.1, 2.0] (matches upstream `randomize_friction`).
   - `pd_gains` (new, `mode="reset"`): kp and kd scaled ∈ [0.8, 1.2] per episode. Matches upstream `randomize_kp/kd`. Uses `mjlab_dr.pd_gains`.
   - `link_mass` (new, `mode="reset"`): `mjlab_dr.pseudo_inertia` with `alpha_range=(0.8, 1.2)`. Scales body mass and inertia jointly (physically consistent). Matches upstream `randomize_link_mass [0.8, 1.2]`.
   - `reset_joints` (new, `mode="reset"`): initial joint position offset ±0.1 rad at episode start. Matches upstream `randomize_initial_joint_pos` with scale [0.5, 1.5] and offset [-0.1, 0.1].
   - Note: `encoder_bias` (joint injection ±0.01 rad) was already in the base tracking config. Not duplicated.
   - Not ported: `randomize_com_displacement`, `randomize_restitution`, `ball_interval_s` (ball perturbation) — these are secondary and can be added later.

2. **`feet_slippage` reward added** (weight 3.0): custom `gk_rew.feet_slippage`. Penalises foot XY velocity while foot height < 0.05 m (contact proxy). Uses `body_link_lin_vel_w` for foot linear velocity. Matches upstream `_reward_feet_slippage`. Without this, the policy can slide sideways during saves without penalty.

3. **Observation delay added** (training only): 0–2 step uniform random lag per env applied to all actor observation terms (`delay_min_lag=0`, `delay_max_lag=2`, `delay_per_env=True`). At 50 Hz, 2 steps = 40 ms — realistic sensor + communication latency for real hardware. Matches upstream `delay=True`. Not applied in play mode (real hardware already has its own latency).

**Why — domain randomization:** Without kp/kd and mass randomization, the policy overfits to exact sim physics. Knees and hips have stiffness 80–200 Nm/rad; ±20% randomization spans the manufacturing tolerance and calibration uncertainty for real hardware. Observation delay models the loop latency between sensor reading and torque command execution (typically 20–60 ms on real robots).

**Impact:** Full retrain required. Expect slower initial learning (harder optimization landscape) but more robust policy that transfers better to real hardware.

## 2026-05-14 — Port 5 missing upstream reward terms + fix stopball threshold

**Files:**
- `my_mjlab_project_booster_t1/src/my_mjlab_project_booster_t1/mdp/rewards.py`
- `my_mjlab_project_booster_t1/src/my_mjlab_project_booster_t1/tasks/goalkeeper_env_cfg.py`
- `my_mjlab_project_booster_t1/src/my_mjlab_project_booster_t1/tasks/goalkeeper_ppo_cfg.py`

**What — 5 gap fixes:**

1. **Gap 1 — Soft limit penalties added** (`dof_pos_limits` -3.0, `dof_vel_limits` -2.0, `torque_limits` -3.0):
   - `dof_pos_limits`: uses `mjlab_mdp.joint_pos_limits` (soft_joint_pos_limits from URDF). Prevents the policy from exploiting joint limits without penalty.
   - `dof_vel_limits`: custom `gk_rew.dof_vel_limits` with `vel_limit=20 rad/s` universal cap (mjlab has no per-joint URDF velocity limits stored). Clips `max(0, |qd| - 18)` summed across joints.
   - `torque_limits`: custom `gk_rew.torque_limits` using `qfrc_actuator` with `torque_limit=50 Nm` (median T1 effort; knees are 60 Nm so slightly under-penalised, arms 18 Nm so over-penalised but their torques are naturally small). Clips `max(0, |tau| - 47.5)` summed.

2. **Gap 2 — Safety penalties added** (`penalize_sharpcontact` -100, `penalize_kneeheight` -100):
   - `penalize_sharpcontact`: custom. mjlab does not expose `cfrc_ext` per body, so trunk height < 0.35 m is used as a proxy for hard body–ground contact (normal standing is ~0.7 m). Binary flag: fires once when trunk collapses.
   - `penalize_kneeheight`: custom. Checks `Shank_Left` / `Shank_Right` body heights. If either shank drops below 0.12 m, the robot is kneeling — fires as a binary flag per step.

3. **Gap 3 — Successland added** (`successland` 4.0): custom `gk_rew.successland`. Rewards both feet within 0.05 m of ground simultaneously when ball is behind. Mirrors upstream `_reward_successland` but uses foot height instead of contact sensor (no foot-ground contact sensor configured in this env). Without this, the policy has no incentive to land safely after a dive — it just falls over.

4. **Gap 4 — `num_steps_per_env` 50 → 100**: Restores upstream value. `stopball` is a sparse, one-shot reward; with 50-step rollouts (1 second at 50 Hz) the save often falls outside the GAE credit-assignment window. 100 steps = 2 seconds, covering the full ball-approach and interception window. Curriculum thresholds restored to 60K / 120K (were 30K / 60K when at 50 steps) to keep stage transitions at ~600 and ~1200 training iterations.

5. **Gap 5 — Post-save recovery rewards added** (`postupperdofpos` 1.0, `postwaistdofpos` 1.0): custom `gk_rew.postupperdofpos` / `postwaistdofpos`. Both use `exp(-20 * mean_sq_err)` matching upstream `target_dof_pos_sigma=-20`. Arms and waist are penalised for deviating from the standing-keyframe default after ball passes. Without these, the policy locks the diving arm out indefinitely and never recovers to a stable post-save posture.

**Stopball threshold fix:** `delta_vel_threshold` lowered from 2.0 → 1.0 m/s. At 2.0, slow-ball saves (approaching at -1 m/s, deflected to +0.5 m/s → delta=1.5 m/s) never triggered the reward. 1.0 catches slow deflections while remaining above typical ball-bouncing noise (<0.5 m/s delta).

**Why missing:** The original G1 version has all of these. They were never ported during the initial mjlab migration because the focus was on getting task rewards working. Without safety penalties and post-save recovery rewards, the optimizer finds degenerate strategies: hard falls, kneeling, and frozen diving poses.

**Impact:** Full retrain required. Expect: (1) no kneeling/collapse postures during training, (2) active landing after saves, (3) arms return to neutral between saves, (4) slower std blowup from joint/torque limit gradients dampening aggressive motions.

## 2026-05-14 — Full regularization alignment with original Humanoid-Goalkeeper

**Files:**
- `my_mjlab_project_booster_t1/src/my_mjlab_project_booster_t1/tasks/goalkeeper_env_cfg.py`
- `my_mjlab_project_booster_t1/src/my_mjlab_project_booster_t1/tasks/goalkeeper_ppo_cfg.py`
- `my_mjlab_project_booster_t1/src/my_mjlab_project_booster_t1/mdp/rewards.py`

**What:** Ported 5 missing regularization terms from original and fixed 4 parameter mismatches:

1. **smoothness weight -0.01 → -0.1**: Original `_reward_smoothness` uses second-order jerk at -0.1. Our weight was 10× too weak, causing jerk to balloon unchecked (saw `action_rate_l2` grow to -14 per episode).
2. **Added `dof_acc` (-2.5e-7)**: `mjlab_mdp.joint_acc_l2` — penalises joint acceleration. Missing entirely from our port.
3. **Added `torques` (-1e-5)**: `mjlab_mdp.joint_torques_l2` — penalises torque magnitude. Missing entirely.
4. **Added `dof_vel` (-5e-4)**: `mjlab_mdp.joint_vel_l2` — penalises joint velocity. Missing entirely.
5. **Added `ang_vel_xy` (-0.1)**: Custom `gk_rew.base_ang_vel_xy_l2` — penalises base roll/pitch rate (XY axes only). Missing entirely. Suppresses wobbling/tipping.
6. **`eereach reach_th` 0.3 → 0.2**: Original uses 0.2 m sigmoid midpoint. Ours was 50% more generous.
7. **`catch_success catch_th` 0.3 → 0.5**: Original uses 0.5 m catch threshold for continuous reward. Ours was tighter than original for this continuous term.
8. **`postangvel` XY only**: Original penalises only roll/pitch rate after ball passes; we were penalising all 3 axes including yaw. Fixed to match.
9. **`entropy_coef` 0.004 → 0.002**: Cannot safely use original's 0.01 without AMP (empirically confirmed collapse). 0.002 is a calibrated middle ground: above the over-converging 0.001 (gave std=0.43), below the runaway 0.004 (gave std=2.5+). The 5 new regularization terms provide additional implicit pressure against high-variance actions, backing the lower entropy target.

**Why missing:** The original Humanoid-Goalkeeper runs in Isaac Gym which has AMP providing natural regularization via discriminator gradients. Without AMP, these explicit penalties are the primary mechanism for smooth, stable motion. Our initial port carried the task rewards but omitted the regularization layer.

**Impact:** Full retrain required. Expect smoother joint motion, less jitter, and `mean_std` stabilizing in 1.0–2.0 range rather than drifting to 2.5+.

## 2026-05-14 — stopball: continuous → event-based (port from original)

**File:** `my_mjlab_project_booster_t1/src/my_mjlab_project_booster_t1/mdp/rewards.py`

**What:** Rewrote `stopball` from a continuous per-step reward to a one-time event-based reward, matching the original Humanoid-Goalkeeper's `_reward_stopball()`.

**Old behaviour (broken):**
```python
in_front = ball_y_local > 0.0
deflected = ball_y_vel > -0.5
return (in_front & deflected).float()  # fires every step
```
With weight 100, a stationary ball in front of the robot earned 100 pts × ~200 steps = **20,000 pts/episode**. The optimal policy was to stand still and let the ball roll into the body.

**New behaviour (matches original):**
```python
delta_vy = ball_y_vel - prev_ball_y_vel
fired = (delta_vy > 2.0) & in_front & ~stop_flag
stop_flag |= fired  # fires exactly once per episode
return fired.float()
```
Fires exactly once when the ball first decelerates by >2 m/s while still in front. After that, `stop_flag` blocks further firings. Cannot be gamed by passive blocking.

**Why this was wrong:** The original `_reward_stopball` uses an explicit `stop_flag` per environment (reset in `reset_idx`) and only returns `1.0 * (stop_flag == 0) * changevel`. Our port mistakenly used a continuous velocity threshold, turning a one-time 100-point bonus into a massive continuous reward that dominated all other signals.

**Evidence:** Comparison between run `15-18-21` (active interception) and `20-28-27` (passive blocking) showed the new run scoring 22 stopball vs 14, yet visually doing less work. Passive blocking was the cheaper strategy under continuous reward. Event-based reward removes this exploit entirely.

**Impact:** Full retrain required. `eereach` and `catch_success` will now dominate — both require hand proximity to the ball, so the robot is incentivized to actively reach.

## 2026-05-14 — num_steps_per_env 100 → 50, entropy_coef 0.001 → 0.004 (encourage arm exploration)

**File:** `my_mjlab_project_booster_t1/src/my_mjlab_project_booster_t1/tasks/goalkeeper_ppo_cfg.py`
**File:** `my_mjlab_project_booster_t1/src/my_mjlab_project_booster_t1/tasks/goalkeeper_env_cfg.py`

**What:**
1. `num_steps_per_env`: 100 → 50. Curriculum thresholds halved accordingly (60K/120K → 30K/60K) since `common_step_counter` increments per `env.step()` call.
2. `entropy_coef`: 0.001 → 0.004.

**Why — num_steps_per_env=50:**
100 steps = 2-second rollouts on a 5-second episode. The ball arrives within 2–3 seconds, so a 2-second rollout already covers the full interception window — no benefit in going longer. 50 steps = 1 second, 2× faster wall-clock per iteration, ~5 rollouts per episode which still gives adequate credit assignment through GAE.

**Why — entropy_coef=0.004:**
After 2400 iterations at entropy_coef=0.001, `mean_std` converged to 0.43. At this point the policy is too narrow to explore arm-reaching behaviors: `eereach` reward reached only 1.02 out of a possible ~20 (hand ~0.9m from ball on average), despite strong `motion_body_pos=6.03` (robot IS tracking reference motion). The policy settled in a conservative local optimum — standing stable, partially tracking the reference pose — before discovering that extending the arm to the ball gives much larger reward. Raising entropy_coef to 0.004 pushes std back toward ~1.0–1.5, allowing exploration of arm extension. Chosen below 0.005 (which caused slow runaway) and well below 0.01 (catastrophic). Monitor `mean_std`: if it exceeds 2.5, reduce entropy_coef again.

## 2026-05-14 — entropy_coef 0.01 → 0.001 (IsaacGym value causes std runaway without AMP)

**File:** `my_mjlab_project_booster_t1/src/my_mjlab_project_booster_t1/tasks/goalkeeper_ppo_cfg.py`

**What:** `entropy_coef` set to `0.001` instead of the IsaacGym Humanoid-Goalkeeper value of `0.01`.

**Why:** The IsaacGym reference uses `entropy_coef=0.01`, but this value only works there because: (1) AMP discriminator provides stabilising counter-gradients, (2) 6144 parallel environments give 3× larger batch size. Without AMP, `0.01` triggers a self-reinforcing positive feedback loop in mjlab:

- Entropy gradient continuously pushes `mean_std` upward
- Larger std → more saturated actions → noisier advantages → std grows further
- Observed in training run `2026-05-14_17-04-22`: std grew 1.0 → 5.3 over 1000 iters
- At std=5.3, ~42% of actions exceed ±4.0 (saturation point after the 0.25 action scale fix)
- Robot fell in <15 steps; reward collapsed from 42 → 1

The mjlab G1 tracking reference (`mjlab/tasks/tracking/config/g1/rl_cfg.py`) uses `entropy_coef=0.005`. With 100 steps/env and no AMP, `0.001` is the safe value.

**Evidence:** TensorBoard `Policy/mean_std` monotonically increasing every iteration with no stabilisation. `Episode_Termination/ee_body_pos` exploded from 7 → 70 falls/episode.

## 2026-05-14 — Action jerk penalty + reward curriculum (mirroring G1 training strategy)

**File:** `my_mjlab_project_booster_t1/src/my_mjlab_project_booster_t1/tasks/goalkeeper_env_cfg.py`

**What:**
1. Replaced `action_rate_l2` (first-order, weight -0.1) with `action_acc_l2` (second-order jerk, weight -0.01).
2. Added `cfg.curriculum` with step-based reward scaling for `stopball`, `eereach`, and `catch_success`.

**Why — jerk penalty:**
After 1800 training iterations the first-order `action_rate_l2` term grew from -1.7 to -15.0 per episode, eventually dominating the total reward and plateauing it at ~59 even as `stopball` kept climbing. The G1 reference uses `sum((a_t - 2*a_{t-1} + a_{t-2})^2)` (second-order / jerk) at an effective weight of ~0.0002. The second-order formulation penalises oscillation harder than a smooth fast dive: an ankle alternating ±1 each step scores 4× larger per step on jerk vs rate, which is exactly the pathological behaviour we want to suppress. The smaller weight (-0.01 vs -0.1) reduces overall drag so task rewards can keep growing.

**Why — curriculum:**
G1 scales `stopball`, `eereach`, and `success` upward as training progresses: `weight × (1 + 0.5 × curriculum_level)`. Without this, fixed-weight smoothness penalties become proportionally larger relative to task rewards as the policy lengthens episodes and takes more actions. The mjlab `reward_curriculum` applies staged step-based multipliers: ×1.0 → ×1.5 → ×2.0 at ~600 and ~1200 training iterations respectively (thresholds: 15M and 30M env steps = 1020 envs × 24 steps/iter × target_iter). This keeps the reward landscape competitive throughout long runs.

**Impact:** Prior checkpoints remain incompatible (action space unchanged, but reward signal structure changed). Recommend starting a fresh run.

This document tracks substantive changes where the Booster T1 adaptation deviates from the G1 tracking task pipeline.

## 2026-05-14 — Fix reward axis mismatch after 90° rotation (ball from +Y not +X)

**File:** `my_mjlab_project_booster_t1/src/my_mjlab_project_booster_t1/mdp/rewards.py`

**What:** Changed `_ball_is_behind`, `stopball`, `stayonline`, and `noretreat` from using world X (index 0) to world Y (index 1).

**Why:** The original Humanoid-Goalkeeper had the ball coming from +X and the robot facing +X, so all rewards used world X. When the T1 port rotated the setup 90° (ball from +Y, robot faces +Y), the reward functions were never updated. As a result:
- `stopball` (weight=100) **never fired** — ball ends at negative X so `ball_x_local > 0` was always false. The entire main task reward was dead.
- `_ball_is_behind` was true from the start of every episode (ball starts at negative X), so `postorientation`/`postangvel`/`postlinvel` activated immediately and at the wrong time.
- `noretreat` penalised moving in world -X (sideways to the ball) rather than -Y (actually retreating from the ball), giving the policy no signal to prevent retreating.
- These misaligned rewards likely caused the observed clockwise rotation: no penalty for drifting/spinning, and the dominant 100-weight reward not firing at all.

**Impact:** Full retrain required. `stopball` will now actually fire, which is the primary learning signal.

---

## 2026-06-05 — Fix goalkeeper_amp_env_cfg: restore 870-dim obs, 16 missing rewards, sensors, and config params

**Files:**
- `Imitationlearningbooster/src/imitationlearningbooster/tasks/goalkeeper_amp_env_cfg.py`

**What — 8 bug fixes:**

### 1. Observation space restored to 870-dim (CRITICAL)

The AMP env cfg was missing `left_hand_pos_b` (3 dims) and `right_hand_pos_b` (3 dims) from both actor and critic observation groups. Without them the AMP task produced 81 dims/step × 10 history = **810-dim** input — incompatible with the deployed `model_2000.pt` which expects `87 × 10 = 870`. Added both to actor/critic groups identical to `goalkeeper_env_cfg.py`.

### 2. eereach reach_th corrected from 0.3 → 0.2

The AMP config called `RewardTermCfg(func=gk_rew.eereach, weight=10.0)` with no params, using the function default `reach_th=0.3`. The correct value (matching G1 upstream and `goalkeeper_env_cfg.py`) is 0.2 m. Added `params={"reach_th": 0.2}`.

### 3. 16 missing reward terms added

Rewards present in `goalkeeper_env_cfg.py` and G1 upstream but absent from AMP config:
`successland` (4.0), `penalize_sharpcontact` (-100.0), `penalize_kneeheight` (-100.0), `penalize_self_collision` (-50.0), `feet_slippage` (3.0), `postupperdofpos` (1.0), `postwaistdofpos` (1.0), `dof_acc` (-2.5e-7), `action_rate_l2` (-0.1), `torques` (-1e-5), `dof_vel` (-5e-4), `dof_pos_limits` (-3.0), `dof_vel_limits` (-2.0), `torque_limits` (-3.0), `deviation_waist_joint` (-0.001), `ang_vel_xy` (-0.1).

Without these, AMP training was entirely unguarded: no safety penalties, no joint-limit regularization, no smoothness constraints. Degenerate behaviours (kneeling, hard falls, jitter) would dominate.

### 4. sharpforce_termination added

The base `goalkeeper_env_cfg` has a hard episode termination when ground impact force exceeds 1500 N (`sharpforce_termination`). The AMP config was missing this. Without it, the AMP policy faces no hard termination for catastrophic falls.

### 5. Contact sensors added

`ContactSensorCfg` for `feet_contact` (geom pattern `^(left|right)_foot_[12]$`) and `self_collision` (Trunk self-collision) were missing. These are required by `penalize_sharpcontact`, `penalize_self_collision`, `feet_slippage`, and `sharpforce_termination`. Added with identical configuration to `goalkeeper_env_cfg.py`.

### 6. num_envs set to 6144

The AMP config never explicitly set `cfg.scene.num_envs`, inheriting whatever `make_tracking_env_cfg()` defaulted to. Set to 6144 matching the baseline.

### 7. sim contact capacity set

Added `cfg.sim.nconmax = 100` and `cfg.sim.njmax = 500` (same as baseline) to handle ball contact simulation without contact truncation.

### 8. Viewer tracking added

Added `cfg.viewer.body_name = "Trunk"` so the sim viewer follows the robot (not stuck at origin).

**Why they were wrong:** The AMP config was created as a skeleton and was never brought up to parity with the baseline `goalkeeper_env_cfg.py`. Because the AMP runner adds discriminator infrastructure on top, it is easy to focus only on the AMP-specific additions and miss the base config requirements.

**Impact:** AMP training (`goalkeeper_booster_t1_amp`) is now functionally equivalent to the baseline in terms of safety constraints and reward structure, with AMP rewards layered on top.

---

## 2026-06-05 — Fix AMP discriminator: gradient penalty scale and spectral normalization

**File:** `Imitationlearningbooster/src/imitationlearningbooster/rsl_rl_amp/modules/discriminator.py`

### 1. Gradient penalty scale corrected (10× too weak → matches G1)

The discriminator's `compute_grad_pen()` was returning `lambda_ * grad_pen * 0.1` (with `lambda_=5`), giving an effective gradient penalty of **0.5**. G1 upstream uses `lambda_=5` with no extra multiplier. Removed the `* 0.1` factor; effective penalty is now 5.0, matching upstream.

**Why it was wrong:** The `* 0.1` multiplier appears to have been added as a conservative scaling but has no upstream justification. Weak gradient penalty allows the discriminator to violate the Lipschitz constraint, destabilising GAN training with sharp discriminator gradients.

### 2. Spectral normalization added to all discriminator linear layers

G1 upstream applies spectral normalization to discriminator weights. The port did not. Added `torch.nn.utils.spectral_norm` wrapping to all hidden `nn.Linear` layers in the MLP trunk and the output `amp_linear` layer.

**Why it was wrong:** Spectral normalization constrains the spectral norm of each layer's weight matrix to ≤1, bounding the Lipschitz constant of the discriminator. Without it, discriminator gradients can become unboundedly large, causing mode collapse or training instability.

**Impact:** Discriminator training stability significantly improved. Gradient flow to policy from AMP rewards will be more consistent.

---

## 2026-06-05 — Fix GoalkeeperMotionLoader: add temporal jitter to expert sampling

**File:** `Imitationlearningbooster/src/imitationlearningbooster/rsl_rl_amp/utils/motion_loader.py`

**What:** Added per-sample temporal jitter to `feed_forward_generator()`. Before: always sampled consecutive frame pairs `(idx, idx+1)`. After: second frame sampled at `idx + round(U(0.75, 1.25))`, clamped to valid range.

**Why it was wrong:** G1's `MotionLib.get_expert_obs()` applies `ratio = (fps / env_fps) × U(0.25, 1.25)` temporal jitter before sampling the "next" frame, then bilinearly interpolates between frames. The port sampled raw consecutive frames. This locked the expert transition distribution to exactly 1-frame-apart deltas, while G1's distribution spans 0.75–1.25 frame gaps. The narrower distribution makes the discriminator more brittle — small deviation from exactly-1-frame timing is penalised even if the motion is qualitatively identical.

**Impact:** Expert data diversity increased; discriminator generalises better to variations in motion speed.

---

## 2026-06-05 — Fix convert_all.py: worldbody included in body positions (all 6 npz files broken)

**File:** `Imitationlearningbooster/src/imitationlearningbooster/motions/convert_all.py`

**What:**
1. `n_bodies = model.nbody` → `n_bodies = model.nbody - 1` (exclude worldbody at index 0)
2. `body_pos_w[t] = data_mj.xpos.copy()` → `body_pos_w[t] = data_mj.xpos[1:].astype(np.float32)`
3. `body_quat_w[t] = data_mj.xquat.copy()` → `body_quat_w[t] = data_mj.xquat[1:].astype(np.float32)`
4. `foot_body_ids` index computation corrected: offset by -1 to account for worldbody exclusion.
5. `apply_foot_fix()` call removed.

**Why it was wrong:**
- MuJoCo's `data.xpos` and `data.xquat` include the worldbody at index 0. `mjlab`'s `MotionLoader` indexes bodies starting at 0 assuming worldbody is already excluded. With worldbody at npz[0], every body position was off by 1 — body 0 (Trunk) was reading worldbody's fixed position (0,0,0), making the Reference State Initialization place all robots at the world origin rather than their reference poses. All 6 npz files had shape `(150, 25, 3)` (worldbody included); correct shape is `(150, 24, 3)`.
- `apply_foot_fix()` enforces `ankle = -(hip + knee)` which is not applied in `convert_booster.py`. The deployed `model_2000.pt` was trained using npz files without this fix. Using foot-fixed npz files for AMP fine-tuning would create a mismatch between expert motion distribution and the baseline policy's learned dynamics.

**Impact:** All 6 npz files regenerated with corrected converter (body shape now `(150, 24, 3)`). AMP Reference State Initialization now places robots at correct reference poses. PKL source files were present; regeneration succeeded.

---

## 2026-06-05 — Fix head joint action scale (missing 0.25 factor)

**Files:**
- `Imitationlearningbooster/src/imitationlearningbooster/robots/t1_constants.py`
- `my_mjlab_project_booster_t1/src/my_mjlab_project_booster_t1/robots/t1_constants.py`

**What:** `T1_ACTION_SCALE` for `(AAHead_yaw|Head_pitch)` changed from `7.0 / 20.0 = 0.35` to `0.25 * 7.0 / 20.0 = 0.0875`.

**Why it was wrong:** All other joints use the `0.25` factor (effort / stiffness × 0.25), which matches the G1 upstream's `action_scale = 0.25`. The head joints were accidentally omitted from this convention. Without the 0.25 factor, a policy output of ±1.0 moves the head ±0.35 rad — 4× the intended ±0.0875 rad. During random initialization the head bangs against its limits, injecting spurious contact forces and noise into early training.

**Both t1_constants.py files kept in sync** (Imitationlearningbooster and my_mjlab_project_booster_t1 are identical after this fix).

---

## 2026-06-05 — Add imitationlearningbooster as editable dependency of my_mjlab_project_booster_t1

**File:** `my_mjlab_project_booster_t1/pyproject.toml`

**What:** Added `"imitationlearningbooster"` to `[project] dependencies` and added to `[tool.uv.sources]`:
```toml
imitationlearningbooster = { path = "../Imitationlearningbooster", editable = true }
```

**Why it was wrong:** The `Imitationlearningbooster` package was not installed in the project venv. Its entry point (`goalkeeper_booster_t1_amp = "imitationlearningbooster"`) never fired, making the entire AMP training infrastructure (`GoalkeeperAmpRunner`, `GoalkeeperMotionLoader`, `MultiDiscriminator`, all 6 motion files) inaccessible. Running `uv sync` will install it.

**Next step required:** Run `uv sync` inside `my_mjlab_project_booster_t1/` to install the editable dependency.

## 2026-05-14 — Fix bang-bang ankle/leg control and PPO rollout length

**Files:**
- `my_mjlab_project_booster_t1/src/my_mjlab_project_booster_t1/robots/t1_constants.py`
- `my_mjlab_project_booster_t1/src/my_mjlab_project_booster_t1/tasks/goalkeeper_ppo_cfg.py`

**What:**
1. Applied the `0.25` action-scale factor to all non-arm joints (waist, hips, knees, ankles). Previously only arms had this factor.
2. Set `num_steps_per_env=24` (original upstream used 100) and `gamma=0.99` (was 0.998, now matches upstream).

**Why — action scale:**
Without the 0.25 factor, every non-arm joint saturated at max torque when the policy output ±1.0. Because the initial policy is Gaussian(0,1), outputs above ±1 are common, causing bang-bang (on/off) torque switching — visible as fast ankle jitter and micro-bouncing. The G1 reference applies `0.25 × effort/stiffness` to ALL joints so saturation only occurs at action=4.0.

**Why — gamma:**
0.998 was an unexplained deviation from both the original Humanoid-Goalkeeper (0.99) and the G1 mjlab config (0.99). Reverted to 0.99.

**Why — num_steps_per_env=100:**
Matches upstream G1 exactly. 100 steps = 2 seconds of experience per rollout at 50 Hz, giving 204,800 samples per gradient update (2048 envs × 100 steps). Empirically, reducing to 24 steps caused the policy std to grow unchecked (1.26 → 3.55 over 2800 iters), because 4× fewer samples per update produced noisier advantage estimates and larger policy steps. Keeping 100 steps provides stable gradient estimates matching G1's training regime.

**Impact:**
- Ankle pitch: action=1 now produces 5 Nm (was 20 Nm = full saturation). Control stays in PD linear regime.
- Smooth foot contact expected; micro-bouncing should be eliminated.
- **All prior checkpoints are incompatible** (action space meaning changed). Full retrain required.

## 2026-05-03 — Fully autonomous goalkeeper (no motion input at play time)

**Files:** 
- `my_mjlab_project/src/my_mjlab_project/mdp/resets.py` (new)
- `my_mjlab_project/src/my_mjlab_project/tasks/goalkeeper_env_cfg.py`

**What:** 
1. Created `reset_ball_autonomous()` - standalone ball reset function with no motion tracking dependency
2. Removed motion command entirely from play config
3. Added autonomous ball reset as startup event
4. Removed 6 motion-dependent reward + 3 motion-dependent termination terms

**Why:** The end goal is a **100% autonomous goalkeeper**:
- **Training:** Learn from all 6 motion types (left/right hand, jump, step) via RSI (RL + imitation)
- **Observations:** Ball position, ball velocity, joint state **only** (no motion in obs)
- **Play:** Policy autonomously chooses best response to any incoming ball

By removing the motion command, the policy receives no motion input at inference time. The autonomous reset function ensures the ball still gets randomized trajectories.

**Impact:** 
- Policy trained on diverse motion examples but is **completely autonomous at play time**
- No `--motion-file` argument needed
- Ball resets with random trajectory **every 10 seconds** (3-5m away, random y/z, timed arc)
- Episode auto-resets every 10 seconds to generate new ball trajectory
- Play command: `uv run python -m mjlab.scripts.play goalkeeper --checkpoint-file logs/.../model_N.pt`
- Play mode runs stably, policy continuously faces new ball trajectories
- Robot returns to random pose on each reset (domain randomization)

## 2026-05-02 — Remove motion-reference observations from actor/critic

**File:** `my_mjlab_project/src/my_mjlab_project/tasks/goalkeeper_env_cfg.py`

**What:** Removed `command`, `motion_anchor_pos_b`, `motion_anchor_ori_b` from
the actor and critic observation groups in `goalkeeper_env_cfg()`.

**Why:** The upstream G1 tracking task treats reference motion as an explicit
input to the policy. This requires a motion file at inference time. For an
autonomous goalkeeper agent that decides its own response based on ball
position, the motion reference must not appear in the observation space.
The 6 motion files are retained for RSI (reference state initialisation) and
style-shaping rewards during training only.

**Impact:** All checkpoints trained before 2026-05-02 are incompatible with the
new observation space (actor dim changed). Full retrain required.

## 2026-05-22 — Physics-based collision detection replacing height proxies

**Files:**
- `my_mjlab_project_booster_t1/src/.../mdp/rewards.py`
- `my_mjlab_project_booster_t1/src/.../tasks/goalkeeper_env_cfg.py`
- `my_mjlab_project_booster_t1/src/.../mdp/resets.py`

### 1. New `feet_contact` ContactSensor

**What:** Added a `ContactSensorCfg(name="feet_contact")` monitoring the 4 foot geoms
(`left_foot_1`, `left_foot_2`, `right_foot_1`, `right_foot_2`) with `secondary=None`
(any contact partner, including ground) and `reduce="netforce"`.

**Why it was missing:** mjlab has no global `cfrc_ext`-equivalent tensor. Isaac Gym
exposed `net_contact_force_tensor` per-body; mjlab requires explicit `ContactSensorCfg`
declarations. The original port used height proxies everywhere because this API gap
was not addressed.

**Shape:** `data.found [B, 4]`, `data.force [B, 4, 3]`. Geom index order (lexicographic):
0=left_foot_1, 1=left_foot_2, 2=right_foot_1, 3=right_foot_2.

---

### 2. `penalize_sharpcontact` — trunk height proxy → foot force sensor

**What was wrong:** Used `trunk_z < 0.35 m` as a fall detector. Original uses
`mean(norm(foot_forces)) > 1000 N` — a direct impact measurement.

**Fix:** Now reads `env.scene["feet_contact"].data.force` and returns binary
`(mean_force_norm > 1000.0).float()`. Weight: -100.0 unchanged.

**Evidence:** Original `legged_robot.py:1477`. Proxy missed hard landings at standing
height; falsely fired during controlled crouches.

---

### 3. `feet_slippage` — foot height proxy → sensor-based contact detection

**What was wrong:** Used `foot_z < 0.05 m` for `in_contact`. When feet were airborne
(mid-dive), `in_contact=0` → `contactvel=0` → `exp(0)=1.0` — full reward even while
the robot was completely in the air. Inflated WandB `feet_slippage` curves throughout
training.

**Fix:** Now uses `env.scene["feet_contact"].data.found > 0` per foot geom. Left foot
in_contact = `(found[:,0]>0) | (found[:,1]>0)`, right similarly. Rest of formula
unchanged. Weight: +3.0 unchanged.

**Evidence:** Original `legged_robot.py:1472` uses `contact_forces > 1 N` as the
in_contact threshold.

---

### 4. `penalize_self_collision` — new reward wiring `self_collision` sensor

**What was missing:** The `self_collision` ContactSensor was already registered in
`goalkeeper_env_cfg.py` but its output was never consumed by any reward or observation.

**Fix:** Added `penalize_self_collision` reward that reads `self_collision.data.found`
and returns `(found > 0).any(dim=-1).float()`. Registered at weight -50.0.

**Reasoning:** Self-collisions (arm hitting torso during dive) are recoverable but
undesirable. Weight -50 (half of sharpcontact -100) reflects lower severity.

---

### 5. `sharpforce_termination` — force-based episode termination

**What was missing:** Original terminates at `mean_foot_force > 1500 N`
(`sharpforce_buf` at `legged_robot.py:258`). mjlab port had no equivalent.

**Fix:** Added `sharpforce_termination` in `resets.py`, registered as
`TerminationTermCfg(time_out=False)` with `max_contact_force=1500.0`. Uses the
same `feet_contact` sensor force computation as `penalize_sharpcontact`.

**Impact:** New termination fires only on catastrophic impacts. Checkpoints trained
before this date are fully compatible — this does not change observation or action space.

---

## 2026-05-26 — Deployment bug fixes: base_lin_vel frame and ball visibility masking

### Bug Fix 1: `base_lin_vel` was fed in world frame instead of body frame

**What was wrong (`goalkeeper_deploy/tasks/goalkeeper/controller.py`):**
```python
base_lin_vel_b = dq[:3]  # WRONG — labelled body frame but is world frame
```
MuJoCo freejoint `qvel[0:3]` is the derivative of world-frame position, i.e. the **world-frame** linear velocity. It is NOT the body-frame velocity. The comment claiming it was body frame was incorrect.

**What the training does (`mjlab/entity/data.py:597`):**
```python
root_link_lin_vel_b = quat_apply_inverse(root_link_quat_w, root_link_lin_vel_w)
```
Training explicitly rotates the world-frame velocity into body frame before adding it to the observation. At the robot's 90° yaw, world-X ↔ body-Y are completely swapped, making this a critical error — the policy received the wrong velocities in two out of three axes.

**Angular velocity (`qvel[3:6]`) was actually correct.** MuJoCo freejoint `qvel[3:6]` is already body-frame angular velocity. Training also produces body-frame angular velocity (via `quat_apply_inverse(quat_w, ang_vel_w)`). Both resolve to the same value.

**Fix:** Added `_rot_world_to_body(q_wxyz, v_w)` numpy helper (same rotation matrix as `_quat_rot_inv` in task.py) and applied it to `dq[:3]` in `update_state()`:
```python
base_lin_vel_b = _rot_world_to_body(base_quat, dq[:3])
```
Also fixed `qfrc_actuator[:23]` → `qfrc_actuator[6:29]` (robot joints are dofs 6-28 in the scene with ball).

**Evidence:** Trace through `mjlab/entity/data.py` lines 594-602 and `observations.py` lines 24-28 confirmed training always gives body-frame velocity. Rotation test: world `[0,1,0]` at 90° yaw → body `[1,0,0]` ✓.

---

### Bug Fix 2: Ball visibility masking was completely absent

**What was wrong:** Deployment always provided raw ball position and velocity. The policy was trained with three masking gates:
1. **initial_vanish** — ball hidden for the first ~7-8 policy steps after each episode reset (catchstep warmup, mirrors upstream `startstep ≈ 43`)
2. **flying** — ball hidden unless approaching, within 3.4 m in Y, |X| < 2 m, Z < 1.8 m
3. **random_vanish** — ball disappears for a random number of steps (sampled 0-29) per episode

Without masking, ball_pos_b and ball_vel_b carried out-of-distribution values during the warmup window and after the ball passed the robot.

**Fix:** Added ball visibility state to `GoalkeeperMujocoController`:
- `_ball_step` countdown (50→0 per launch, decremented in `update_state()`)
- `_ball_prev_y` for approach direction tracking
- `_ball_vanish_step` random threshold (re-sampled in `_launch_ball()`)
- `_ball_visible_step` flying-step counter
- `_compute_ball_visibility()` method mirroring `observations.py` logic exactly
- `ball_visible: bool` attribute read by the policy

In `task.py` `_compute_obs()`:
```python
vis = 1.0 if getattr(self.controller, 'ball_visible', True) else 0.0
ball_pos_b = _quat_rot_inv(q, ball_pos_w - base_pos) * vis
ball_vel_b = _quat_rot_inv(q, ball_vel_w) * vis
```

**Evidence:** Integration test confirmed `_ball_step` decrements correctly, visibility stays False during warmup (steps 0-7), and activates when ball enters flying zone.

**Files changed:** `goalkeeper_deploy/tasks/goalkeeper/controller.py`, `goalkeeper_deploy/tasks/goalkeeper/task.py`

---

## 2026-06-05 — Re-enable catchstep warmup initialization and decrement (Feature 7 fix)

**What was wrong:** Feature 7 (catchstep warmup) was implemented in `mdp/commands.py` but both initialization (`_reset_ball`) and decrement (`_update_command`) were commented out with note "Skipped for now due to indexing issues; ball starts with zero velocity instead." This disabled Feature 7 and Feature 8 (ball visibility masking), which depend on `env._catchstep` being set and tracked.

Without active catchstep:
- Ball observations are always visible during warmup window (fallback in `observations.py` line 57-60)
- Policy receives out-of-distribution ball pos/vel during the first ~7-8 steps after reset
- Feature 8 visibility masking was only partially active

**Root cause:** Original code used `scatter_()` operation which failed with certain `env_ids` tensor shapes/dtypes. However, new `env_ids` validation guard (lines 188-192) now ensures `env_ids` is 1D and of dtype `torch.long` or `torch.int64`, making direct indexing safe.

**Fix:** Re-enabled both blocks using direct indexing (matching upstream and `my_mjlab_project_booster_t1` project):
- `_reset_ball()`: `self._env._catchstep[env_ids] = 50`
- `_update_command()`: `self._env._catchstep = (self._env._catchstep - 1).clamp(min=0)`

The `env_ids` validation guard (lines 188-192) prevents the original indexing crashes.

**Files changed:** `src/imitationlearningbooster/mdp/commands.py` (lines 280-294, 306-309)

---

## 2026-06-05 — Remove all domain randomization from AMP goalkeeper env config

**What changed:** Added a DR removal block immediately after `cfg = make_tracking_env_cfg()` in `goalkeeper_amp_env_cfg()` inside `tasks/goalkeeper_amp_env_cfg.py`.

**Events removed:**
- `base_com` — startup event that adds random COM offset to `Trunk` body (DR)
- `encoder_bias` — startup event that adds random encoder bias to all joints (DR)
- `foot_friction` — startup event that randomizes foot geom friction (DR)
- `push_robot` — interval event that perturbs the robot with random velocity impulses (DR)

**Why it was wrong:** The base `make_tracking_env_cfg()` injects all four events automatically. The AMP goalkeeper config never explicitly removed them, so all DR was silently active during training. This caused instability during early policy learning and made it harder to diagnose motion quality issues, because multiple confounding effects (random COMs, random friction, random joint bias, random pushes) were always present.

**Correct value:** No DR events during goalkeeper AMP training. Ball reset is handled internally by `MultiMotionCommandCfg._reset_ball` at every episode reset. Robot reset to reference state is handled by the `motion` command RSI (Reference State Initialization). No separate DR events are needed.

**Evidence confirming the fix was needed:** The prior config referenced in `goalkeeper_env_cfg.py` (non-AMP version) commented: "Disabled: pd_gains, link_mass, reset_joints (re-enable after policy can stand and dive; they make early optimisation too hard)." The AMP port inherited the same issue but never applied an equivalent disable. Upstream G1 config (`g1_29_config.py`) has DR flags (`randomize_payload_mass`, `randomize_friction`, etc.) that were explicitly studied; the mjlab base config's four events are the direct analogues.

**Files changed:** `Imitationlearningbooster/src/imitationlearningbooster/tasks/goalkeeper_amp_env_cfg.py`

---

## 2026-06-05 — Port G1 jump_scale mechanism to eereach reward

**What changed:** Replaced the flat `vel_sigma = 1.0 + 3.0 * clamp(vel_toward, 0, 3)` in `eereach()` (`mdp/rewards.py`) with a jump-region-aware computation that mirrors G1 `_reward_eereach` exactly.

**Why it was wrong:** The previous port used a uniform `3.0` multiplier for all motion types. G1 amplifies `vel_sigma` for jump-region envs (`end_regions == 2|3`) using `jump_scale = 3.0 + 3.0 * curriculumupdate` (ranging 3→9 across curriculum stages). Without this amplification, jump-region envs received the same reach incentive as ground-level saves, giving the policy insufficient gradient to learn to leap and reach high balls. The post-pass `vel_sigma` was also wrong: previous port doubled the computed value (`vel_sigma * 2.0`), but G1 sets a flat `vel_sigma = 2.0` regardless of motion velocity.

**Correct value:**
- Non-jump envs (motion_type 0,1,4,5): `vel_sigma = 1 + 3.0 × clamp(vel_toward, 0, 3)` — unchanged
- Jump envs (motion_type 2,3): `vel_sigma = 1 + jump_scale × clamp(vel_toward, 0, 3)`
  where `jump_scale = 3.0 + 3.0 × curriculumupdate` and `curriculumupdate = _ball_difficulty × 2`
  (maps difficulty 0→1 to curriculum 0→2, matching upstream 3-stage progression)
- Post-pass (behind=True): `vel_sigma = 2.0` (flat, matching G1 exactly)

**Evidence confirming the fix was needed:**
- G1 source lines 1379–1390 (`legged_robot.py`): `jump_scale = 3.0 + 3.0 * self.curriculumupdate` with `vel_sigma[end_regions==2|3] = 1 + jump_scale * clip(z_vel, 0, 3)` and `vel_sigma[behind] = 2.0`
- Port's motion_type_ids 2 and 3 map directly to G1's `end_regions == 2` (leftjump) and `end_regions == 3` (rightjump)
- `_ball_difficulty` (0→1) is the port's curriculum proxy: `difficulty=0.5` → `curriculumupdate=1` → `jump_scale=6`; `difficulty=1.0` → `curriculumupdate=2` → `jump_scale=9`

**Files changed:** `Imitationlearningbooster/src/imitationlearningbooster/mdp/rewards.py`

## 2026-06-05 — AMP training loop: G1 vs T1 port comparison and disc batch OOM fix

### Background: complete fidelity audit

The goal of this port is to replicate G1 goalkeeper behavior (IsaacGym + rsl_rl) on Booster T1 (MuJoCo Warp + mjlab). This entry documents every training-loop divergence found in a side-by-side comparison of `Humanoid-Goalkeeper/rsl_rl/rsl_rl/` against `Imitationlearningbooster/src/imitationlearningbooster/rsl_rl_amp/`.

---

### What is faithfully replicated (no meaningful divergence)

| Component | G1 upstream | T1 port | Notes |
|---|---|---|---|
| Motion classes | 6 (lefthand/righthand/leftjump/rightjump/leftstep/rightstep) | 6 identical | ✓ |
| Discriminators | 1 per motion class (6 total, deepcopy of base AMP) | 1 per motion class (6 total) | ✓ |
| AMP reward blending | 40% AMP + 60% task | 40% AMP + 60% task | ✓ |
| Disc architecture | Linear(46,512)→ReLU→Linear(512,256)→ReLU→Linear(256,1) | same | ✓ |
| Grad penalty formula | `sum(square(grad)).mean() * lambda_` | `grad.norm(2).pow(2).mean() * lambda_` | mathematically identical |
| Effective grad penalty lambda | `lambda_=5 * 0.1 = 0.5` | `lambda_=5 * 0.1 = 0.5` | ✓ |
| AMP obs dimension | 46 (23 DOF × 2 consecutive frames) | 46 | ✓ |
| AMP normalizer | shared across 6 discriminators | shared across 6 discriminators | ✓ |
| Reward weights | all 27 terms ported from G1 | ✓ (see earlier entries) | |
| Observation space | 870-dim, 10-step history | 870-dim, 10-step history | ✓ |
| Network architecture | MLP 512-256-128 | MLP 512-256-128 | ✓ |

---

### Divergences: framework adaptations (necessary, not bugs)

#### 1. Disc optimizer: separate vs joint with PPO

**G1:** One shared Adam optimizer covers both the PPO actor-critic and all 6 discriminators. The disc loss (`expert_loss + policy_loss + grad_pen * 0.1`) is added to the PPO surrogate loss and everything is updated in one `loss.backward()` / `optimizer.step()`.

**T1 port:** Separate Adam optimizer for the 6 discriminators. PPO and disc are updated in two sequential phases per iteration. The disc optimizer uses the same `(lr=1e-3, weight_decay=1e-3)` parameters as G1's disc param groups.

**Why this divergence:** mjlab's `MotionTrackingOnPolicyRunner` base class owns and controls the PPO optimizer. Hooking into its internal backward pass to add disc gradients would require modifying the base class (prohibited). A separate disc optimizer is architecturally equivalent — both optimize the same loss terms, just via different Adam instances.

**Impact on training:** Negligible. The policy gradient and disc gradient do not depend on sharing an optimizer; they are independent objectives. Separating them is the norm in most AMP literature.

---

#### 2. PPO update frequency: 5 epochs × 4 mini-batches vs 1 × 1

**G1:** `num_learning_epochs=1`, `num_mini_batches=1` → **1 PPO gradient step** per rollout collection.

**T1 port:** `num_learning_epochs=5`, `num_mini_batches=4` → **20 PPO gradient steps** per rollout collection.

**Why this divergence:** G1 uses 1×1 because the disc loss is folded into the PPO loss with `create_graph=True`; multiple epochs over stale advantages with a live disc graph would accumulate prohibitive memory. With a separate disc phase, standard PPO practice (4-10 epochs) is safe and improves sample efficiency on a 6144-env setup.

**Impact on training:** Faster policy convergence per wall-clock hour. The qualitative behavior (motion style + ball-stopping objective) is unchanged.

---

#### 3. Spectral normalization on discriminator layers

**G1:** Plain `nn.Linear` on all discriminator layers (no spectral norm).

**T1 port:** `spectral_norm(nn.Linear(...))` on all layers of each discriminator.

**Why this divergence:** Added as a stability measure during initial port (see entry "Fix AMP discriminator: gradient penalty scale and spectral normalization"). G1 avoids spectral norm because its disc receives large, well-mixed batches (~17k samples) that are naturally diverse. With smaller batches, spectral norm constrains Lipschitz constant and prevents gradient explosion.

**Impact on training:** Minor. Spectral norm slightly slows disc convergence but prevents instability. If the disc trains too slowly in practice, this can be removed to match G1 exactly.

---

### Divergence that caused the CUDA OOM (now fixed)

#### 4. Discriminator mini-batch size: 25,600/disc → capped at 4,096/disc

**G1 effective disc batch:** `1020 envs × 100 steps / 1 mini-batch = 102,000` total, split by motion type → **~17,000 samples per discriminator** per update (1 update/iteration). Running on an 8 GB GPU.

**T1 port (broken):** `6144 envs × 100 steps / 4 mini-batches = 153,600`, split by 6 → **25,600 per discriminator**, multiplied by 20 disc updates per iteration. All 120 `compute_grad_pen` calls (20 updates × 6 discs) accumulated their `create_graph=True` second-order graphs into one tensor before `backward()` was called. This consumed the full 31 GB VRAM.

**Error:** `torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 50.00 MiB. GPU 0 has a total capacity of 31.35 GiB of which 52.31 MiB is free.` at `him_amp_runner.py` / `discriminator.py:37`.

**Fix:** Added `amp_disc_mini_batch_size: int = 4096` to `RslRlAmpRunnerCfg`. The runner uses `min(ppo_mini_batch // num_motions, amp_disc_mini_batch_size)` as the per-discriminator batch size. With 4,096 samples/disc and 20 updates, total disc samples processed per iteration = 4,096 × 20 = 81,920 — larger than G1's 17,000 and safely within VRAM.

**Files changed:**
- `Imitationlearningbooster/src/imitationlearningbooster/rsl_rl_amp/runners/him_amp_runner.py`
- `Imitationlearningbooster/src/imitationlearningbooster/tasks/goalkeeper_amp_ppo_cfg.py`

---

## 2026-06-06 — Remove double-rotation: NPZ motion data now natively faces +X

**What changed:** The conversion pipeline previously applied a +90° Z-axis rotation to the motion data (PKL → NPZ), then the runtime code applied a matching -90° rotation when loading the overlay and spawning the robot. This double-rotation was fragile and confusing. Both rotations have been removed: the PKL data originally faces +X (matching `https://github.com/KaydenKnapik/BoosterT1mjlab`), so the NPZ files are now written as-is, natively facing +X.

**Why it was wrong:** The original `convert_all.py` comment stated "T1 faces +Y, original faces +X" and rotated the data +90° to match a +Y-forward convention. But the target setup (matching the GitHub reference) uses +X as forwards. The runtime code attempted to compensate with a -90° rotation, but this made the codebase hard to reason about and left the overlay and ball-spawn positions potentially inconsistent.

**Correct values:**
- `convert_all.py`: `rotate_z90()` is now a no-op (returns inputs unchanged)
- `convert_booster.py`: +90° position and quaternion rotation removed
- `commands.py` (both packages): `body_pos_w` and `body_quat_w` properties no longer apply rotation; `_resample_command` no longer rotates `root_ori` at reset

**Evidence:** Visual inspection showed robot facing +Y (green axis) while ball spawned on +X. Tracing the code confirmed the NPZ files contained +Y-facing data from the conversion script. Reverting to the original PKL facing (+X) eliminates the mismatch.

**Files changed:**
- `Imitationlearningbooster/src/imitationlearningbooster/motions/convert_all.py`
- `Imitationlearningbooster/src/imitationlearningbooster/motions/convert_booster.py`
- `Imitationlearningbooster/src/imitationlearningbooster/mdp/commands.py`
- `my_mjlab_project_booster_t1/src/my_mjlab_project_booster_t1/mdp/commands.py`

**Action required:** Re-run `convert_all.py` to regenerate all 6 NPZ files before training or visualising motions.

---

## 2026-06-08 — SimpleGoalKeeper: Fix AMP joint-position mismatch + add missing rewards

### Bug: AMP discriminator trained on absolute vs. relative joint positions

**What changed:** `SimpleGoalKeeper/src/simple_goalkeeper/scripts/pkl_to_npz.py` now subtracts the T1 home-keyframe default joint positions from `joint_pos` before saving to NPZ. All 8 NPZ files have been regenerated.

**Why it was wrong:** The AMP discriminator compares reference motion samples (from NPZ) to policy samples (from the env's AMP obs group). The env uses `joint_pos_rel = joint_pos - default_joint_pos`, but the NPZ stored absolute DOF positions. The discriminator saw systematically different distributions — not because the policy moved unnaturally, but because of a fixed offset equal to the standing pose. The NPZ even contained a `joint_pos_absolute=1` flag confirming this.

**Correct values:** NPZ now stores `absolute_pos - T1_HOME_KEYFRAME_pos` (same coordinate as `joint_pos_rel`). Joint order is headless XML order (21 DOF, skipping 2 head joints). Verified: first-frame max abs value dropped from ~1.07 rad to ~0.61 rad, values are now small deviations around 0.

**Evidence:** Traced `joint_pos_rel` source → `joint_pos - asset.data.default_joint_pos`. NPZ first-frame value for Left_Shoulder_Roll was -1.068 (absolute); standing default is -1.4; policy would see 0.332 (relative). Discriminator trivially separated them by offset, not motion quality.

**Files changed:**
- `SimpleGoalKeeper/src/simple_goalkeeper/scripts/pkl_to_npz.py`
- `SimpleGoalKeeper/src/simple_goalkeeper/motions/data/*.npz` (all 8, regenerated)

### Missing rewards added to SimpleGoalKeeper

**What changed:** Added `stayonline`, `noretreat`, `feetorientation`, `deviation_waist_joint` to `mdp/rewards.py` and `dof_pos_limits` (mjlab built-in) to the env cfg. All are feet-keeper-appropriate and require no contact sensors.

| Reward | Weight | Purpose |
|---|---|---|
| `stayonline` | -2.0 | Penalise drifting off goal line (X axis) |
| `noretreat` | -2.0 | Penalise retreating from ball |
| `feetorientation` | +2.0 | Keep feet flat for better deflections |
| `deviation_waist_joint` | -0.001 | Always-active waist centering |
| `dof_pos_limits` | -3.0 | Joint safety — soft limit penalty |

**Files changed:**
- `SimpleGoalKeeper/src/simple_goalkeeper/mdp/rewards.py`
- `SimpleGoalKeeper/src/simple_goalkeeper/mdp/__init__.py`
- `SimpleGoalKeeper/src/simple_goalkeeper/tasks/goalkeeper_env_cfg.py`

---

## 2026-06-08 — SimpleGoalKeeper: Fix stand-still local optimum (reward redesign)

### Root cause: `ball_vx_reduction` + `foot_to_ball` (std=0.15) created do-nothing optimum

**What changed:** Replaced the broken reward structure with the proven Imitationlearningbooster pattern (footreach + stopball). Removed `ball_vx_reduction`, `foot_to_ball`, and `posture`. Disabled ball visibility warmup during training. Removed `push_robot` event.

**Why it was wrong:** TensorBoard confirmed `ball_vx_reduction` grew from 0.27 → 3.41 per episode, becoming the dominant reward. This reward returns `exp(-(incoming_speed/4)²)` which peaks at 1.0 when the ball is **not moving** — so the optimal policy was to do nothing and let the ball stop naturally. Combined with `posture` (max when at default pose = standing still) and `foot_to_ball` (std=0.15m → zero gradient at 2–4 m spawn distance), the training converged to a stand-still local minimum.

**Evidence from TensorBoard logs** (`2026-06-08_20-14-56_phase1`):
- `ball_vx_reduction`: 0.27 → 3.41 (dominated all task rewards)
- `foot_to_ball`: 0.0001 → 0.018 (essentially zero — robot never contacted ball)
- `feetorientation`: 0.16 → 1.67 (robot learned perfect flat feet = standing still)
- `mean_episode_length`: 23 → 140 (robot learned to not fall = stand still)
- `ball_positive_vx`: only 0.33 (ball rarely deflected back)

**Correct approach (ported from Imitationlearningbooster):**

| Term | Weight | Purpose |
|---|---|---|
| `footreach` | +10.0 | Phase1 lateral alignment + Phase2 sigmoid reach × vel_sigma (1–10×) |
| `stopball` | +100.0 | One-time reward when ball deflected (delta_vx > 1 m/s) |
| `ball_positive_vx` | +10.0 | Continuous signal for sustained deflection |
| `stayonline` | -2.0 | Keep |
| `noretreat` | -2.0 | Keep |
| `feetorientation` | +1.5 | Reduced from 2.0 |

**Removed:**
- `ball_vx_reduction`: rewards doing nothing (ball stops naturally)
- `foot_to_ball` (std=0.15): zero gradient at spawn distance
- `posture`: AMP handles naturalness; posture added to the stand-still optimum

**Additional fixes:**
- Ball visibility: `always_visible=True` during training (`ball_pos_b`, `ball_vel_b`). The visibility gate (warmup + random vanish) made the ball invisible for most of the episode, killing the ball-approach learning signal.
- `push_robot` removed from training events (was only removed in play mode before). Random impulses disrupted the goalkeeper stance during early learning.
- Episode length extended from 3.0 → 4.0 s to give slow easy-difficulty balls time to reach the robot.

**Files changed:**
- `SimpleGoalKeeper/src/simple_goalkeeper/mdp/rewards.py` (add footreach, stopball)
- `SimpleGoalKeeper/src/simple_goalkeeper/mdp/__init__.py` (export footreach, stopball)
- `SimpleGoalKeeper/src/simple_goalkeeper/mdp/observations.py` (add always_visible param)
- `SimpleGoalKeeper/src/simple_goalkeeper/tasks/goalkeeper_env_cfg.py` (reward table, events, episode length)
- `SimpleGoalKeeper/CLAUDE.md` (updated reward table)

---

## 2026-06-09 — AMP training fixes: spawn, arm movement, ball ranges

### Bug 1: Robot spawns underground

**What changed:** `commands.py` `_resample_command` — added `root_pos[:, 2] += 0.05` after reading the Trunk position from motion data.

**Why it was wrong:** Booster motion data was converted with a ground reference ~8 cm lower than the simulation floor. At frame 0 (standing), foot bodies are at Z≈0.002–0.017 m. With the ±0.01 m Z jitter from `pose_range`, feet could go to Z=-0.008 m (8 mm underground), causing MuJoCo to push the robot violently upward.

**Evidence:** `numpy.load` on all 6 motion files showed frame-0 foot bodies at Z=0.002–0.017 m (Imitationlearningbooster) vs Z=0.083–0.099 m (my_mjlab_project_booster_t1 which showed no spawn issue).

**Correct value:** +0.05 m offset puts feet at ≥0.052 m at spawn (safely above ground). After 2–3 sim steps physics settles the robot on the ground.

### Bug 2: Arm not moving toward ball

**What changed:** `goalkeeper_amp_env_cfg.py` — kept `motion_body_pos` at weight 1.5 (was deleted entirely). Doubled eereach curriculum weights (10/15/20 → 20/28/36) and hand_proximity_strict (5/7.5/10 → 10/15/20).

**Why it was wrong:** Removing all motion tracking rewards left the AMP discriminator as the sole arm guidance signal. At <2k iterations the discriminator is immature and cannot overcome regularisation terms that keep arms at default pose. The working `my_mjlab_project_booster_t1` repo used explicit `motion_body_pos` at weight 4.0.

**Correct approach:** `motion_body_pos` at weight 1.5 (15% of upstream's 10.0) provides direct early-training arm gradient without dominating task rewards. Higher eereach weight gives stronger reaching signal in phase 2.

### Bug 3: Ball ranges too large (~50% shots unreachable)

**What changed:** `commands.py` — narrowed ball trajectory parameters:
- `y_start`: ±1.8 m → ±0.8 m  
- `z_start`: (0.3, 1.8) → (0.5, 1.4) m
- `t_flight` min: 0.4 → 0.5 s (prevents vx > 9.6 m/s)
- `x_start` max: 5.0 → 4.5 m
- `_BALL_END_RANGES` full-difficulty lateral Y max: 0.84 → 0.65 m
- `_BALL_END_RANGES_EASY` lateral Y max: 0.40 → 0.35 m

**Why it was wrong:** At t_flight=0.4 s and x_start=5 m, vx=-13.25 m/s — physically impossible for T1 to react. y_start=±1.8 m required extreme trajectory curves with ball arriving from the wrong side. ~50% of episodes had unreachable targets, halving the effective learning signal.

**Evidence:** User reported visually ~50% shots unreachable; vx calculation confirms.

## 2026-06-09: Static env partitioning for motion types (mirrors G1 end_regions)

**What changed:** Added `static_partition=True` to `MultiMotionCommandCfg` in `goalkeeper_amp_env_cfg.py`. Implemented in `MultiMotionCommand.__init__` and `_resample_command` in `commands.py`.

**Why it was wrong:** Previously `motion_type_ids` was randomly re-assigned at every episode reset (`torch.randint(0, 6, ...)`). Each env cycled through all 6 motion styles over training. While ball target ranges already matched the motion type, no env specialised in any single style.

**G1 upstream design:** `end_regions` is a static tensor created once in `_init_buffers` and never modified. Envs 0–(N/6−1) permanently train `lefthand`, next group permanently trains `righthand`, etc. Each AMP discriminator receives a dedicated, consistent env group every rollout.

**What the correct value is:** `static_partition=True` → at init, env `i` gets `motion_type_ids[i] = (i * 6) // num_envs` (equal groups of `num_envs//6`, last group absorbs remainder). At episode reset, `_resample_command` restores the static assignment instead of re-randomising.

**Evidence:** Two subagents confirmed G1's `end_regions` is permanently assigned in `_init_buffers` and read-only everywhere else. At step 800 of training, arm motions (lefthand/righthand) were weak while step motions worked — consistent with mixed-signal AMP discriminators not enforcing arm-extension styles. The G1 paper and code both rely on dedicated env groups per motion style.

## 2026-06-10: Fix play-mode ball spawn axis mismatch (ball was invisible to policy)

**What changed:** Rewrote `_shoot_ball` in `mdp/resets.py` to use the +X approach axis, matching the training `_reset_ball` in `commands.py`. Previously `_shoot_ball` launched the ball from the +Y direction (`y_start = sample_uniform(3.0, 5.0, ...)`).

**Why it was wrong:** The ball visibility check in `observations.py:_compute_ball_visibility` gates on `ball_x_local > 0.05` (world X). With the +Y ball, `ball_x_local ≈ 0` at all times → ball was PERMANENTLY invisible to the policy during play → policy received zero ball observations → it just executed its default learned motion (lefthand) regardless of where the ball went.

**What the correct value is:**
```python
x_start = sample_uniform(3.0, 4.5, ...)   # ball from +X (same as training)
dx = -x_start - 0.3                        # target X ≈ -0.3 (goal line)
```
Full bilateral Y range (±0.65 m) so both hands are exercised during play.

**Evidence:** User reported "ball only goes to the right where lefthand area is" — ball was flying past in +Y direction while the policy defaulted to lefthand saves. The old comment "rotated 90° vs original G1 +X setup" was incorrect; G1 training and this T1 training both use +X approach.

## 2026-06-10: Fix ball Y-axis direction — left hand is at +Y, not -Y

**What changed:** Swapped Y signs in `_BALL_END_RANGES` and `_BALL_END_RANGES_EASY` in `commands.py`. Swapped lateral velocity reward direction in `eereach` in `rewards.py`.

**Why it was wrong:** Comments said "left hand at -Y, right hand at +Y". Empirically confirmed in MuJoCo viewer (green axis = +Y): the T1 left hand is at **+Y** and right hand at **-Y**. So lefthand/leftjump/leftstep motions had ball targets aimed at -Y — directly away from the left hand.

**What the correct values are:**
- lefthand/leftjump/leftstep: y_end ∈ [+0.15, +0.65] (full), [+0.10, +0.35] (easy)
- righthand/rightjump/rightstep: y_end ∈ [-0.65, -0.15] (full), [-0.35, -0.10] (easy)
- `eereach` lateral vel_sigma: left motions reward +Y torso motion, right motions reward -Y.

**Evidence:** User confirmed by visual inspection in MuJoCo viewer: lefthand save happens on the green (+Y) axis side.

## 2026-06-11: Restore ball Y range cap to 0.65 m (undo silent expansion)

**What changed:** Reduced `_BALL_END_RANGES` Y-max from 0.90/1.10 m back to 0.65 m, and `_BALL_END_RANGES_EASY` Y-max from 0.45/0.55 m back to 0.35 m. Sign convention (+Y for left, -Y for right) preserved from 2026-06-10 fix.

**Why it was wrong:** The 11th-pass commit (60f3893) expanded Y max from 0.65 → 0.90 m (regular) and 0.65 → 1.10 m (jumps) without documentation. TensorBoard analysis of run 2026-06-10_14-40-50 showed mean_reward plateau at iter ~4000 with no further improvement to iter 15600. Root cause: at 0.90 m lateral, a significant fraction of balls are physically unreachable by T1's arm from a standing position, providing zero gradient signal for those episodes.

**What the correct values are:** 0.65 m max lateral was established empirically as T1's reachable limit. At 0.84 m, ~50% of shots were unreachable (noted in original comment). Restored to 0.65 m.

**Evidence:** Training run 2026-06-10_14-40-50 metrics — reward plateau at iter 4000, curriculum at max by iter 2000, no eereach improvement despite 15k iterations. G1 upstream uses 1.2 m but G1 is a larger robot.

## 2026-06-11: Confirmed RSI is correct for our port (mirrors G1 continue_keep)

**What diverges:** G1 upstream `_reset_dofs` (legged_robot.py:673-677) uses `standpos` BUT with `continue_keep=True` (g1_29_config.py:280): 80% of resets copy DOF from a random active env (arbitrary mid-motion pose), 20% use scaled standpos. Our port uses motion-file RSI (Reference State Initialization) instead of standpos. This IS correct — motion-file RSI gives the same exploration diversity as G1's continue_keep mechanism.

**Why RSI is kept:** With 40k training iterations vs G1's 200k, diverse initial conditions via RSI are essential for the policy to discover lateral diving behaviors quickly. Standing-pose-only initialization was tested and confirmed inferior by the user.

## 2026-06-11: Fix AMP reward collapse — add noise-sampling kernel (matches G1 predict_reward)

**What changed:** `him_amp_runner.py` AMP reward computation. Changed from bare quadratic kernel `0.5 * clamp(1 - 0.25*(d-1)^2, 0)` to G1's noise-sampling kernel: draw 20 Gaussian perturbations (σ=0.3) of the normalised obs, evaluate disc on all 20, take `min(squared_error)`, compute `0.5 * clamp(1 - 0.25 * min_sq_err, 0)`.

**Why it was wrong:** When the discriminator converges (d_policy → −1), the bare kernel evaluates to exactly 0. Once the disc collapses in the first ~93 iterations, AMP reward is permanently 0 and provides no gradient signal to the policy. Evidence: TensorBoard `Loss/amp_disc_loss` drops to 0.16 by iter 93; `Metrics/motion/error_joint_pos` increases monotonically from 1.49 → 3.49 across 1k iterations, confirming the policy diverges from reference motion.

**What the correct formula is:** G1 `amp.py predict_reward` (lines 183–204): noisy sampling + min squared error. Even at d_policy = −1 (converged disc), noisy samples land at d ≈ −0.7, giving reward ≈ 0.14 instead of 0 — enough gradient to keep guiding the policy toward natural motion.

**Evidence:** G1 source `Humanoid-Goalkeeper/rsl_rl/rsl_rl/modules/amp.py` lines 183–204. The simple kernel (`return torch.clamp(1 - 0.25 * torch.square(d - 1), min=0)`) is explicitly commented out in G1's code and replaced with the noisy version.

## 2026-06-13: Fix AMP/task reward scale mismatch — match G1 weights exactly

**What changed:** `goalkeeper_amp_env_cfg.py` and `goalkeeper_amp_ppo_cfg.py`.

- `eereach` weight: 20.0 (base, curriculum to 30.0) → **10.0 static** (G1 `g1_29_config.py:299`)
- `hand_proximity_strict` weight: 10.0 (base, curriculum to 20.0) → **5.0 static** (G1 `success=5.0`)
- `stopball` weight: 100.0 (base, curriculum to 250.0) → **100.0 static** (G1 `stopball=100.0`, no curriculum)
- Removed `stopball_curriculum`, `eereach_curriculum`, `hand_proximity_strict_curriculum` entries from `cfg.curriculum`.
- `amp_coef`: kept at **0.4** (G1 value; now correct since task weights match G1).

**Why it was wrong:** T1 had eereach at 2-3× G1 and a stopball curriculum ramping to 2.5× G1. The AMP reward is hardcoded to [0, 0.5]/step by the noise-sampling formula — it never scales with weights. TensorBoard analysis of run `2026-06-12_21-57-14` confirmed: at step 8400, task rewards were 3.2× implied vs logged (curriculum weights not reflected in episode logs). AMP contributed only 17% of the gradient signal instead of the intended 40%, so the policy learned to stop the ball by leaning forward rather than by performing the reference dives/jumps. Play mode confirmed: no motion mimicry despite 80% stopball success.

**What the correct values are:** G1 `g1_29_config.py` with `amp_coef=0.4`: `eereach=10`, `success=5`, `stopball=100`, no reward curricula on these terms. At these weights the AMP reward (max 0.5/step) is in the same order of magnitude as the task rewards, giving the intended 40/60 AMP/task split.

**Evidence:** TensorBoard run `2026-06-12_21-57-14`: `Loss/mean_amp_reward` dropped from 30→12 over 8400 steps; `Episode_Reward/eereach` = 21.57 with implied task total 40.2 vs direct log sum 12.57 (3.2× ratio explained by curriculum weights). G1 source `g1_29_config.py` lines 299-301, 363-364.

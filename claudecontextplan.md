# Goalkeeper RL Improvement Plan — Booster T1 (mjlab)

**Date:** 2026-05-24  
**Status:** Exhaustive comparison done (2026-05-24). Implementing missing items in priority order.

---

## What Was Done Before (2026-05-20)

| Fix | File | What changed |
|-----|------|-------------|
| IMU velocity noise | `goalkeeper_env_cfg.py` | Added `Unoise(±0.1)` to `base_lin_vel`, `Unoise(±0.2)` to `base_ang_vel` |
| Per-joint torque limits | `rewards.py` `_T1_EFFORT_MAP` | Replaced universal 50 Nm cap with KaydenKnapik values |
| Effort limits | `robots/t1_constants.py` | Updated all effort_limit to KaydenKnapik hardware-verified values |
| XML force clamps removed | `assets/booster_t1/T1_serial_clean.xml` | Removed all `actuatorfrcrange` |
| Joint velocity noise | `goalkeeper_env_cfg.py` | Increased from ±0.5 to ±1.5 |
| Actuator command delay | `robots/t1_constants.py` | Added `delay_min_lag=2, delay_max_lag=8` |

---

## Exhaustive Comparison Results (2026-05-24)

Full systematic diff of original Isaac Gym G1 vs MJLab Booster T1 port. Prioritized by training impact.

---

## PRIORITY 1 — Bugs That Actively Hurt Training (Fix Now)

### 1A. `stopball` — `_sb_flag` not set when ball crosses goal line

**VERIFIED 2026-05-24: NOT an active training bug. Low priority.**

The divergence is real (port only sets `_sb_flag` on velocity spike, not on ball-behind-goal) but has zero practical impact:
- `fired` requires `in_front = ball_y_local > 0.0` — once ball is behind goal, this is always False, so stopball can never refire regardless of `_sb_flag`
- `_ball_is_behind()` uses the position check `ball_y_local < 0.0` independently, which is always correct — stale `_sb_init_vy` has no effect on this branch
- No back-wall geometry exists in the scene, so a behind-goal ball cannot bounce back to in-front

**No fix needed for training correctness.** Could add for code clarity but not a priority.

---

### 1B. `successland` — fires everywhere, not just jump regions, missing airborne tracking

**VERIFIED 2026-05-24: REAL ACTIVE BUG. Severity 4/5.**

Original only fires for regions 2+3 (jump regions = 33% of envs) AND only after `has_in_air` is latched True (robot actually left ground, root Z > 1.0 m). Components: +1 per step airborne, +5 at two-foot landing, −1 at one-foot landing.

Port fires for 100% of envs every post-pass step when feet are within 0.05 m of ground — trivially satisfied while standing still. This creates a spurious constant reward baseline in the post-save phase for all envs, drowning the intended jump-landing safety signal and potentially discouraging movement after the ball passes.

**Fix:** Add `_has_in_air` tracking and gate the reward on it:
```python
# In successland():
in_air = robot.data.root_link_pos_w[:, 2] - env_z > 1.0
if not hasattr(env, "_has_in_air"):
    env._has_in_air = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
just_reset = env.episode_length_buf <= 1
env._has_in_air[just_reset] = False
env._has_in_air |= in_air
feet_down = (foot_z - env_z < height_threshold).all(dim=-1)
return (feet_down & env._has_in_air & behind).float()
```
**Note:** The original also gives +5 landing bonus and −1 one-foot penalty — the simplified port loses these incentives too.

**File:** `my_mjlab_project_booster_t1/src/my_mjlab_project_booster_t1/mdp/rewards.py`

---

### 1C. `penalize_sharpcontact` / `sharpforce` termination — wrong mean divisor

**VERIFIED 2026-05-24: REAL ACTIVE BUG. Severity 3/5.**

Original averages contact force over 2 bodies (L+R ankle_roll_link). Isaac Gym's net contact force at each body already aggregates all geom contact points. Mean over 2 → a single foot at 2000 N → mean = 1000 N → fires.

Port averages over 4 geoms. Same 2000 N on one foot splits across left_foot_1 (1000 N) + left_foot_2 (1000 N) + right_foot_1 (0) + right_foot_2 (0) → mean = 500 N → does NOT fire. Port is exactly 2× less sensitive. A 1500 N single-foot impact (termination-worthy) produces 750 N mean → below both thresholds.

**Fix:** Use per-foot max then mean over feet (not mean over all 4 geoms):
```python
force_per_geom = sensor.data.force.norm(dim=-1)  # [B, 4]
left_max  = force_per_geom[:, :2].max(dim=-1).values
right_max = force_per_geom[:, 2:].max(dim=-1).values
mean_force = (left_max + right_max) / 2.0
```

**Files:** `rewards.py` (`penalize_sharpcontact`), `resets.py` (`sharpforce_termination`)

---

## PRIORITY 2 — Missing Curriculum Mechanisms (Add Next)

### 2A. Ball difficulty curriculum — completely missing

**VERIFIED 2026-05-24: REAL, HIGH IMPACT.**

**Original starting vs max ranges per region:**
| Region | Starting width | Max width | Starting height | Max height |
|---|---|---|---|---|
| 0/1 (low, L+R) | ±[0.2, 1.2] | ±[0.0, 1.8] | [0.4, 1.2] | [0.3, 1.5] |
| 2/3 (high, L+R) | ±[0.0, 1.0] | ±[0.0, 1.5] | [1.2, 1.6] | [1.2, 1.8] |
| 4/5 (ground, L+R) | ±[0.2, 1.2] | ±[0.0, 1.8] | [0.1, 0.3] | [0.1, 0.3] |

Original starts at ≥0.2 m from center (excludes easy center shots) and expands outward as `curriculumupdate` grows. `curriculumupdate = floor(mean_episode_length / 50)`, triggered every 500 steps. Range expands by `0.3 × curriculumupdate` per update, clipped to `command_bound`.

**Note:** During training, ball reset is handled by `MultiMotionCommand._reset_ball` (in `commands.py`), NOT `_shoot_ball` in `resets.py`. `_reset_ball` uses `_BALL_END_RANGES[0] = (-0.84, -0.2, 0.4, 1.2)` — fixed to the LEFT-HAND intercept zone only. This is intentional for the single lefthand motion but means there is zero difficulty expansion.

**Fix:** Add a `ball_difficulty_curriculum` CurriculumTermCfg that sets `env._ball_end_x_range` on the env, then `MultiMotionCommand._reset_ball` reads it instead of the hardcoded `_BALL_END_RANGES`. Start narrow (center-ish shots), expand over training.

**Stages (match original's progressive expansion):**
- Stage 0 (step 0): x_end ∈ [−0.4, −0.1], z_end ∈ [0.5, 1.1] (easy central-left)
- Stage 1 (step stage1_step): x_end ∈ [−0.7, −0.1], z_end ∈ [0.4, 1.2] (medium)
- Stage 2 (step stage2_step): x_end ∈ [−0.84, −0.2], z_end ∈ [0.4, 1.2] (full lefthand range)

**Files:** `mdp/commands.py` (`_reset_ball` reads `env._ball_end_x_range`), `tasks/goalkeeper_env_cfg.py` (add curriculum term)

---

### 2B. `dof_pos_limits` and `torque_limits` — not in curriculum

**VERIFIED 2026-05-24: REAL, MODERATE IMPACT.**

Original: at `curriculumupdate > 1.0` → weight doubles (−3 → −6); at `> 2.0` → triples (−3 → −9). `curriculumupdate = floor(mean_episode_length / 50)`, triggers every 500 common steps. Port keeps both fixed at −3.0.

**Fix:** Add both to the existing `cfg.curriculum` dict:
```python
"dof_pos_limits_curriculum": CurriculumTermCfg(
    func=mjlab_mdp.reward_curriculum,
    params={
        "reward_name": "dof_pos_limits",
        "stages": [
            {"step": 0,           "weight": -3.0},
            {"step": stage1_step, "weight": -6.0},
            {"step": stage2_step, "weight": -9.0},
        ],
    },
),
"torque_limits_curriculum": CurriculumTermCfg(
    func=mjlab_mdp.reward_curriculum,
    params={
        "reward_name": "torque_limits",
        "stages": [
            {"step": 0,           "weight": -3.0},
            {"step": stage1_step, "weight": -6.0},
            {"step": stage2_step, "weight": -9.0},
        ],
    },
),
```

**File:** `goalkeeper_env_cfg.py`

---

### 2C. `hand_proximity_strict` — not in curriculum

**Original:** `success` reward scales with `curriculumupdate` same as eereach/stopball.

**Port:** Fixed at weight 5.0.

**Fix:** Add to curriculum dict at stages 5.0 → 7.5 → 10.0.

---

## PRIORITY 3 — Reward Logic Improvements (Train After P1+P2 Stable)

### 3A. `eereach` — measures to current ball position, not predicted intercept

**VERIFIED 2026-05-24: Real, moderate impact.**

Original: `end_target` is computed at reset as the ball intercept point at x=0.1 m (arm reach plane). `catch_prop = (0.1 - ball_start_x) / (ball_end_x - ball_start_x)`. Updated mid-flight (snaps to current ball position when ball is 0.1–0.5 m from robot). `dist` is per-env distance to `end_target` using the region-assigned hand (left for regions 0,2,4; right for 1,3,5). `dist` is also included directly in the critic observation.

Port: measures `min_dist` from closest hand to current ball position. No region-to-hand assignment.

**Impact:** During ball-far phase (>0.5 m), original trains predictive intercept (robot leads the ball). Port trains reactive tracking (robot chases the ball). Moderate — the velocity amplification term partially compensates. Not blocking current training but worth implementing for faster convergence.

**Fix:** In `MultiMotionCommand._reset_ball` (commands.py), compute and store `env._ball_end_target`:
```python
catch_prop = (0.1 - y_start_local) / (-y_start_local - 0.3)  # when ball reaches Y~0.1
env._ball_end_target[env_ids] = ball_pos_w + delta * catch_prop
```
Then `eereach()` uses `env._ball_end_target` when ball Y > 0.5, current ball pos when Y ≤ 0.5.

---

### 3B. `eereach` — Phase 1 pre-positioning missing

**VERIFIED 2026-05-24: Real, moderate-high impact.**

Original: when ball is >1.5 m away (~15% of episode = ~22 policy steps), the full eereach reward is replaced by a pre-positioning reward:
- `asidegoal = clip(end_target_local_y, -1, 1)`, zeroed if abs < 0.3 — lateral alignment to intercept
- `verticalgoal = clip(torso_z - clip(end_target_z, 0.3, 1.2), 0, 1)` — height adjustment

Port: sigmoid of distance to a ball 3–5 m away evaluates to ~0 (`exp(-5 * (3m - 0.2m)) ≈ 0`). No gradient signal during Phase 1 — ~15% of the episode produces near-zero eereach signal.

**Impact:** Missing ~15% of dense gradient per episode. Policy has no incentive to pre-position laterally toward the correct side before the ball arrives. `stayonline` penalty only prevents Y-drift, doesn't reward X-lateral pre-positioning.

**Fix:** Add Phase 1 branch in `eereach()` when `ball_y_local > 1.5`:
```python
far = ball_y_local > 1.5
# Phase 1: reward lateral X pre-positioning toward predicted intercept
if env._ball_end_target is available:
    aside = clamp(end_target_local_x / 0.8, -1, 1)  # normalize to ±1
    phase1_rew = 1.0 - aside.abs().clamp(0, 1)       # reward being on correct side
    rew = torch.where(far, phase1_rew, rew)
```

---

### 3C. `success_flag` / `success_rate` tracking — absent

**VERIFIED 2026-05-24: Real but low priority for training, important for logging.**

Original tracks per-env save rate: `success_rate (N, 3)` = [successes, attempts, rate]. Used in logging. The `_sb_flag` proxy in the port adequately replaces the reward-multiplier role.

**Impact on training:** Minimal. **Impact on insight:** You have no per-env save-rate visibility during training.

---

## PRIORITY 4 — Domain Randomization (After P1–P3 Stable)

*(These were already planned. Unchanged.)*

### Phase 1 — KaydenKnapik Baseline DR

#### 1a. Push Robot (6-DOF, interval 1–3 s)
```python
cfg.events["push_robot"].interval_range_s = (1.0, 3.0)
cfg.events["push_robot"].params["velocity_range"] = {
    "x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (-0.4, 0.4),
    "roll": (-0.52, 0.52), "pitch": (-0.52, 0.52), "yaw": (-0.78, 0.78),
}
```

#### 1b. Foot Friction (startup, shared random)
```python
cfg.events["foot_friction"].params["asset_cfg"].geom_names = r"^(left|right)_foot_[12]$"
# range: (0.3, 1.2), shared_random=True — already configured
```

#### 1c. Encoder Bias (startup)
```python
cfg.events["encoder_bias"] = EventTermCfg(
    mode="startup", func=mjlab_dr.encoder_bias,
    params={"asset_cfg": SceneEntityCfg("robot"), "bias_range": (-0.015, 0.015)},
)
```

#### 1d. Base COM Offset (startup)
```python
cfg.events["base_com"] = EventTermCfg(
    mode="startup", func=mjlab_dr.body_com_offset,
    params={
        "asset_cfg": SceneEntityCfg("robot", body_names=("Trunk",)),
        "operation": "add",
        "ranges": {0: (-0.025, 0.025), 1: (-0.025, 0.025), 2: (-0.03, 0.03)},
    },
)
```

### Phase 2 — Ball Domain Randomization

#### 2a. Ball Mass
```python
cfg.events["ball_mass"] = EventTermCfg(
    mode="reset", func=mjlab_dr.body_mass_offset,
    params={"asset_cfg": SceneEntityCfg("ball"), "operation": "scale", "ranges": (0.7, 1.3)},
)
```

#### 2b. Ball Friction
```python
cfg.events["ball_friction"] = EventTermCfg(
    mode="reset", func=mjlab_dr.geom_friction,
    params={"asset_cfg": SceneEntityCfg("ball", geom_names=("ball_geom",)),
            "operation": "scale", "ranges": (0.5, 2.0), "shared_random": True},
)
```

### Phase 3 — Observation Delay (optional, after P1+P2 stable)
```python
cfg.observations["actor"].delay_lag = (0, 3)
```

### Phase 4 — PD Gain Randomization (optional, advanced)
```python
cfg.events["pd_gains"] = EventTermCfg(
    mode="startup", func=mjlab_dr.pd_gains,
    params={"asset_cfg": SceneEntityCfg("robot"),
            "stiffness_range": (0.85, 1.15), "damping_range": (0.85, 1.15)},
)
```

---

## KaydenKnapik Effort Limits (reference)

| Joint group | Value |
|---|---|
| Arms (Shoulder/Elbow 4×2) | 36 Nm |
| Waist | 40 Nm |
| Hip Pitch | 55 Nm |
| Hip Roll / Hip Yaw | 40 Nm |
| Knee Pitch | 65 Nm |
| Ankle Pitch | 50 Nm |
| Ankle Roll | 50 Nm |

---

## PRIORITY 5 — Additional Findings from Full Codebase Read (2026-05-24)

These were found by reading every file in the original, not just reward functions.

### 5A. `base_lin_vel` / `base_ang_vel` computed from pelvis body, not root_states

**Original:** `base_lin_vel` and `base_ang_vel` are extracted from the **pelvis rigid body** (`rigid_body_states[:, upper_body_index, 7:10]`), not from `root_states`. This matters because for G1 (and T1), the root link might not be at the pelvis.

**Port:** Uses `root_link_lin_vel_b` and `root_link_ang_vel_b`. For T1 where `Trunk` is the root body, these should match. Low risk but worth confirming body identity.

### 5B. `catchstep` warmup — robot forced to stand during ball launch

**Original:** `catchstep` counts down from 50 steps. While `catchstep > startstep` (~40–47 steps), `_compute_torques` overrides `joint_pos_target` to `init_dof_pos` — **forcing the robot to hold standing pose while the ball is launched**. This prevents the robot from acting before seeing the ball.

**Port:** No `catchstep` warmup. Robot acts from step 0. Ball starts at Y=3–5m and takes 0.5–1.0s to arrive, so the robot does "see" the ball from the start — but it could act erratically in those early frames.

**Impact:** Low — the ball arrival time gives natural de-facto warmup. But the explicit standing pose lock may improve early training stability.

### 5C. Ball visibility curriculum — 3-phase masking

**Original:** Ball position observation is masked in 3 cases:
1. `initial_vanish`: ball zeroed until `catchstep < startstep` (before ball is launched)
2. `flying`: ball only visible when in a valid flight region (x∈[0.05,3.4], z<1.8, etc.)
3. `random_vanish`: ball position zeroed when `catchstep > vanish_step` (random 0–30)

This trains the robot to handle partial/missing ball information — critical for sim-to-real where ball tracking may be noisy.

**Port:** Ball always visible (obs delay handles some noise). No masking curriculum.

**Impact:** Missing sim-to-real robustness for ball tracking failures.

### 5D. HIM-PPO internal model — actor input is 119-D, not full obs

**Original actor input:** `cat(obs_history[-96:], history_latent(16), estimate_ball(6), argmax(estimate_region)(1))` = 119-D
- `history_encoder`: 960→128→64→16 (compresses 10-step history to latent)
- `ball_estimator`: 960→128→32→6 (predicts ball pos+vel from history)
- `region_estimator`: 960→128→32→6 (classifies which of 6 regions, cross-entropy loss)

**Port actor:** Standard MLP on full 900-D (or whatever obs size). No internal model. Standard PPO.

**Impact:** Significant architectural difference. The internal model allows the actor to use compressed temporal information and estimated ball state even when ball obs is masked. Without it, the policy relies entirely on raw noisy obs. This is the "HIM" in HIM-PPO.

### 5E. AMP reward blending — 40% AMP + 60% task

**Original:** `rewards = 0.4 * amp_reward + 0.6 * task_reward`. 6 separate discriminators (one per region). AMP obs = DOF positions at t and t+1 (58 dims).

**Port:** Pure task rewards + motion tracking (explicit L2 penalties). Different mechanism but serves same purpose.

### 5F. Domain randomization — port has subset of original

Original has all these active:
- Joint injection `±0.01` (actuation noise)
- Actuation offset `±0.01` (systematic bias)
- Payload mass `[-5, +10] kg`
- CoM displacement `±0.1 m`
- **Link mass randomization** `[0.8, 1.2]×` — NOT in port (disabled)
- **Friction randomization** `[0.1, 2.0]` — port has [0.3, 1.2] (narrower)
- **Restitution randomization** `[0.0, 1.0]` — NOT in port
- Kp/Kd randomization `[0.8, 1.2]×` — disabled in port
- Initial joint pos randomization with `continue_keep` (80% copy from another env)
- Push robots
- Ball velocity perturbation every 0.5s

Port is missing: link mass DR, full friction range, restitution DR, joint injection/actuation offset equivalents.

### 5G. `_reset_dofs` — 80% state continuation

**Original:** 80% chance: copy DOF state from a randomly selected other env (`continue_keep=True`). 20% chance: sample fresh from standpos ± noise. This reuses existing good states across envs, accelerating convergence.

**Port:** Uses RSI (Reference State Initialization) from motion library — starts from a random frame of the reference motion. Different approach but serves similar diversity-of-init purpose.

---

## Items NOT Ported (Intentional or Hardware-Appropriate)

| Item | Reason |
|---|---|
| `airfeetorientation` | Weight = 0 in original G1 config — inactive, not a bug |
| `catch_success` reward | Defined in port's rewards.py but NOT wired into cfg — dead code, remove or wire up |
| `end_regions` (6 regions) | Region-based ball partitioning replaced by uniform random + difficulty curriculum |
| `startstep` / ball vanish | Ball always visible in port (handled by obs delay) |
| `static_obs` | Vestigial in original, not used in port |
| `waist_yaw / waist_roll` | T1 only has 1 waist DOF vs G1's 3 |
| AMP adversarial discriminator | Replaced by MJLab reference-state tracking — architectural difference, intentional |
| Wrist joints | T1 has no wrist DOF |

---

## Verification Checklist Per Fix

Before marking any fix as done:
- [ ] No NaN in rewards after 1000 steps
- [ ] `stopball` only fires on genuine saves (ball slows to near-stop)
- [ ] `eereach` still climbing (not regressing)
- [ ] No new termination spikes in first 100 episodes
- [ ] DIVERGENCE_FROM_UPSTREAM.md updated with dated entry

---

## Background Agents (launched 2026-05-24)

Two subagents are doing full reads of:
- `/home/isaak/BEPImitationlearning/Humanoid-Goalkeeper` (all files)
- `/home/isaak/BEPImitationlearning/Imitationlearningbooster` (all files)

This plan will be updated once those complete — they may reveal additional discrepancies not caught in the initial reward-focused comparison.

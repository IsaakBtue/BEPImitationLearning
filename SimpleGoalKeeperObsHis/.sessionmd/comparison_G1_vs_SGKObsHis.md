# Full Comparison: Humanoid-Goalkeeper (G1) vs SimpleGoalKeeperObsHis (T1)

**Purpose:** Verify every divergence between the frozen G1 reference and our T1 port.  
**Date verified:** 2026-06-18  
**Status key:** ✅ Match | ⚠ Divergence (intentional) | ❌ Bug / missing (not intentional)

---

## 1. HIM Actor Architecture

| Property | G1 (`actor_critic.py`) | SGKObsHis (`him_actor.py`) | Status |
|---|---|---|---|
| `history_latent_dim` | `16` | `16` | ✅ |
| `estimate_ball_dim` | `6` | `6` | ✅ |
| `num_regions` | `6` (hand L/R, jump L/R, step L/R) | `2` (step L / step R) | ⚠ Intentional — 6 AMP discs → 2 |
| history_encoder | `Linear(H,128)→ReLU→Linear(128,64)→ReLU→Linear(64,16)` | Identical | ✅ |
| ball_estimator | `Linear(H,128)→ReLU→Linear(128,32)→ReLU→Linear(32,6)` | Identical | ✅ |
| region_estimator | `Linear(H,128)→ReLU→Linear(128,32)→ReLU→Linear(32,N_regions)` | Identical | ✅ |
| actor MLP hidden dims | `[512, 256, 128]` | `(512, 256, 128)` | ✅ |
| actor_input formula | `[cur_obs ‖ latent(16) ‖ ball_est(6) ‖ argmax(region)(1)]` | Identical | ✅ |
| `actor_history_length` | `10` (`num_actor_history=10`) | `10` | ✅ |
| `num_one_step_obs` | `96` (`6+3+29×2+29`) | `86` (`3+3+3+21+21+21+3+3+3+3+1+1`) | ⚠ Intentional — T1 21-DOF vs G1 29-DOF, feet obs added, hand obs removed |
| actor_input dim | `96 + 23 = 119` | `86 + 23 = 109` | ⚠ Follows from DOF count |
| obs normalizer | **None** — raw obs fed to networks | `EmpiricalNormalization(860)` applied inside `_build_him_input` | ⚠ SGKObsHis adds normalisation; G1 uses manual obs scaling (×0.25, ×0.05, etc.) |
| Distribution | `Normal(mean, scalar std param)` | `GaussianDistribution(21, scalar)` | ✅ Equivalent |

---

## 2. Training Update

### 2a. Losses

| Loss | G1 (`him_ppo.py`) | SGKObsHis (`him_amp_runner.py`) | Status |
|---|---|---|---|
| Surrogate | `max(-adv×ratio, -adv×clamp(ratio,1±clip))` | Identical | ✅ |
| Value (clipped) | `max((v−ret)², (v_clip−ret)²)` | Identical | ✅ |
| Entropy | `−entropy_coef × entropy.mean()` | Identical | ✅ |
| est_loss (ball) | `MSE(ball_estimator(history), gt_ball)` | Identical | ✅ |
| region_loss | `CrossEntropy(region_logits, gt_region_class)` | Identical | ✅ |
| Smooth loss | `policy_smooth_coef×‖μ−μ_mix‖² + value_smooth_coef×‖v−v_mix‖²` | Identical | ✅ |
| AMP disc loss | `(d_expert−1)² + (d_policy+1)² + grad_pen×0.1` | Identical | ✅ |
| **Total** | `surrogate + est + region + value_coef×value − entropy_coef×entropy + smooth + amp_disc` | Identical | ✅ |

### 2b. GT Extraction for Supervision

| Item | G1 | SGKObsHis | Status |
|---|---|---|---|
| GT ball pos+vel (6D) | `critic_obs_batch[:, -13:-7]` — sliced from end of flat 113D privileged obs | `privileged[:, :6]` — from dedicated `"privileged"` obs group | ⚠ Different storage layout, identical signal |
| GT region class | `(3 × critic_obs_batch[:, -14]).long()` — stored as `end_regions/3`, scaled back | `privileged[:, 6].long()` — raw 0 or 1 (no ÷3 needed for 2 classes) | ⚠ G1 uses `/3` encoding for 6 classes; SGKObsHis stores raw 0/1 for 2 classes. Correct for respective designs. |
| Motion ID for AMP routing | `3 × critic_obs[:, num_one_step_obs + 3]` extracted inline from obs | `env._motion_type_ids` — dedicated tensor set at startup | ⚠ Cleaner in SGKObsHis |

### 2c. Smoothness Loss

| Coefficient | G1 | SGKObsHis | Status |
|---|---|---|---|
| `smoothness_lower_bound` | `0.1` | `0.1` | ✅ |
| `smoothness_upper_bound` | `1.0` | `1.0` | ✅ |
| `eps = lower/(upper−lower)` | `0.111` | `0.111` | ✅ |
| `policy_smooth_coef` | `upper × eps = 0.111` | Identical | ✅ |
| `value_smooth_coef` | `0.1 × policy_smooth_coef = 0.0111` | Identical | ✅ |

### 2d. AMP Reward Kernel

| Property | G1 (runner) | SGKObsHis | Status |
|---|---|---|---|
| Noise σ | `0.3` | `0.3` | ✅ |
| N noise samples | `20` | `20` | ✅ |
| Reward formula | `clamp(1 − 0.25 × sq_err.min(), 0) × 0.5` | Identical | ✅ |
| AMP blend coef | `amp_coef = 0.4` | `0.4` | ✅ |

### 2e. PPO Hyperparameters

| Property | G1 | SGKObsHis | Status |
|---|---|---|---|
| `num_learning_epochs` | **`1`** (him_ppo.py L42) | **`5`** | ❌ Not documented as intentional. SGKObsHis does 5× more gradient steps per rollout. |
| `num_mini_batches` | **`1`** (L43) | **`4`** | ❌ Same: SGKObsHis does 20 gradient updates total vs G1's 1. |
| `gamma` | **`0.998`** (L45) | **`0.99`** | ❌ Not documented as intentional. Shorter effective horizon in SGKObsHis. |
| `schedule` | `"fixed"` (L52) | `"adaptive"` | ⚠ SGKObsHis uses adaptive LR (improvement over G1) |
| `clip_param` | `0.2` | `0.2` | ✅ |
| `value_loss_coef` | `1.0` | `1.0` | ✅ |
| `entropy_coef` | `0.01` (g1_29_config L369) | `0.01` | ✅ |
| `desired_kl` | `0.01` | `0.01` | ✅ |
| `max_grad_norm` | `1.0` | `1.0` | ✅ |
| `learning_rate` | `1e-3` | `1e-3` | ✅ |
| `lam` | `0.95` | `0.95` | ✅ |
| `num_steps_per_env` | **`100`** | **`50`** | ⚠ Intentional — GPU memory (32 GB vs 8 GB) |
| Adam disc trunk `weight_decay` | `1e-3` (`10e-4`) | `1e-3` | ✅ |
| Adam disc head `weight_decay` | `1e-1` (`10e-2`) | `1e-1` | ✅ |
| Grad clip scope | `actor_critic` only; disc unclipped | `chain(actor, critic)` only | ✅ |

---

## 3. Observations

### 3a. Actor one-step terms

| Term | G1 | SGKObsHis | Status |
|---|---|---|---|
| `base_lin_vel` (3D) | `× 2.0` scaling | No manual scaling (normalizer) | ⚠ Different scaling mechanism, same signal |
| `base_ang_vel` (3D) | `× 0.25` scaling | No manual scaling | ⚠ Same |
| `projected_gravity` (3D) | `× 1.0` | `× 1.0` | ✅ |
| `joint_pos_rel` (29D/21D) | `(dof_pos − default) × 1.0` | `joint_pos_rel` | ✅ |
| `joint_vel` (29D/21D) | `dof_vel × 0.05` | No manual scaling | ⚠ Same |
| `actions` (29D/21D) | `self.actions` | `last_action` | ✅ |
| `ball_pos_b` (3D) | `end_target_local` (gated) | `ball_pos_b` (gated, `always_visible=True` during train) | ✅ |
| `ball_vel_b` (3D) | `ball_vel × 0.2` | `ball_vel_b` | ✅ (scaling via normalizer) |
| `motion_type_id` (1D) | `end_regions / 3.0` ∈ {0, 0.33, 0.67, 1.0, 1.33, 1.67} | raw `0.0` or `1.0` | ⚠ Encoding differs; SGKObsHis never applies `/3` |
| `ball_landing_y` (1D) | `end_target_local` 3D full target position | `ball_landing_y_b` 1D Y-only kinematic prediction | ⚠ G1: 3D; SGKObsHis: 1D Y only |
| `hand_pos_r` (3D) | ✓ present | ❌ missing | ⚠ Intentional — feet-only, no hand obs |
| `hand_pos_l` (3D) | ✓ present | ❌ missing | ⚠ Intentional |
| `dist` (1D) | `min_hand_to_ball_dist` | ❌ missing | ❌ No foot-to-ball distance equivalent. `footreach` partially covers this but one-shot bonus (`success`) that uses this is also absent. |
| `left_foot_pos_b` (3D) | ❌ not present | ✓ added | ⚠ Intentional addition |
| `right_foot_pos_b` (3D) | ❌ not present | ✓ added | ⚠ Intentional addition |

### 3b. Privileged / Critic obs

| Item | G1 | SGKObsHis | Status |
|---|---|---|---|
| Critic obs source | `privileged_obs_buf` — flat 113D tensor = actor obs + `[lin_vel(3), end_regions/3(1), end_target(3), ball_vel(3), hand_r(3), hand_l(3), dist(1)]` | `"critic"` group = actor obs (no noise). Separate `"privileged"` group = `[ball_pos_b_true(3), ball_vel_b_true(3), motion_type_id(1)]` = 7D | ⚠ Different structure; equivalent supervision signal |
| GT available for ball estimator | Yes (last 13 of 113D) | Yes (from `"privileged"` group) | ✅ |
| GT available for region estimator | Yes (14th from end) | Yes (`privileged[:, 6]`) | ✅ |

### 3c. AMP Observations

| Property | G1 | SGKObsHis | Status |
|---|---|---|---|
| AMP obs content | `dof_pos` (joint positions) | `joint_pos_abs` | ✅ |
| Dims per frame | `29` (G1 29-DOF) | `21` (T1 21-DOF) | ⚠ Robot DOF count |
| 2-frame AMP obs dim | `29 × 2 = 58` | `21 × 2 = 42` (`AMP_OBS_DIM=42`) | ⚠ Robot DOF count |
| Terminal frame handling | No explicit handling | Pre-step obs used for done envs (prevents contamination) | ⚠ SGKObsHis more careful |

### 3d. History stacking

| Property | G1 | SGKObsHis |
|---|---|---|
| `history_length` | `10` | `10` |
| Mechanism | Manual rolling buffer shifted each step (`obs_buf = cat(obs_buf[one_step:], current_obs)`) | mjlab `ObservationGroupCfg(history_length=10)` manages internally |

---

## 4. Rewards

### 4a. Full reward table

| Reward | G1 weight | SGKObsHis weight | Notes | Status |
|---|---|---|---|---|
| `eereach` / `footreach` | +10.0 (curriculum) | +10→15→20 (curriculum) | Feet vs hands; broader ramp | ⚠ |
| `success` (one-time close-contact) | +5.0 | ❌ missing | G1 fires when `dist < 0.15 m`; no foot equivalent | ❌ Undocumented missing |
| `stopball` | +100.0 (flat) | +100→175→250 (curriculum) | SGKObsHis adds ramp | ⚠ |
| `softstop` | ❌ not present | +50→75→100 (curriculum) | SGKObsHis addition (partial deflection) | ⚠ |
| `stayonline` | −2.0 | −2.0 | | ✅ |
| `noretreat` | −2.0 | −2.0 | | ✅ |
| `successland` | +4.0 | ❌ missing | Jump landing reward; jump not in SGKObsHis | ⚠ Intentional |
| `feetorientation` | +3.0 | +3.0 | | ✅ |
| `penalize_sharpcontact` | −100.0 | −100.0 | | ✅ |
| `penalize_kneeheight` | −100.0 | −100.0 | | ✅ |
| `feet_slippage` | +3.0 | +3.0 | | ✅ |
| `postorientation` | +3.0, **behind-gated** | +3.0, **always active** | SGKObsHis removes gate — documented in docstring: AMP can't push root upright; gating removes the only upright signal during ball approach | ⚠ Intentional divergence |
| `postangvel` | +3.0, behind-gated | +3.0, behind-gated | | ✅ |
| `postlinvel` | +1.0, behind-gated | +1.0, behind-gated | | ✅ |
| `postupperdofpos` | +1.0, behind-gated | +1.0, behind-gated | | ✅ |
| `postwaistdofpos` | +1.0, behind-gated | +1.0, behind-gated | | ✅ |
| `penalize_self_collision` | ❌ not present | −50.0 | SGKObsHis addition | ⚠ |
| `ang_vel_xy` | −0.1 | −0.1 | | ✅ |
| `ang_vel_z` | ❌ not present | −2.0 | SGKObsHis addition (yaw spin prevention) | ⚠ |
| `dof_acc` | −2.5e-7 | −2.5e-7 | | ✅ |
| `action_acc_l2` / `smoothness` | −0.1 (second-order) | −0.1 | | ✅ |
| `action_rate_l2` | ❌ not present | −0.3 | SGKObsHis adds first-order smoothness | ⚠ |
| `torques` | −1e-5 | −1e-5 | | ✅ |
| `dof_vel` | −5e-4 | −5e-4 | | ✅ |
| `dof_pos_limits` | −3.0 | −3.0 | | ✅ |
| `dof_vel_limits` | −2.0 | −2.0 | | ✅ |
| `torque_limits` | −3.0 (flat) | −3.0 (flat in cfg) | CLAUDE.md documents −3→−9 curriculum; **not wired** | ❌ See below |
| `deviation_waist_joint` | −0.001 (waist_pitch) | −0.001 (full Waist joint) | | ✅ |

### 4b. stopball threshold

| | G1 | SGKObsHis |
|---|---|---|
| Threshold | `delta_vx > 2.0 m/s` | `delta_vx > 1.0 m/s` |
| Justification | PhysX rigid contacts produce sharp velocity changes | MuJoCo soft contacts produce smaller impulses; lowered documented in CLAUDE.md |

### 4c. Curriculum stages

| Curriculum | G1 | SGKObsHis |
|---|---|---|
| Ball difficulty | `curriculumupdate = int(mean_ep_len / 50)`, expands command ranges by `× 0.3` | `ball_difficulty ∈ [0.0, 0.5, 1.0]` based on mean ep_len thresholds 50/100 steps |
| Reward ramps | None (flat weights) | stopball: 100→175→250; footreach: 10→15→20; softstop: 50→75→100 at 0/2M/4M steps |
| `torque_limits` | −3.0 (flat) | CLAUDE.md says −3→−6→−9 but **curriculum entry is missing from `goalkeeper_env_cfg.py`** |

---

## 5. RSI (Random State Initialization)

### G1 (source: `legged_robot.py`)

**`_reset_dofs`:**
```python
if self.cfg.domain_rand.continue_keep and torch.rand(1).item() > 0.2:
    self.dof_pos[env_ids] = self.dof_pos[rand_src]   # 80%: copy live DOF positions
else:
    self.dof_pos[env_ids] = standpos * rand_scale + rand_offset  # 20%: perturbed default pose
self.dof_vel[env_ids] = 0.  # velocities ALWAYS zero
```

**`_reset_root_states`:**
```python
self.root_states[env_ids] = self.base_init_state           # standing pose
self.root_states[env_ids, :3] += self.env_origins[env_ids] # at env origin
self.root_states[env_ids, 7:13] = rand_float(-0.3, 0.3)   # small random root velocity
```

### SGKObsHis (source: `events.py`)

**80% live-copy:**
- `joint_pos_out[mask] = robot.data.joint_pos[src_ids]` — DOF positions from random live env
- `joint_vel_out` stays zero
- Root: `positions_out = env_origins[env_ids]` (goal line), identity quat + `±0.1 rad` yaw jitter
- Root velocity: **zero** (G1 uses `±0.3 m/s` random)

**20% NPZ motion frames:**
- Full frame: `joint_pos`, `joint_vel`, `root_pos[:, 2]` (Z only), `root_quat`, `root_lin_vel`, `root_ang_vel`
- Root XY always from `env_origins` — NPZ world-space XY discarded

| Property | G1 | SGKObsHis | Status |
|---|---|---|---|
| 80% DOF copy | joint positions only | joint positions only | ✅ |
| 80% DOF velocity | zero | zero | ✅ |
| 80% root | always home + `±0.3 m/s` random velocity | home + `±0.1 rad` yaw jitter, **zero velocity** | ⚠ SGKObsHis drops root velocity noise |
| 20% path | perturbed standing pose | NPZ stepping motion frame | ⚠ Intentional — NPZ provides stepping diversity; G1 stays near standing |

---

## 6. Ball Spawn

| Parameter | G1 | SGKObsHis (`reset_ball_rolling`) | Status |
|---|---|---|---|
| `x_start` range | `[3.0, 5.0]` m | `(1.5, 2.5)` m | ⚠ G1 spawns further (3–5 m); T1 closer (1.5–2.5 m) |
| `z_start` | Region-dependent: `0.1–0.3 m` (step), `1.2–1.6 m` (jump) | Fixed `floor_z + 0.12 m` (ground-level) | ⚠ Intentional — feet-only, foot-height ball |
| `speed` | `t_flight ∈ [0.4, 1.0]` s (implicit speed) | `(2.0, 3.5)` m/s explicit | ⚠ Different parameterisation |
| `y_end` routing | 6 disc groups with per-group command ranges | Left disc → `y_end ∈ [0, hard]`; right → `[−hard, 0]` | ⚠ 2 vs 6 groups; same principle |
| Center shots | All 6 groups span positive and negative Y | **Never** — each group sees one-sided spawns only (`y_end` never crosses zero) | ❌ SGKObsHis has no center-shot training |
| Curriculum mechanism | `curriculumupdate` expands command ranges `× 0.3` per update | `_ball_difficulty ∈ [0,1]` lerps easy→hard ranges | ⚠ Different mechanism, same intent |
| `aerial` / `reset_ball_local_frame` | Jump-height spawns active for jump disc groups | `reset_ball_local_frame` defined but **never hooked into events** | ❌ Aerial ball path is dead code |

---

## 7. AMP / Discriminators

| Property | G1 | SGKObsHis | Status |
|---|---|---|---|
| Number of discs | **6**: lefthand, righthand, leftjump, rightjump, leftstep, rightstep | **2**: leftstep, rightstep | ⚠ Intentional — feet-only, no hand/jump |
| Motion files (left) | `lefthand_*.pkl`, `leftjump_*.pkl`, `leftstep_*.pkl` | 4 left-step NPZ files | ⚠ Stepping motions only |
| Motion files (right) | `righthand_*.pkl`, `rightjump_*.pkl`, `rightstep_*.pkl` | 3 right-step NPZ files | ⚠ Same |
| AMP obs | `dof_pos` only | `joint_pos_abs` only | ✅ |
| Disc architecture | `trunk=[512→256]` with **SpectralNorm** + `amp_linear(256→1)` | `trunk=[512→256]` **plain Linear** + `amp_linear(256→1)` | ⚠ SpectralNorm removed; gradient penalty only |
| Weight init | `uniform(−1, 1)`, zero bias | Identical | ✅ |
| Expert target | +1 | +1 | ✅ |
| Policy target | −1 | −1 | ✅ |
| Grad pen formula | `λ_total = 5.0 × 0.1 = 0.5` | Identical | ✅ |
| Normalizer type | `Normalizer` (running mean/std from utils.py) | `EmpiricalNormalization` (custom, same principle) | ✅ Equivalent |

---

## 8. Terminations

| Condition | G1 | SGKObsHis | Status |
|---|---|---|---|
| `time_out` | episode > `3 s` | episode > `4.0 s` (train) / `10.0 s` (play) | ⚠ 1 s longer in SGKObsHis |
| `bad_orientation` | `‖projected_gravity[:2]‖ > 0.8` | `limit_angle = 1.0 rad` → `sin(1.0) ≈ 0.84` | ✅ Approximately equivalent |
| knee / base height | knee Z < `0.10 m` (body link) | base height < `0.4 m` | ⚠ Different metric; SGKObsHis catches falls earlier via base height |
| `sharpforce` | mean foot force > `1500 N` | mean foot force > `1500 N` | ✅ |
| `ball_exit` | ❌ not present | ball_x_local < `−0.5 m` | ⚠ SGKObsHis addition |
| `post_save_timeout` | ❌ not present | 80 steps after save | ⚠ SGKObsHis addition |

---

## 9. Robot

| Property | G1 | SGKObsHis | Status |
|---|---|---|---|
| Model | Unitree G1 | Booster T1 headless (21-DOF) | ⚠ Intentional |
| DOF count | `29` | `21` | ⚠ Intentional |
| Simulator | Isaac Gym (PhysX) | MuJoCo-Warp (mjlab) | ⚠ Intentional |
| `action_scale` | uniform `0.25` | per-joint `0.25 × effort / stiffness` | ⚠ SGKObsHis normalises by joint stiffness |

---

## Issues Requiring Action

### ❌ Bugs / undocumented divergences to fix

| # | Issue | Location |
|---|---|---|
| B1 | **`num_learning_epochs=5`, `num_mini_batches=4`** vs G1's `1×1` — SGKObsHis does 20 gradient steps per rollout where G1 does 1. Not documented as intentional. May cause policy over-fitting to each rollout. | `goalkeeper_amp_ppo_cfg.py` |
| B2 | **`gamma=0.99`** vs G1's `0.998` — shorter horizon; not documented. May reduce value of distant save rewards. | `goalkeeper_amp_ppo_cfg.py` |
| B3 | **`torque_limits` curriculum missing** — CLAUDE.md documents −3→−6→−9 ramp but no `torque_limits_curriculum` entry is wired in `goalkeeper_env_cfg.py`. Currently flat −3.0. | `goalkeeper_env_cfg.py` |
| B4 | **No center-shot training** — `y_end` routing means neither disc ever sees a ball aimed at the goal center. Robot never trained to defend a straight shot. | `events.py reset_ball_rolling` |
| B5 | **`reset_ball_local_frame` (aerial ball) dead code** — defined with full curriculum but never hooked into any event in `goalkeeper_env_cfg.py`. | `goalkeeper_env_cfg.py` |
| B6 | **`success` reward missing** — G1 gives a +5.0 one-time bonus when the effector comes within 0.15 m of the ball. No foot-distance equivalent in SGKObsHis (closest is `footreach`, but that's a continuous reward, not a proximity bonus). | `rewards.py` |

### ⚠ Intentional divergences (documented)

| # | Divergence | Justification |
|---|---|---|
| D1 | `num_regions` 6 → 2 | Feet-only: no hand or jump motions |
| D2 | `num_steps_per_env` 100 → 50 | GPU memory (32 GB vs 8 GB) |
| D3 | stopball threshold 2.0 → 1.0 m/s | MuJoCo soft contacts vs PhysX |
| D4 | 20% NPZ stepping frames vs perturbed standing | Stepping diversity from motion capture |
| D5 | SpectralNorm removed from disc | Gradient penalty alone sufficient |
| D6 | `postorientation` always active (not behind-gated) | AMP can't push root upright; gating removed only upright signal during ball approach |
| D7 | `schedule` fixed → adaptive | Quality improvement |
| D8 | Ball spawn height foot-level | Feet-only goalkeeping |
| D9 | `ang_vel_z` −2.0 added | Prevents yaw-spin local optimum |
| D10 | `penalize_self_collision` −50.0 added | MuJoCo has no implicit self-collision prevention |
| D11 | `action_rate_l2` −0.3 added | Additional first-order smoothness signal |

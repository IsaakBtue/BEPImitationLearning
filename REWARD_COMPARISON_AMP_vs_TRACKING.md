# Reward Comparison: AMP (G1 / IsaacGym) vs Motion Tracking (Booster T1 / MJLab)

**Date:** 2026-05-19  
**Original:** `/home/robocup/IsaakB/BEPImitationLearning/Humanoid-Goalkeeper/` — Unitree G1, 29 DOF, IsaacGym + HIM-PPO + AMP  
**Port:** `/home/robocup/IsaakB/BEPImitationLearning/my_mjlab_project_booster_t1/` — Booster T1, 23 DOF, MJLab + RSL-RL PPO + explicit motion tracking

---

## 1. Reward Functions — Side-by-Side

All G1 scales are multiplied by `dt = decimation × sim_dt = 4 × 0.005 = 0.02 s` at runtime (G1 `_prepare_reward_function`). T1 uses the same dt (50 Hz policy). Effective per-step weights are identical unless noted.

### 1a. Task Rewards

| Reward | G1 formula | G1 weight | T1 formula | T1 weight | Match? | Issues |
|--------|-----------|-----------|-----------|-----------|--------|--------|
| `eereach` | Sigmoid `1 - 1/(1+exp(-σ*(dist-reach_th)))` on dist to `end_target` (predicted intercept point), multiplied by velocity modulation `vel_sigma` per region, gated by upright | 10.0 | Sigmoid `1 - 1/(1+exp(-5*(dist-0.2)))` on dist from hand to **current** ball position, gated by upright | 10.0 | **Partial** | G1 rewards reaching the predicted goal-line intercept (`end_target`); T1 rewards reaching the live ball position. For fast balls at y=3m, the intercept (x=-0.5, z=0.6) can be far from current ball position. G1's approach is anticipatory; T1's is reactive. Also: G1's `curriculumsigma` grows from 5.0; T1 hardcodes σ=5. G1 multiplies by region-specific velocity bonus; T1 splits this into a separate term. |
| `eereach_velmod` | Embedded in `_reward_eereach` as `vel_sigma`: region 0/1 = lateral Y body vel × 3; region 2/3 = vertical vel × (3+3×curriculum); region 4/5 = lateral | — | `eereach_sigmoid × upright × (nu_lateral + nu_vertical)` where lateral = `sign(ball_rel_x) × lin_vel_w_x` clamped [0,3] × 3; vertical = `lin_vel_w_z` × 3 × is_high_ball | 10.0 | **New in T1** | T1 splits the velocity bonus out as a separate weighted term (weight 10.0). G1 integrates it multiplicatively inside `eereach`. Direction is computed dynamically from `sign(ball_rel_x)`. Architecturally different but functionally equivalent intent. |
| `success` / `hand_proximity_strict` | `(stop_flag + 1.0) × (dist < strict_th=0.15)` | 5.0 | `(1.0 + stopped.float()) × (dist < strict_th=0.15)` | 5.0 | **Match** | Identical logic. Returns 1 pre-save, 2 post-save. |
| `stopball` | `1.0 × (stop_flag==0) × (ball_vx_delta > 2.0) & (ball_x > origin_x)` — fires once per episode | 100.0 (+ curriculum) | `fired.float()` where `fired = (delta_vy > 2.0) & (ball_y_local > 0) & ~_sb_flag` | 100.0 (curriculum → 150 → 200) | **Match** | G1 checks X-axis speed delta; T1 checks Y-axis speed delta — correct for respective coordinate systems. Both compare against episode-start velocity, not per-step delta. G1 curriculum is continuous via `curriculumupdate`; T1 uses discrete iteration stages. |

### 1b. Stability / Balance Rewards

| Reward | G1 formula | G1 weight | T1 formula | T1 weight | Match? | Issues |
|--------|-----------|-----------|-----------|-----------|--------|--------|
| `stayonline` | `clip(abs(torso_x - origin_x), 0.2, 1.2) - 0.2` — X deviation from goal line | -2.0 | `clip(abs(robot_y_local), 0.2, 1.2) - 0.2` — Y deviation from goal line | -2.0 | **Match** | Correctly adapted for 90° rotation: G1 uses X axis, T1 uses Y axis. |
| `noretreat` | `-1 × clip(base_lin_vel_body_x, -1, 0)` — penalises backing away from goal | -2.0 | `-clip(fwd_vel_b, -1, 0)` where `fwd_vel_b = lin_vel_b[:,0]` (body forward = +Y world) | -2.0 | **Match** | Body-frame forward velocity correctly handles yaw during dive. |
| `feetorientation` | `exp(-5 × (left_grav_xy² + right_grav_xy²))` — gravity projected onto foot frame | +3.0 | `exp(-5 × sum(gravity_foot[...,2]²))` over both feet | +3.0 | **Match** | Identical formula; G1 uses sigma=5 hardcoded; T1 uses `sigma=5.0` parameter. |
| `postorientation` | `exp(-3 × proj_grav_xy_norm²) × behind` | +3.0 | `exp(-3.0 × err) × behind.float()` | +3.0 | **Match** | Identical. |
| `postangvel` | `exp(-3 × ang_vel_xy²) × behind` | +3.0 | `exp(-3.0 × err) × behind.float()` with `ang_vel[:,:2]` | +3.0 | **Match** | Identical. |
| `postlinvel` | `exp(-3 × base_lin_vel_x²) × behind` | +1.0 | `exp(-3.0 × lin_vel_b[:,0]²) × behind.float()` | +1.0 | **Match** | Both use body-frame forward velocity. Identical. |
| `postupperdofpos` | `exp(-1 × sum_sq(dof_pos[elbow+wrist] - default)) × behind` — 8 joints (4 elbow + 4 wrist, no shoulders) | +1.0 | `exp(-1.0 × sum_sq(arm_joint_pos - default)) × behind` — 8 joints (4 shoulder + 4 elbow, no wrist) | +1.0 | **Minor difference** | G1 covers elbow+wrist; T1 covers shoulder+elbow. T1 includes shoulders in recovery reward (larger range joints); G1 does not. T1 has no wrist joints to include. |
| `postwaistdofpos` | `exp(-3 × sum_sq(waist_3joints - default)) × behind` | +1.0 | `exp(-3.0 × sum_sq(waist_1joint - default)) × behind` | +1.0 | **Match** | Formula identical; T1 has 1 waist joint vs G1's 3 (yaw/roll/pitch). |

### 1c. Feet / Ground Contact Rewards

| Reward | G1 formula | G1 weight | T1 formula | T1 weight | Match? | Issues |
|--------|-----------|-----------|-----------|-----------|--------|--------|
| `successland` | `(in_air_reward×1 + landing×5 + one_foot_punish×-1) × jump_regions_only (2,3)` — tracks has_in_air flag, uses actual contact forces | +4.0 | `(foot_z - env_z < 0.05).all() × behind.float()` — both feet low, fires after ball passes, **all** episodes | +4.0 | **Divergence** | G1: sophisticated jump-detect-and-land, regions 2+3 only, uses contact force sensor, returns up to 6.0 per step. T1: simplified height proxy, fires for all trajectories (including non-jump low-ball saves), max return 1.0. Effective reward differs substantially. |
| `feet_slippage` | `exp(-10 × sum(norm(foot_vel_3d) × (contact_force > 1N)))` | +3.0 | `exp(-10.0 × sum(foot_speed × (foot_z < 0.05)))` | +3.0 | **Match** | Formula identical; G1 uses force > 1 N for contact detection, T1 uses height proxy. |
| `penalize_sharpcontact` | `(mean(norm(foot_contact_force)) > 1000N) × 1.0` — force spike on feet | -100.0 | `(trunk_z - env_z < 0.35).float()` — trunk height collapse proxy | -100.0 | **Divergence** | G1 detects force spikes; T1 detects trunk collapse. T1 proxy fires for any fall, misses rapid contact spikes, and also fires for slow collapses G1 would not penalize. |
| `penalize_kneeheight` | `(min(knee_body_z) < 0.15) × 1.0` | -100.0 | `(shank_z - env_z < 0.12).any().float()` | -100.0 | **Minor difference** | G1 threshold 0.15 m; T1 threshold 0.12 m on shank bodies (proxy for knee). Slightly more lenient. |

### 1d. Regularization Rewards

| Reward | G1 formula | G1 weight | T1 formula | T1 weight | Match? | Issues |
|--------|-----------|-----------|-----------|-----------|--------|--------|
| `ang_vel_xy` | `sum(ang_vel_b[:,:2]²)` | -0.1 | `sum(ang_vel_b[:,:2]²)` | -0.1 | **Match** | Identical. |
| `dof_acc` | `sum((last_dof_vel - dof_vel)² / dt²)` | -2.5e-7 | `joint_acc_l2` (mjlab builtin) | -2.5e-7 | **Match** | Both compute squared joint acceleration. |
| `smoothness` | `sum((actions - 2×last_actions + last_last_actions)²)` — second-order finite diff on actions | -0.1 | `action_acc_l2` — second-order action smoothness | -0.1 | **Match** | Equivalent second-order jerk penalty. |
| `torques` | `sum((torques / p_gains)²)` — normalized by per-joint kp | -1e-5 | `sum((torques × kp_inv)²)` using `_T1_KP_MAP` | -1e-5 | **Match** | Same formula. T1 kp values differ from G1 (e.g. knee: 200 vs 300) so absolute values differ, but normalization intent is identical. |
| `dof_vel` | `sum(dof_vel²)` | -5e-4 | `joint_vel_l2` | -5e-4 | **Match** | Identical. |
| `dof_pos_limits` | `sum(clip(dof_pos - soft_upper, min=0) + clip(soft_lower - dof_pos, min=0))`, soft=0.9× | -3.0 | `joint_pos_limits` (mjlab builtin, soft_factor=0.9 per `T1_ARTICULATION`) | -3.0 | **Match** | Identical formula. Limits read from URDF/MJCF respectively. |
| `dof_vel_limits` | `sum(clip(abs(dof_vel) - urdf_vel_limit×0.9, min=0))` — per-joint URDF limits | -2.0 | `sum(clip(abs(joint_vel) - 20.0×0.9, min=0))` — universal 20 rad/s cap | -2.0 | **Divergence** | G1 uses URDF-specified per-joint velocity limits. T1 uses a universal 20 rad/s cap (conservative). For slow high-torque joints at their real velocity limit, T1's penalty may never fire, making this term ineffective. |
| `torque_limits` | `sum(clip(abs(torques) - torque_limit×0.95, min=0))` | -3.0 | `sum(clip(abs(qfrc_actuator) - effort_limit×0.95, min=0))` from `_T1_EFFORT_MAP` | -3.0 | **Match** | T1 uses per-joint effort limits from `_T1_EFFORT_MAP` with correct values. |
| `deviation_waist_pitch_joint` | `sum(sq(dof_pos - default)[:, waist_pitch_idx])` — G1 waist_pitch only (index 2 of 3 waist joints) | -0.001 | `sum(sq(waist_joint_pos - default_waist))` — single T1 Waist joint | -0.001 | **Match** | T1 has one combined Waist joint; G1 penalized only the pitch component. Functionally equivalent. |

### 1e. Motion Tracking Rewards (T1-specific — replaces AMP discriminator)

These six rewards are the fundamental architectural replacement for AMP. G1 has no equivalent explicit reward terms; AMP handles style implicitly.

| Reward | Formula (mjlab builtin) | T1 weight | Purpose |
|--------|------------------------|-----------|---------|
| `motion_global_root_pos` | L2/exp error: root pos vs reference | 10.0 (from 0.5) | Force global sideways dive position |
| `motion_global_root_ori` | Orientation error vs reference | 5.0 (from 0.5) | Force dive rotation |
| `motion_body_pos` | Sum of body position errors, 14 bodies | 10.0 (from 1.0) | Force full-body pose match |
| `motion_body_ori` | Sum of body orientation errors | 3.0 (from 1.0) | Body orientation tracking |
| `motion_body_lin_vel` | Sum of linear velocity errors per body | 3.0 (from 1.0) | Force motion dynamics |
| `motion_body_ang_vel` | Sum of angular velocity errors per body | 3.0 (from 1.0) | Force motion dynamics |

**G1 AMP equivalent:** Discriminator produces a reward in `[0, 1]` per step via `predict_reward` (perturbation-smoothed, min over 20 noise samples). Blended as `total = amp_reward × 0.4 + raw_reward × 0.6`. The discriminator is trained adversarially with gradient penalty (λ=5, scale=0.1) on 6 motion files. AMP state = 58 dims (29 DOF pos at t concatenated with t+1).

---

## 2. Critical Reward Formula Differences

### 2.1 eereach — Anticipatory (G1) vs Reactive (T1)

**G1 `_reward_eereach`:**
```python
# self.dist = distance from hand to END_TARGET (predicted ball intercept at goal line)
# end_target updated when ball is 0.1-0.5m from goal: ball_pos extrapolated to x=0.1
taskrew = 1 - 1/(1 + exp(-curriculumsigma × (self.dist - reach_th)))
taskrew *= vel_sigma   # region-specific: lateral Y-vel for side saves, Z-vel for jumps
taskrew[phase1] = phase1_rew[phase1]  # early phase: linear pos reward, not sigmoid
return taskrew × upright_gate
```

**T1 `eereach`:**
```python
dist = norm(hand_pos_w - ball_pos_w)  # distance to CURRENT ball, not intercept
rew = 1 - 1/(1 + exp(-5 × (min_dist - 0.2)))
return rew × upright
```

G1 rewards anticipatory positioning at the predicted goal-line crossing. T1 rewards chasing the live ball. For a ball at y=3m aimed at x=-0.5, z=0.6: G1 gives reward for hand at (-0.5, 0, 0.6) immediately; T1 gives reward for hand at (x_ball, y_ball=3, z_ball) which is far from the intercept. As the ball approaches, T1's gradient converges on the correct position. This difference means T1 may learn slower for fast balls but may also generalize better to varied speeds.

### 2.2 successland — Semantics Differ Significantly

**G1:** Fires only for jump regions (2+3), tracks actual air-time via `has_in_air` flag, uses contact force sensor, returns up to 6.0 (5.0 landing + 1.0 in-air). Teaches jump-and-land.

**T1:** Fires for all episodes, uses foot height proxy (<5 cm), returns 1.0 when both feet are down after ball passes. Teaches feet-on-ground post-save. For a lefthand dive (no jump), this may actually be the correct behavior — reward the robot for not being airborne after deflecting the ball.

### 2.3 penalize_sharpcontact — Different Trigger

**G1:** `(mean(foot_contact_force_norm) > 1000 N)` — fires on sudden foot force spikes (impact falls).  
**T1:** `(trunk_z - env_z < 0.35)` — fires when robot folds at waist (trunk too low). These measure different collapse modes. A robot could fall forward (trunk low) without high foot force, or vice versa. The T1 proxy is less precise but requires no contact force sensor.

---

## 3. Coordinate System / 90-Degree Rotation Audit

The T1 spawns with `rot=(0.7071068, 0, 0, 0.7071068)` = +90° yaw around Z → robot faces +Y world.

| Location | Check | Result |
|----------|-------|--------|
| `stayonline` | G1 uses X deviation; T1 uses Y deviation | **Correct** |
| `noretreat` | Both use body-frame forward velocity (body X) | **Correct** |
| `ball_pos_b` | Both rotate world ball position into body frame | **Correct** |
| `ball_vel_b` (critic) | T1 rotates world ball velocity into body frame | **Correct** |
| `stopball` condition | G1: `ball_x > origin_x`; T1: `ball_y_local > 0` | **Correct** |
| `_ball_is_behind` | G1: `ball_x < 0`; T1: `ball_y_local < 0` | **Correct** |
| `ball_intercept_pos_b` | Uses `ball_vy` (Y-component) for time calculation | **Correct** for +Y approach |
| `eereach_velmod` lateral | Rewards X-velocity toward ball's X position (robot slides in X for lateral intercept) | **Correct** |
| `convert_booster.py` XY rotation | `root_pos[:,0] = -xy[:,1]; root_pos[:,1] = xy[:,0]` + `quat_mul(q_90, root_rot)` | **Correct** |
| Motion `body_pos_w` in commands.py | Adds `env_origins` offset; rotation already baked into npz | **Correct** |
| Ball spawn in `_reset_ball` | `y_start = uniform(3.0, 5.0)` (positive Y front); `vy = -y_start/t_flight` (negative, approaching) | **Correct** |

**No coordinate system errors found.** The 90° rotation is correctly applied in all rewards, observations, resets, and motion data.

---

## 4. Observations

### G1 Actor Observation Vector (per step, 96 dims)

```
ball_pos_local (3)              ← gated by initial_vanish (3-10 steps) + random_vanish (0-30 steps) + flying condition
base_ang_vel × 0.25 (3)
projected_gravity (3)
(dof_pos - default) × 1.0 (29)
dof_vel × 0.05 (29)
actions (29)
─── CUTOFF at 96 dims — below is privileged_obs_buf only ───
base_lin_vel × 2.0 (3)         ← critic only
end_regions / 3.0 (1)          ← critic only (which of 6 goal regions)
end_target_local (3)           ← critic only (predicted intercept in body frame)
ball_vel × 0.2 (3)             ← critic only
hand_pos_right (3)             ← critic only
hand_pos_left (3)              ← critic only
dist_hand_to_intercept (1)     ← critic only
```

**Actor total:** 96 × 10 steps history = **960 dims**  
**Critic total:** 96 dims (privileged, single step, no history)

### T1 Actor Observation Vector (per step, ~78 dims estimated)

```
base_ang_vel (3)               ← noise Unoise(±0.2)
projected_gravity (3)          ← noise Unoise(±0.05)
joint_pos (23)                 ← noise Unoise(±0.01), includes 2 head joints
joint_vel (23)                 ← noise Unoise(±1.5)
actions (23)
ball_pos_b (3)                 ← noise Unoise(±0.05); gated by vanish window (3-40 steps)
```

**Actor per-step:** ~78 dims × 10 steps history = **~780 dims**  
**Critic adds (per step):** `ball_vel_b (3)`, `left_hand_pos_b (3)`, `right_hand_pos_b (3)`, `ball_intercept_pos_b (3)`, `reach_dist_to_intercept (1)` = 13 extra dims  
**Critic total:** (~78 + 13) × 10 steps history = **~910 dims**

### Observation Comparison

| Observation | G1 actor | G1 critic | T1 actor | T1 critic | Match? | Notes |
|-------------|----------|-----------|----------|-----------|--------|-------|
| `ball_pos` (body frame, 3d, gated) | Yes | Yes | Yes | Yes | **Match** | Vanish window: G1 3-10 steps initial + 0-30 random; T1 3-10 + 0-30 (identical logic) |
| `ball_vel` (body frame, 3d) | No | Yes | No | Yes | **Match** | Correctly critic-only in both |
| `ang_vel` (body frame, 3d) | Yes (×0.25) | Yes | Yes (no explicit scale) | Yes | **Difference** | G1 applies ×0.25 fixed scale; T1 uses running normalization |
| `projected_gravity` (3d) | Yes | Yes | Yes | Yes | **Match** | |
| `dof_pos - default` (N dims) | Yes (N=29, ×1.0) | Yes | Yes (N=23, no explicit scale) | Yes | DOF count: 29 vs 23 |
| `dof_vel` (N dims) | Yes (N=29, ×0.05) | Yes | Yes (N=23, no scale) | Yes | **Difference** | G1 ×0.05 fixed; T1 running normalization |
| `actions` (N dims) | Yes (N=29) | Yes | Yes (N=23) | Yes | DOF count: 29 vs 23 |
| `base_lin_vel` (3d) | No | Yes | No | Yes | **Match** | Correctly privileged in both |
| `end_regions / 3.0` (1d) | No | Yes | **No** | **No** | **Missing in T1** | G1 critic sees which of 6 goal regions. T1 has single motion; no concept of regions. Not a bug — T1 scope reduction. |
| `end_target` / `ball_intercept_pos_b` (3d) | No | Yes | No | Yes | **Match** | Both give body-frame intercept position. G1 uses `end_target` (updated from trajectory); T1 computes from projectile formula. |
| `hand_pos_l`, `hand_pos_r` (3d each) | No | Yes | No | Yes | **Match** | |
| `dist_hand_to_intercept` (1d) | No | Yes | No | Yes | **Match** | |
| History stacking (actor) | 10 steps | 1 step | 10 steps | 10 steps | **Difference** | G1 critic sees 1-step privileged obs; T1 critic also gets 10-step history. T1 critic has more temporal context — may improve value estimates but increases input size. |

### Observation normalization difference

G1 uses fixed manual scales applied in `compute_observations()`: `ang_vel × 0.25`, `dof_vel × 0.05`, `ball_vel × 0.2`, `ball_pos × 0.3` (via flying mask). T1 uses `obs_normalization=True` in PPO config — running mean/std normalization applied per observation term at training time. T1's approach adapts automatically but needs a warmup period; G1's is deterministic.

---

## 5. Ball Trajectory and Reset

### G1 Ball Spawn

```python
# 6 ball regions partition 1020 envs into 6 groups of 170
# Ball x_start: uniform(3.0, 5.0) m in front (+X)
# Ball y_start (lateral): region-specific width range
# Ball z_start: uniform(maxh_min, maxh_max)
# Ball x_end: -uniform(0.1, 0.6) m behind goal
# Ball y_end, z_end: curriculum-expanded range
# t_flight: uniform(0.4, 1.0)
# Drag force: -0.5 × 1.225 × 0.47 × π×0.01 × speed × velocity (per step)
# Periodic perturbation: ±0.5 m/s every 0.5 s (25 steps)

# Curriculum: starts narrow, expands as curriculumupdate grows
# ranges_0 example: width=[0.2,1.2] → max [0.0,1.8]; height=[0.4,1.2] → max [0.3,1.5]
```

### T1 Ball Spawn (active: `commands.py _reset_ball`)

```python
# Ball approaches from +Y (correctly rotated)
y_start = uniform(3.0, 5.0)        # same as G1
x_end = uniform(-0.85, -0.2)       # constrained: left side, 70% reach cap, NO jump height
z_end = uniform(0.4, 0.85)         # constrained: 40-85 cm, no-jump height only
x_start = uniform(-1.8, 1.8)
z_start = uniform(0.3, 1.2)
t_flight = uniform(0.5, 1.0)       # similar to G1's 0.4-1.0
# Projectile formula with gravity (no explicit drag)
# Periodic perturbation: ±0.5 m/s every 25 steps — matches G1
```

**Note:** `resets.py reset_ball_training` exists with broader ranges (`x_end=[-1.2,1.2]`, `z_end=[0.1,1.6]`) but is NOT called during normal training. The `MultiMotionCommand._reset_ball` is the active reset function.

### Differences

| Property | G1 | T1 | Issue |
|----------|----|----|-------|
| Ball target lateral range | 6 regions covering ±1.8 m | Single range [-0.85, -0.2] (left only) | **Scope reduction** — T1 only trains left-hand saves |
| Ball target height range | Up to 1.8 m (jump saves in regions 2,3) | Max 0.85 m (no-jump cap) | **Deliberate** — T1 excludes jump trajectories |
| Ball curriculum | Starts narrow, expands with training | **No curriculum — fixed ranges from start** | T1 immediately trains on full range |
| Aerodynamic drag | Explicit per-step force application | Not applied (MuJoCo may or may not have fluid drag) | May cause ball physics differences at high speeds |

---

## 6. Domain Randomization

### Comparison Table

| Parameter | G1 | T1 | Mode G1 | Mode T1 | Match? |
|-----------|----|----|---------|---------|--------|
| Kp scaling | [0.8, 1.2]× | [0.8, 1.2]× | **Per episode** | Startup only | **Different mode** |
| Kd scaling | [0.8, 1.2]× | [0.8, 1.2]× | **Per episode** | Startup only | **Different mode** |
| Joint injection (torque noise) | ±1% torque limit per step | Not present | Per step | — | **Missing in T1** |
| Actuation offset (bias) | ±1% torque limit | Encoder bias ±0.015 rad | Per episode | Startup | Different mechanism |
| Payload mass | [-5, +10] kg | Not present | Per episode | — | **Missing in T1** |
| CoM displacement | ±0.1 m XYZ | Via `base_com` event | Per episode | Per episode | Match |
| Link mass scaling | [0.8, 1.2]× per link | Not present | Startup | — | **Missing in T1** |
| Friction | [0.1, 2.0] | [0.3, 1.2] (feet only) | Per episode | Startup | Narrower range, startup |
| Restitution | [0.0, 1.0] | Not present | Per episode | — | **Missing in T1** |
| Initial joint pos | ×[0.5,1.5] ± 0.1 rad | RSI from motion frame 0 ± 0.1 rad | Per episode | Per episode | Partial match — RSI is more physically grounded |
| Robot push | ±1.5 m/s XY | ±0.5 m/s XY, ±0.4 Z, ±0.52 roll/pitch, ±0.78 yaw | Every 15 s | Every 3-8 s | T1 adds rotation, more frequent, less lateral magnitude |
| Ball velocity perturbation | ±0.5 m/s every 0.5 s | ±0.5 m/s every 0.5 s | Per step | Per step | **Match** |
| Observation delay | 0-3 steps | 2-8 steps (40-160 ms) | Per env | Per env | T1 more conservative |
| Ball mass | None | ×[0.8, 1.2] | — | Per episode | **T1-specific** |
| Ball friction | None | [0.2, 0.8] | — | Startup | **T1-specific** |

---

## 7. Training Configuration (PPO Hyperparameters)

| Parameter | G1 | T1 | Match? |
|-----------|----|----|--------|
| `num_envs` | 1020 | 6144 | No (6× more in T1) |
| `num_steps_per_env` | 100 | 100 | **Match** |
| `gamma` | 0.998 | 0.998 | **Match** |
| `lam` | 0.95 | 0.95 | **Match** |
| `learning_rate` | 1e-3 (adaptive) | 1e-3 (adaptive) | **Match** |
| `clip_param` | 0.2 | 0.2 | **Match** |
| `entropy_coef` | 0.01 | 0.01 | **Match** |
| `num_learning_epochs` | 5 | 5 | **Match** |
| `num_mini_batches` | 4 | 4 | **Match** |
| `desired_kl` | 0.01 | 0.01 | **Match** |
| `max_grad_norm` | 1.0 | 1.0 | **Match** |
| `episode_length_s` | 3.0 s | 3.0 s | **Match** |
| `policy_dt` | 0.02 s (50 Hz) | 0.02 s (50 Hz) | **Match** |
| `actor_hidden_dims` | [512, 256, 256] | [512, 256, 128] | Minor difference |
| `critic_hidden_dims` | [512, 256, 256] | [512, 256, 128] | Minor difference |
| `action_scale` | 0.25 (uniform all joints) | Per-joint: 0.25×(effort/kp) ratio | **Difference** — T1 scales by stiffness ratio |
| `obs_normalization` | Fixed manual scales | Running mean/std (`obs_normalization=True`) | Architectural difference |
| AMP | amp_coef=0.4 (0.4×AMP + 0.6×raw) | N/A | N/A |
| `max_iterations` | 200,000 | 40,000 | T1 target 5× lower |

### Additional G1 losses (not in T1)

G1's `HIMPPO.update()` includes beyond PPO:
- `est_loss`: auxiliary ball position prediction head from actor, vs critic's ball_pos ground truth
- `region_loss`: CrossEntropy for which of 6 regions the ball targets
- `smooth_loss`: interpolated policy/value smoothness constraint between adjacent timesteps

None of these are in T1. The HIM (Hybrid Internal Model) architecture was specifically designed for the 6-region G1 task and is not replicated in T1's simpler RSL-RL setup.

---

## 8. Robot Model — DOF Count and PD Gains

### G1 (29 DOF)

| Joint group | Joints | kp (N·m/rad) | kd (N·m·s/rad) |
|------------|--------|-------------|----------------|
| hip_yaw/roll/pitch (×2) | 6 | 150 | 2 |
| knee (×2) | 2 | 300 | 4 |
| ankle_pitch/roll (×2) | 4 | 40 | 2 |
| waist yaw/roll/pitch | 3 | 150 | 2 |
| shoulder_pitch/roll/yaw (×2) | 6 | 150 | 2 |
| elbow (×2) | 2 | 150 | 2 |
| wrist roll/pitch/yaw (×2) | 6 | 20 | 0.5 |
| **Total** | **29** | | |

### Booster T1 (23 DOF)

| Joint group | Joints | kp | effort limit |
|------------|--------|----|--------------|
| Head yaw/pitch | 2 | 20 | 7 Nm |
| shoulder_pitch/roll + elbow_pitch/yaw (×2) | 8 | 15 | 18 Nm |
| waist | 1 | 80 | 30 Nm |
| hip_pitch (×2) | 2 | 120 | 45 Nm |
| hip_roll/yaw (×2) | 4 | 80 | 30 Nm |
| knee_pitch (×2) | 2 | 200 | 60 Nm |
| ankle_pitch (×2) | 2 | 50 | 20 Nm |
| ankle_roll (×2) | 2 | 40 | 15 Nm |
| **Total** | **23** | | |

**Key structural differences:**
- T1 has no wrist joints (G1 had 6); T1 has 2 head joints (G1 had none actuated)
- T1 arm kp = 15 vs G1 shoulder kp = 150: arms 10× less stiff
- T1 knee kp = 200 vs G1 knee kp = 300: T1 knee less stiff despite higher effort cap
- T1 uses per-joint action scale (effort/kp ratio): knee gets scale 0.075 (tight), arms get 0.3 (looser)

---

## 9. Architecture Summary: AMP vs Explicit Motion Tracking

### G1: HIM-PPO + AMP

```
Actor (960 dims → actions):     MLP 512-256-256, 10-step history
Critic (96 dims → value):       MLP 512-256-256, 1-step privileged
Auxiliary heads:                 Ball position estimator + Region classifier (6-class)

AMP Discriminators (6×):        lefthand, righthand, leftjump, rightjump, leftstep, rightstep
  Input:                         29 DOF pos at t + t+1 = 58 dims
  Architecture:                  MLP 512-256 + linear head
  Training:                      Adversarial (least-squares GAN + gradient penalty)
  Reward:                        predict_reward: min over 20 perturbations, exp smoothed → [0,1]
  Blend:                         total_reward = 0.4 × AMP_reward + 0.6 × task_reward

Motion files:                    6 pkl files (all goalkeeper directions)
DOF:                             29
```

### T1: RSL-RL PPO + Explicit Tracking

```
Actor (~780 dims → actions):    MLP 512-256-128, 10-step history
Critic (~910 dims → value):     MLP 512-256-128, 10-step history (actor + privileged)

No AMP discriminator.
Motion tracking via mjlab MotionCommand:
  Reference motion:              lefthand_t1.npz, 150 frames @ 50 Hz = 3.0 s
  Tracked bodies:                14 bodies (trunk, hips, shanks, ankles, waist, arms, hands)
  Rewards:                       6 explicit L2/exp terms on pos, ori, lin_vel, ang_vel
  Motion sampling:               always frame 0 (RSI at start of each episode)

Motion files:                    1 npz (single lefthand motion)
DOF:                             23
```

### AMP vs Motion Tracking Comparison

| Property | AMP (G1) | Motion Tracking (T1) |
|----------|----------|---------------------|
| Style signal | Implicit (learned discriminator logit) | Explicit (L2 error per body) |
| Style optimization | Through discriminator backward pass | Direct reward gradient |
| Reference data needed | Raw joint position sequences | FK-computed body pos/ori/vel |
| Discriminator stability | Can be unstable (mode collapse, gradient exploding) | No discriminator — always stable |
| Style generalization | Discriminator can generalize across similar motions | Strictly ties to single reference frame sequence |
| Gaming risk | Hard to game (adversarial) | Can match body positions without kinematically natural motion |
| Computation | 6 discriminator fwd + grad penalty | 6 L2 sums (trivial) |
| Motion coverage | Full goal coverage (all 6 directions/heights) | Left-hand saves only (x_end ∈ [-0.85,-0.2]) |
| RSI (Reference State Init) | Not used — random init from episode joint pos randomization | Yes — episodes start at motion frame 0 ± jitter |

---

## 10. Issues Ranked by Severity

### High Severity (likely affects learning quality or convergence)

**H1. `eereach` uses current ball position, not predicted intercept.**  
G1 always guides the hand toward `end_target` (where the ball will be at the goal line). T1 chases the live ball position. For balls at y=3 m, the live position and intercept can differ by >1 m. T1's `eereach` signal is reactive rather than anticipatory, which may slow convergence for short-flight-time balls.

**H2. No ball position curriculum.**  
G1 starts with a narrow target range and expands as training progresses. T1 uses fixed ranges from step 0. Early training sees the full distribution including the hardest trajectories (fast balls at extreme lateral/height), which may slow initial learning and cause the policy to settle in conservative local optima (stay centered, never dive).

**H3. Only one motion (lefthand) vs six in G1.**  
T1 cannot learn right-side saves, jump saves, or step saves. The policy is trained on a much narrower distribution. Ball target is constrained to x_end ∈ [-0.85, -0.2], z_end ∈ [0.4, 0.85]. This is deliberate scope reduction, but it means the policy never encounters the full goalkeeper task.

**H4. PD gain DR is startup-only in T1 vs per-episode in G1.**  
G1 resamples Kp/Kd factors each episode, so the policy is robust to any gain from [0.8, 1.2]× continuously throughout training. T1 randomizes at startup only — if the startup gains happen to be all-high or all-low, training may overfit to those gains. Per-episode DR is strongly recommended for sim2real transfer.

### Medium Severity (may cause instability or suboptimal behavior)

**M1. `successland` semantics differ significantly.**  
G1 rewards jump-and-land for high-ball regions only. T1 gives this reward for all episodes when feet are down after ball passes. For a lefthand dive, having feet down post-save is fine, but the height proxy (foot_z < 5 cm) may conflict with the dive motion which may briefly lift feet.

**M2. Missing per-step joint injection (torque noise).**  
G1 applies `±1%×torque_limit` noise per step to simulate actuator randomness. T1 has only startup encoder bias. This makes T1 potentially less robust to real-hardware torque disturbances.

**M3. `dof_vel_limits` uses universal 20 rad/s cap instead of per-joint limits.**  
For high-stiffness slow joints (hip_pitch, knee), the true velocity limit may be much lower than 20 rad/s, meaning T1's penalty never fires for those joints. This term is effectively disabled for leg joints.

**M4. Missing payload mass and link mass randomization.**  
G1 randomizes both total payload (±10 kg) and per-link mass (×[0.8, 1.2]). T1 only has CoM displacement. Mass DR is important for legged locomotion to handle carrying loads and manufacturing tolerance.

**M5. `postupperdofpos` joint set differs.**  
G1 uses elbow+wrist joints (8 joints) for recovery reward; T1 uses shoulder+elbow (8 joints). T1's version pushes shoulders back to neutral which is correct for T1's arm structure, but it's a different set than G1.

### Low Severity / Confirmed Correct

- 90° coordinate rotation: correctly applied throughout
- stopball Y-axis condition: correct for T1
- Ball perturbation ±0.5 m/s, every 25 steps: matches G1
- All core PPO hyperparameters (gamma, lam, entropy, learning_rate): match G1
- All post-save rewards (postorientation, postangvel, postlinvel, postwaistdofpos): match G1 formulas
- Ball vanish window implementation: matches G1 exactly (3-10 initial + 0-30 random steps)
- AMP reward removed: correctly replaced by motion tracking

---

## 11. Hypothesis: Convergence Comparison

### Factors favoring T1 learning faster than G1:

1. **6× more parallel environments** (6144 vs 1020) = more data per wall-clock iteration
2. **Single motion, simpler task** — one left-hand dive is easier to optimize than 6 motions
3. **Explicit tracking gradients** — motion tracking provides dense, stable gradients from episode start. AMP can be sparse and unstable in early training when the discriminator is not well-calibrated
4. **No discriminator instability** — AMP can suffer from mode collapse or exploding gradients; tracking is always stable
5. **Per-joint action scale** — prevents bang-bang control on T1's low-kp arms
6. **Modern framework** — mjlab/RSL-RL with built-in obs normalization and cleaner APIs

### Factors favoring G1 learning faster than T1:

1. **Anticipatory `eereach`** — G1 guides the hand toward the intercept point from episode start, giving a consistent gradient. T1 chases the live ball, which provides a weaker signal early when the ball is far away
2. **Ball curriculum** — G1 starts easy (small target range) and expands. T1 sees full difficulty from step 0
3. **Per-episode PD randomization** — G1 trains a more DR-robust policy from the start
4. **Region-specific velocity bonuses** — G1's `vel_sigma` gives targeted lateral/vertical velocity guidance per ball region; T1's `eereach_velmod` is more generic
5. **6 motion types** — AMP's multi-motion approach trains one policy that covers all goalkeeper actions simultaneously; T1's single motion may require multiple training runs or future extension

### Primary bottleneck prediction:

The most likely bottleneck for T1 training is the combination of reactive `eereach` (ball-chasing rather than intercept-targeting) and the absence of a ball curriculum. The policy may converge to a stable standing pose that accrues `motion_tracking` rewards by following the dive reference, but fails to generalize to the full ball target range because the reward gradient from `eereach`/`stopball` is weaker early. 

The explicit motion tracking weights (root_pos=10, body_pos=10) will dominate early training and force the dive shape — this is the primary advantage over a naive task-only baseline. Whether the policy learns to time the dive correctly with the ball depends critically on how the `eereach` signal guides approach timing. Adding an intercept-based `eereach` (like G1's `end_target`) and a ball position curriculum would likely be the two highest-impact improvements to close the convergence gap with G1.

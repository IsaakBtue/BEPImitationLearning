# SimpleGoalKeeper — Full Context

## What This Is

A **standalone, simplified goalkeeper training environment** for the Booster T1 humanoid robot, built as part of a BEP (Bachelor End Project) on imitation learning for humanoid robots. The goal is to train a policy that intercepts incoming balls using the robot's feet and deflects them back toward the field.

This is **Phase 1**: feet-only. No hand rewards, no arm observations.

## Why Standalone

The parent repo (`BEPImitationlearning`) contains a complex reference implementation (`Imitationlearningbooster`) that uses a 6-discriminator AMP runner, a full motion-tracking command system, and 25+ reward terms. SimpleGoalKeeper is a clean-slate rebuild using `beyondAMP` — a simplified AMP library that replaces all of that with a single `AMPEnvWrapper` + `AMPRunnerCfg`. No imports from any other project in the repo.

---

## Project Layout

```
SimpleGoalKeeper/
├── CLAUDE.md                          ← Phase 1 scope notes (AI instructions)
├── CONTEXT.md                         ← This file
├── pyproject.toml                     ← uv project (beyondAMP editable deps)
├── uv.lock
├── .gitignore
├── beyondAMP/                         ← git clone of Renforce-Dynamics/beyondAMP
│   └── source/
│       ├── beyondAMP/                 ← main AMP library
│       ├── rsl_rl_amp/                ← AMPOnPolicyRunner (fork of rsl_rl)
│       ├── amp_tasks/                 ← reference AMP tasks
│       └── amp_tasks_mjlab/           ← mjlab-backend AMP tasks
└── src/simple_goalkeeper/
    ├── robots/
    │   ├── t1_constants.py            ← actuators, HOME_KEYFRAME, T1_ACTION_SCALE_HEADLESS
    │   └── xmls/
    │       ├── t1.xml                 ← full Booster T1 (23 DOF)
    │       ├── t1_headless.xml        ← headless T1 (21 DOF, no head joints)
    │       ├── ball.xml               ← soccer ball (r=0.11m, m=0.42kg)
    │       └── assets/                ← 24 STL mesh files
    ├── mdp/
    │   ├── observations.py            ← ball_pos_b, ball_vel_b, foot positions
    │   ├── events.py                  ← ball reset in local +X, visibility state
    │   └── rewards.py                 ← 5 goalkeeper reward terms
    ├── motions/
    │   ├── raw/                       ← original PKL files (8 goalkeeper saves)
    │   └── data/                      ← converted NPZ files (beyondAMP format)
    ├── tasks/
    │   ├── goalkeeper_env_cfg.py      ← full env config
    │   ├── goalkeeper_amp_cfg.py      ← AMPRunnerCfg
    │   └── __init__.py                ← registers Mjlab-BeyondAMP-Goalkeeper-T1
    └── scripts/
        ├── pkl_to_npz.py             ← converts PKL → NPZ (with +X-facing fix)
        ├── train.py                   ← training entry point
        └── play.py                    ← play/eval entry point
```

---

## Robot

**Booster T1** — 21-DOF headless variant (head joints removed from policy):

| Joints | Count | Actuator |
|--------|-------|----------|
| Arm (shoulder pitch/roll, elbow pitch/yaw) × 2 | 8 | ARM_ACTUATOR |
| Waist + Hip Roll/Yaw × 2 | 5 | WAIST_HIP_ROLL_YAW |
| Hip Pitch × 2 | 2 | HIP_PITCH |
| Knee Pitch × 2 | 2 | KNEE |
| Ankle Pitch/Roll × 2 | 4 | ANKLE |
| **Total** | **21** | — |

Home pose: z=0.665m, legs slightly bent (hip_pitch=-0.2, knee=0.4, ankle=-0.2), arms counterbalanced.

Action scale: per-joint `0.25 * effort_limit / stiffness` (soft targets, not full range).

---

## Frame Convention

**All direction-sensitive rewards and observations use the robot's local frame:**

- **Origin:** robot base (Trunk) position in world
- **+X:** robot forward (ball approaches FROM here)
- **+Y:** robot left (lateral saves move in this direction)
- **+Z:** up

Ball always spawns in the robot's local +X direction, so the goalkeeper behavior is **world-orientation-independent** — the policy works regardless of which direction the robot faces in the world.

Key utility: `quat_apply(robot_quat_w, local_vec)` → world frame  
Inverse: `quat_apply(quat_inv(robot_quat_w), world_vec)` → robot body frame

---

## Motion Data

### Source
8 PKL files in `motions/raw/` — goalkeeper save motions captured for the Booster T1. Each file contains one motion clip:

| Key | Shape | Description |
|-----|-------|-------------|
| `fps` | int | capture frame rate |
| `root_pos` | (T, 3) | pelvis world XYZ |
| `root_rot` | (T, 4) | pelvis quaternion **xyzw** |
| `dof_pos` | (T, 23) | all 23 DOF joint angles |

### Conversion (`sgk_convert`)

The PKL→NPZ pipeline (`scripts/pkl_to_npz.py`) does:

1. **Resample** to 50 fps via linear interpolation
2. **-90° Z rotation** — PKL data faces +Y (initial yaw ≈ 90°); rotating -90° makes the robot face +X. Lateral saves then appear as Y displacement in world frame. Verified: all 8 motions have y_range > x_range after rotation.
3. **Yaw snap** — small residual yaw after rotation (±7°) is snapped to exactly 0° via a corrective quaternion.
4. **xyzw → wxyz** — MuJoCo quaternion convention
5. **Floor z-correction** — MuJoCo FK pass finds minimum foot Z, lifts root to put feet at ground level
6. **Head joint strip** — PKL has 23 DOF (includes head); headless XML has 21 DOF; skip first 2 (AAHead_yaw, Head_pitch)
7. **FK pass** — runs MuJoCo kinematics to extract body positions and orientations
8. **Finite-difference velocities** — joint velocities and body velocities via central differences

### NPZ Output Format

| Array | Shape | Description |
|-------|-------|-------------|
| `fps` | scalar | 50 (0-d array, `float(data["fps"])` works) |
| `joint_pos` | (T, 21) | absolute joint positions in headless order |
| `joint_vel` | (T, 21) | joint velocities via finite diff |
| `body_pos_w` | (T, nbody, 3) | body world positions from FK |
| `body_quat_w` | (T, nbody, 4) | body world quaternions from FK (wxyz) |
| `body_lin_vel_w` | (T, nbody, 3) | body linear velocities (finite diff) |
| `body_ang_vel_w` | (T, nbody, 3) | body angular velocities (log-map diff) |
| `joint_pos_absolute` | scalar 1 | flag: positions are absolute (not delta) |

---

## Environment Design

### Base

Built on `mjlab.tasks.velocity.velocity_env_cfg.make_velocity_env_cfg()` — the standard mjlab locomotion base. Then:
- Switch terrain to flat plane (no terrain generator)
- Add ball entity (from `robots/xmls/ball.xml`)
- Replace robot with headless T1
- Clear velocity commands and curriculum
- Replace velocity rewards with goalkeeper rewards
- Remove terrain/height sensors and corresponding obs
- Reduce DR to push_robot only

### Observations (810 dims total = 10 history × 81 per step)

| Term | Dim | Notes |
|------|-----|-------|
| `base_ang_vel` | 3 | robot angular velocity in body frame |
| `projected_gravity` | 3 | gravity vector in body frame |
| `joint_pos_rel` | 21 | joint angles relative to default pose |
| `joint_vel` | 21 | joint velocities |
| `actions` | 21 | previous action |
| `ball_pos_b` | 3 | ball position in robot body frame (visibility-masked) |
| `ball_vel_b` | 3 | ball velocity in robot body frame (visibility-masked) |
| `left_foot_pos_b` | 3 | left foot in robot body frame |
| `right_foot_pos_b` | 3 | right foot in robot body frame |

History: 10 frames (total 810). Training has delay 0–2 steps; play has no delay.

**AMP obs:** `joint_pos (21) + joint_vel (21) = 42` — single frame, beyondAMP basic group.

### Ball Visibility System (3-gate)

The policy must work even when the ball is not visible (hold a ready pose):

| Gate | Condition | Effect |
|------|-----------|--------|
| `initial_vanish` | `_catchstep < _startstep` | Ball hidden during warmup (~1s) after reset |
| `flying` | `x_b∈(0.05,3.4)`, `|y_b|<2.0`, `z<1.8`, approaching | Ball in view window |
| `random_vanish` | `_ball_visible_step > _vanish_step` | Random mid-flight disappearance |

Visible = `initial_vanish AND flying AND NOT random_vanish`. Ball obs zeroed when not visible.

`_catchstep` counts down from 50 each step (via `tick_catchstep` interval event). `_startstep = 50 - randint(3,11)` and `_vanish_step = randint(0,30)` are sampled at each reset.

### Ball Reset (`reset_ball_local_frame`)

```python
dist     ~ Uniform(2.0, 4.0) m   # along robot +X
lateral  ~ Uniform(-0.5, 0.5) m  # along robot +Y
height   ~ Uniform(0.1, 0.8) m   # Z above floor
speed    ~ Uniform(2.0, 6.0) m/s

ball_pos_w = robot_pos_w + quat_apply(robot_quat_w, [dist, lateral, height])
ball_vel_w = quat_apply(robot_quat_w, [-speed, 0, 0])  # flying toward robot
```

### Rewards (7 terms)

| Term | Weight | Formula | Purpose |
|------|--------|---------|---------|
| `foot_to_ball` | +3.0 | `exp(-||feet_mid_xy - ball_xy||² / 0.15²)` | Dense approach signal |
| `ball_vx_reduction` | +5.0 | `exp(-clamp(-vx_local,0,8)² / 4²)` | Stop incoming ball |
| `ball_positive_vx` | +10.0 | `clamp(vx_local / 5, 0, 1)` | Deflect ball back |
| `posture` | +1.0 | `exp(-mean(Δjoint²) / 0.25²)` | Stay upright |
| `ang_vel_xy` | -0.1 | `sum(ang_vel_b[:,:2]²)` | No rolling/pitching |
| `action_rate_l2` | -0.3 | `||action_t - action_{t-1}||²` | Smooth actions |
| `dof_vel` | -0.001 | `||joint_vel||²` | Penalise jerk |

All directional rewards (`ball_vx_reduction`, `ball_positive_vx`) use `dot(ball_vel_w, robot_x_axis_w)` — fully world-orientation-independent.

### Terminations

| Term | Condition |
|------|-----------|
| `time_out` | 3.0s (training), 10.0s (play) |
| `bad_orientation` | trunk tilt > 1.0 rad |
| `base_height` | trunk height < 0.4m |

### Domain Randomisation (minimal)

- `push_robot`: random velocity impulse (disabled in play)
- `reset_base`: yaw ±0.15 rad at reset

No COM offset, foot friction, or encoder bias in Phase 1.

---

## AMP Integration (beyondAMP)

`beyondAMP` replaces the complex 6-discriminator runner with a single wrapper:

```python
env = ManagerBasedRlEnv(cfg=goalkeeper_env_cfg())
env = AMPEnvWrapper(env, clip_actions=..., motion_dataset=MotionDatasetCfg(...))
runner = AMPOnPolicyRunner(env, asdict(goalkeeper_amp_runner_cfg()), ...)
runner.learn(num_learning_iterations=50_000)
```

**Key bodies for discriminator** (lower body only, Phase 1):
- `left_foot_link`, `right_foot_link`, `Shank_Left`, `Shank_Right`, `Waist`

**Anchor body:** `Trunk` — discriminator observations are anchored to the pelvis.

**AMP obs type:** `AMPObsBaiscTerms` — joint positions + joint velocities (basic, no body tracking).

**Hyperparameters:**
- `amp_reward_coef=0.5` — AMP reward weight
- `amp_task_reward_lerp=0.7` — task vs AMP blending (0.7 = 70% task reward)

---

## Training

```bash
cd /home/isaak/BEPImitationlearning/SimpleGoalKeeper

# 1. (Already done) Convert motions:
uv run sgk_convert  # uses motions/raw/ → motions/data/ by default

# 2. Train:
uv run sgk_train Mjlab-BeyondAMP-Goalkeeper-T1 --num-envs 4096

# 3. Sanity check (zero policy, no GPU):
uv run sgk_play Mjlab-BeyondAMP-Goalkeeper-T1 --agent zero --num-envs 1

# 4. Play trained policy:
uv run sgk_play Mjlab-BeyondAMP-Goalkeeper-T1 \
    --checkpoint-file logs/rsl_rl/simple_goalkeeper/<run>/model_500.pt
```

Logs go to `logs/rsl_rl/simple_goalkeeper/YYYY-MM-DD_HH-MM-SS_phase1/`.

---

## Key Design Decisions & Rationale

| Decision | Reason |
|----------|--------|
| Feet only (Phase 1) | Simpler reward shaping; AMP bodies exclude arms to allow arm freedom |
| Ball in robot local +X frame | World-orientation-independent goalkeeper — works regardless of field position |
| Flat terrain | Goalkeeper doesn't walk; no terrain needed |
| history_length=10 | Matches G1 upstream (num_actor_history=10); allows policy to track ball trajectory from observation history |
| Visibility warmup (~1s) | Gives robot time to reach ready stance before ball enters view; matches upstream training regime |
| random_vanish gate | Trains the policy to handle partial occlusion and act on memory |
| beyondAMP basic obs (joint_pos+vel) | Sufficient for natural locomotion prior; no complex body-tracking needed |
| No curriculum | Phase 1 simplicity; can add difficulty ramp in Phase 2 |
| -90° PKL rotation | PKL data captured with robot facing +Y; rotation bakes +X facing into NPZ |

---

## Phase 2 Ideas (not implemented)

- Add hand/arm rewards for high-ball saves
- Add ball difficulty curriculum (ramp spawn distance/speed)
- Add foot friction, COM offset, encoder bias DR
- Add sharp-force termination (foot contact force > 1500N)
- Longer episode length with ball re-spawn on contact
- Export policy to ONNX for deployment via `goalkeeper_deploy/`

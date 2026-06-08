# SimpleGoalKeeper — Phase 1 Design Spec

**Date:** 2026-06-08  
**Status:** Approved  
**Goal:** Standalone foot-based goalkeeper training environment for Booster T1, using beyondAMP for motion priors. Phase 1 focuses exclusively on feet — the robot must learn to intercept an incoming ball with its feet and deflect it back along the robot's local +X axis.

---

## 1. Project Structure

```
SimpleGoalKeeper/
├── beyondAMP/                         # git clone https://github.com/Renforce-Dynamics/beyondAMP.git
├── src/
│   └── simple_goalkeeper/
│       ├── __init__.py
│       ├── robots/
│       │   ├── __init__.py
│       │   ├── t1_constants.py        # copied verbatim from BoosterT1mjlab
│       │   └── xmls/                  # copied T1 XML + all STL assets
│       ├── mdp/
│       │   ├── __init__.py
│       │   ├── observations.py        # ball_pos_b, ball_vel_b (full visibility system)
│       │   ├── rewards.py             # all foot-based reward terms
│       │   └── events.py             # reset_ball_local_frame, reset_robot
│       ├── motions/
│       │   └── data/                  # .npz motion files (user-supplied)
│       ├── tasks/
│       │   ├── __init__.py
│       │   ├── goalkeeper_env_cfg.py  # env definition
│       │   └── goalkeeper_amp_cfg.py  # AMPRunnerCfg
│       └── scripts/
│           ├── train.py
│           └── play.py
├── pyproject.toml                     # uv project
├── uv.lock
└── CLAUDE.md
```

**Standalone constraint:** No runtime imports from `Imitationlearningbooster`, `BoosterT1mjlab`, or `HandWavingMotion`. The robot XML, actuator constants, and any shared utilities are copied in.

---

## 2. Dependencies (pyproject.toml)

```toml
[tool.uv.sources]
beyondAMP    = { path = "beyondAMP/source/beyondAMP",    editable = true }
rsl-rl-amp   = { path = "beyondAMP/source/rsl_rl_amp",   editable = true }
amp-tasks    = { path = "beyondAMP/source/amp_tasks",    editable = true }
```

Runtime deps: `mjlab`, `beyondAMP`, `rsl-rl-amp`, `mujoco>=3.8`, `mujoco-warp>=3.8`, `torch`, `tyro`.

---

## 3. Robot & Scene

- **Robot:** Booster T1, 23 DOF (headless variant — no head joints in action/obs space, 21 active DOF). Copied from `BoosterT1mjlab`.
- **Action scale:** `T1_ACTION_SCALE_HEADLESS` per-joint `0.25 * effort / stiffness`.
- **Contact sensor:** `self_collision` (Trunk subtree vs Trunk subtree) — same as upstream.
- **Ball entity:** `radius=0.11m`, `mass=0.42kg`, soccer friction (`0.4, 0.005, 0.0001`), stiff contact (`solref=[0.002, 0.0001]`).
- **`num_envs`:** 4096 (default; override with `--num-envs`).
- **`episode_length_s`:** 3.0s training, 10.0s play.

---

## 4. Frame Convention

The robot's local frame is defined as: **origin = midpoint between feet, +X = robot forward direction** (following BoosterT1mjlab convention). All ball observations and all reward computations that involve direction use the robot's body frame, not the world frame.

Key utility: `robot_x_axis_w(env)` — unit vector of robot's local +X in world coords, derived from `root_link_quat_w`.

---

## 5. Ball Mechanics

### 5a. Spawn (local +X frame)

On every reset, the ball is spawned in the robot's local frame:

```
dist     ~ Uniform(2.0, 4.0) m   # along robot +X
lateral  ~ Uniform(-0.5, 0.5) m  # along robot +Y
height   ~ Uniform(0.1, 0.8) m   # absolute Z above floor

ball_pos_w = robot_pos_w + quat_rotate(robot_quat_w, [dist, lateral, height])
ball_vel_w = quat_rotate(robot_quat_w, [-speed, 0, 0])   # speed ~ Uniform(2.0, 6.0) m/s
```

This makes the task robot-orientation-independent: the goalkeeper behavior transfers regardless of where the robot is facing in the world.

### 5b. Visibility System (full, ported from Imitationlearningbooster)

Three gates, all computed per-env:

| Gate | Logic | Effect |
|------|-------|--------|
| `initial_vanish` | `_catchstep < _startstep` (countdown ~1s) | Ball hidden at episode start |
| `flying` | `x_b ∈ (0.05, 3.4)` AND `|y_b| < 2.0` AND `z < 1.8` AND ball approaching | Ball in view window |
| `random_vanish` | `_ball_visible_step > _vanish_step` (sampled per episode) | Random disappearance mid-flight |

`ball_pos_b` and `ball_vel_b` are zeroed when `not (initial_vanish AND flying AND NOT random_vanish)`. Policy must learn to act on last-known position.

---

## 6. Observations

**Actor group** (history_length=10, delay 0–2 steps):

| Term | Dim | Notes |
|------|-----|-------|
| `base_ang_vel` | 3 | robot body angular velocity |
| `projected_gravity` | 3 | gravity in body frame |
| `joint_pos_rel` | 21 | relative to default pose |
| `joint_vel` | 21 | raw joint velocities |
| `actions` | 21 | previous action |
| `ball_pos_b` | 3 | ball in body frame (visibility-masked) |
| `ball_vel_b` | 3 | ball velocity in body frame (visibility-masked) |
| `left_foot_pos_b` | 3 | left foot in body frame |
| `right_foot_pos_b` | 3 | right foot in body frame |

**AMP group** (history_length=1, no noise, no corruption):
```python
cfg.observations["amp"] = amp_obs_basic_group()  # beyondAMP — joint_pos + joint_vel, single frame
```

**Critic group:** same as actor but no noise and no delay.

---

## 7. Reward Design (all in robot local frame)

### Primary task rewards

**`foot_to_ball`** (weight 3.0, dense)  
Gaussian reward on XY distance from foot midpoint to ball. Provides continuous dense signal from episode start.
```
dist = ||feet_midpoint_w[:2] - ball_pos_w[:2]||
reward = exp(-dist² / 0.15²)
```

**`ball_vx_reduction`** (weight 5.0, dense)  
Reward for neutralising the ball's incoming (negative) local-X velocity. Uses robot-frame projection. `max_speed = 8.0 m/s` (slightly above max spawn speed of 6.0 m/s).
```
vx_local = dot(ball_vel_w, robot_x_axis_w)   # negative when ball incoming
# Reward peaks when vx_local reaches 0 (ball stopped):
reward = exp(-clamp(-vx_local, 0, 8.0)² / 4.0²)
```

**`ball_positive_vx`** (weight 10.0, dense after contact)  
Primary success metric: ball deflected back toward the field (positive local X velocity).
```
vx_local = dot(ball_vel_w, robot_x_axis_w)
reward = clamp(vx_local / 5.0, 0.0, 1.0)   # normalised to [0, 1], saturates at 5 m/s
```

### Secondary shaping rewards

**`posture`** (weight 1.0)  
Gaussian on deviation from default joint pose — prevents bizarre contortions.
```
reward = exp(-mean(||joint_pos - joint_default||²) / 0.25²)
```

**`ang_vel_xy`** (weight -0.1)  
Penalise base rolling/pitching — keep the robot upright.

### Regularisation penalties

| Term | Weight | Formula |
|------|--------|---------|
| `action_rate_l2` | -0.3 | `||action_t - action_{t-1}||²` |
| `dof_vel` | -0.001 | `||joint_vel||²` |

**Total: 7 reward terms** vs 25+ in Imitationlearningbooster.

---

## 8. Terminations

| Name | Condition |
|------|-----------|
| `bad_orientation` | trunk tilt > 1.0 rad |
| `base_height` | trunk height < 0.4m |
| `time_out` | episode length exceeded |

No sharp-force termination in Phase 1 (no contact sensor on feet).

---

## 9. Domain Randomisation

**Minimal by design.** Only:
- `push_robot`: random velocity impulse (disabled in play mode)
- `reset_base`: small yaw variation on reset (±0.15 rad)

No COM offset, no encoder bias, no foot friction randomisation. These can be added in Phase 2.

---

## 10. AMP Integration (beyondAMP)

Training uses the standard beyondAMP pattern — no custom runner:

```python
# goalkeeper_amp_cfg.py
from beyondAMP.mjlab.rsl_rl import AMPRunnerCfg, AMPPPOAlgorithmCfg, RslRlPpoActorCriticCfg
from beyondAMP.motion.motion_dataset import MotionDatasetCfg

def goalkeeper_amp_runner_cfg() -> AMPRunnerCfg:
    return AMPRunnerCfg(
        num_steps_per_env=24,
        max_iterations=50_000,
        experiment_name="simple_goalkeeper",
        policy=RslRlPpoActorCriticCfg(
            actor_hidden_dims=[512, 256, 128],
            critic_hidden_dims=[512, 256, 128],
        ),
        algorithm=AMPPPOAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.005,
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=1.0e-3,
            schedule="adaptive",
            gamma=0.99, lam=0.95,
            desired_kl=0.01, max_grad_norm=1.0,
        ),
        amp_data=MotionDatasetCfg(
            motion_files=[...],     # foot/leg motions supplied by user
            body_names=["left_foot_link", "right_foot_link", "Waist", "Shank_Left", "Shank_Right"],
            amp_obs_terms=AMPObsBaiscTerms,
            anchor_name="Trunk",
        ),
        amp_reward_coef=0.5,
        amp_task_reward_lerp=0.7,
    )
```

The AMP discriminator sees only the lower-body motions (feet, shanks, waist) — it rewards natural leg movement without caring about arms.

---

## 11. Training Command

```bash
cd SimpleGoalKeeper
uv run python src/simple_goalkeeper/scripts/train.py --num-envs 4096
```

The `train.py` entry point mirrors `HandWavingMotion/scripts/train_beyondamp.py`: it imports `simple_goalkeeper.tasks` to register `Mjlab-BeyondAMP-Goalkeeper-T1`, then wraps the env with `AMPEnvWrapper` and runs `AMPOnPolicyRunner`.

---

## 12. CLAUDE.md Notes

The CLAUDE.md for SimpleGoalKeeper will document:
- Phase 1 scope: feet only, no hand rewards, no arm observation terms
- Frame convention: all directions in robot local +X frame
- beyondAMP location: `./beyondAMP/source/`
- Motion files: what format is expected (NPZ, joint order = T1 21-DOF headless)
- Reward intent: foot proximity → velocity neutralisation → positive-X deflection

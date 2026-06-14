# SimpleGoalKeeper — CLAUDE.md

## Phase 1 Scope

**Phase 1 focuses exclusively on foot-based goalkeeping.** The robot must intercept incoming balls using its feet only. There are no hand rewards, no arm-specific observations, and no hand-related AMP body names.

Hand rewards and arm observations are explicitly out of scope and should not be added until Phase 2 is started with a new design review.

## Project Purpose

Standalone, simplified goalkeeper training environment for the Booster T1 humanoid using:
- **mjlab** (MuJoCo-Warp RL framework)
- **beyondAMP** (simplified AMP integration — no custom 6-discriminator runner)
- **21-DOF headless T1** (head joints removed from action/observation space)

## Project Origin

`SimpleGoalKeeper` is a **foot-only** experimental track, distinct from `Humanoid-Goalkeeper` (the original paper) and `Imitationlearningbooster` (the T1 hand-catching port). Both of those use hands/arms; here the robot may only use its feet. AMP motion priors encourage natural bipedal motion while the task rewards focus entirely on foot-ball contact.

## Design Rule: Always Verify Against Humanoid-Goalkeeper First

**Before adding, changing, or removing ANY reward term, spawn parameter, observation, termination condition, curriculum stage, or training hyperparameter**, you MUST:

1. **Read the upstream G1 code** in `Humanoid-Goalkeeper/legged_gym/legged_gym/envs/` — specifically `base/legged_robot.py` (all reward functions, reset logic, behind/deflection conditions) and `g1/g1_29_config.py` (all weights, ranges, thresholds).
2. **Confirm the G1 equivalent exists** and quote the exact line/function. If it does not exist in G1, that is a red flag — explain why it's needed.
3. **Explicitly justify every divergence** — state WHY the upstream G1 value is wrong for this setup (MuJoCo vs PhysX, feet vs hands, mjlab API difference, etc.).
4. **Document it** in the "Divergences from G1 Upstream" table below.

This rule exists because G1 is the only proven working reference. Every undocumented divergence is a potential source of a local optimum or training failure. "It seems reasonable" is not a justification — G1 must be the baseline.

## Divergences from G1 Upstream

| Parameter | G1 value | SimpleGoalKeeper value | Justification |
|-----------|----------|------------------------|---------------|
| **Robot reset (RSI)** | Fixed standing pose | **Random frame from motion data** | **CRITICAL 2026-06-14:** Training without RSI caused robot to stand still—it never learned dynamics from mid-motion states. Now samples random motion file + random frame (0–T) and resets robot to that joint configuration. Exposes policy to varied body poses. |
| Ball spawn height | 0.15–0.4 m (chest-compatible) | **0.05–0.35 m (foot-to-shin)** | **CRITICAL 2026-06-14:** Ball spawn was too high (0.2–0.7 m), forcing upper-body contact. Lowered to foot/shin level for feet-only goalkeeping. Also lowered arrive height from 0.1–0.4 m to 0.05–0.25 m. |
| `ang_vel_xy` weight | -0.1 (roll+pitch only, `[:, :2]`) | -0.1 same | No change |
| `ang_vel_z` (yaw) | **Not penalized** (free) | **-0.5** | G1 yaws to extend arm reach. Feet have ~10 cm reach radius — yawing does not help. Observed training run 9000: robot learned to spin around (0,0). |
| `_ball_is_behind` threshold | `delta_vx > 2.0` (`legged_robot.py:1377`) | `delta_vx > 1.0` | MuJoCo soft contact produces smaller impulses than Isaac Gym PhysX. At 2.0 m/s threshold, slow-ball saves (ball at -1 → +0.5 m/s) never fire `stopball` or activate post-save rewards. Lowered to 1.0 m/s, same as ILB. Structure is identical: `(x<0) \| (delta_vx > threshold)`. |
| `ball_positive_vx` | Not in G1 | **Removed** | Was a SGK-only addition. Caused robot to chase ball after saving it (continuous reward never deactivates). G1/ILB rely on `stopball` + post-save recovery; no continuous vx reward needed. |
| Ball spawn frame | Robot-local | **Robot-local** (`reset_ball_local_frame`) | Same as G1. Spinning prevented by `ang_vel_z=-0.5` + `torque_limits` curriculum. |
| Ball spawn `y_start_range` | ±0.8 m (ILB) | **±1.0 m** | Widened from ±0.8 m to increase approach angle variety. Forces robot to handle balls from varied lateral positions during spawn. |
| Ball spawn `y_end_range` (goal target) | Motion-dependent, wide (G1) | **±1.5 m** | **CRITICAL FIX:** Widened from ±0.5 m to force lateral stepping. Robot width ~0.6 m at shoulders; ±0.5 m range allowed standing-still strategy (ball always hits center). New ±1.5 m range includes easy center shots and extreme edge shots requiring full lateral stepping to intercept with feet instead of passive torso deflection. |
| Effector type | Hands | Feet only | Phase 1 scope. |

## Frame Convention

Ball spawning uses **robot local +X frame** (`reset_ball_local_frame` in `mdp/events.py`).
The ball spawns at `(robot_pos + x_start, robot_pos + y_start)` in local frame (rotated by
robot yaw), aimed at `(-0.3, y_end)` local — 0.3 m behind the goal line so the ball retains
forward momentum at interception. Using local frame makes the policy orientation-invariant;
the yaw penalty (`ang_vel_z = -0.5`) and torque penalties prevent the degenerate spin strategy.

Key frame notes:
- `noretreat`: body-frame X velocity (correct even when robot yaws during a dive)
- `ball_positive_vx`: robot-local +X (consistent with local-frame ball spawn — a save means
  ball travels back in the direction it came from)
- `footreach`: world Y alignment (ball_y_w − robot_y_w)

## Key Files

| File | Purpose |
|------|---------|
| `src/simple_goalkeeper/robots/t1_constants.py` | T1 actuator configs, action scale, home keyframe |
| `src/simple_goalkeeper/robots/xmls/` | T1 headless XML + ball XML + STL assets |
| `src/simple_goalkeeper/mdp/observations.py` | ball_pos_b, ball_vel_b (visibility system), foot positions |
| `src/simple_goalkeeper/mdp/events.py` | reset_ball_local_frame, tick_catchstep |
| `src/simple_goalkeeper/mdp/rewards.py` | 5 goalkeeper reward terms (feet-only) |
| `src/simple_goalkeeper/tasks/goalkeeper_env_cfg.py` | Full env config |
| `src/simple_goalkeeper/tasks/goalkeeper_amp_cfg.py` | AMPRunnerCfg |
| `src/simple_goalkeeper/tasks/__init__.py` | Task registration |
| `src/simple_goalkeeper/motions/data/` | NPZ motion files (converted from PKL) |
| `src/simple_goalkeeper/scripts/train.py` | Training entry point |
| `src/simple_goalkeeper/scripts/play.py` | Play/evaluation entry point |
| `src/simple_goalkeeper/scripts/pkl_to_npz.py` | PKL→NPZ motion converter |

## beyondAMP Location

Cloned at `./beyondAMP/`. The four packages are installed as editable:
- `beyondAMP/source/beyondAMP` → `beyondAMP` package
- `beyondAMP/source/rsl_rl_amp` → `rsl-rl-amp` package
- `beyondAMP/source/amp_tasks` → `amp-tasks` package
- `beyondAMP/source/amp_tasks_mjlab` → `amp-tasks-mjlab` package

## Motion Files

NPZ format, 21-DOF headless T1 joint order. Expected arrays:
- `fps`: sampling rate
- `joint_pos` (T, 21): joint positions (absolute, matching T1 default pose reference)
- `joint_vel` (T, 21): joint velocities via finite differences
- `body_pos_w`, `body_quat_w`, `body_lin_vel_w`, `body_ang_vel_w`: body kinematics from FK

Convert PKL → NPZ:
```bash
uv run sgk_convert --input-dir /path/to/Motions --output-dir src/simple_goalkeeper/motions/data
```

## Reward Design

Phase 1 reward structure (ported from proven Imitationlearningbooster pattern):

| Term | Weight | Purpose |
|------|--------|---------|
| `stopball` | +100→250 (curriculum) | One-time bonus when ball is deflected (delta_vx > 1 m/s). Primary signal. |
| `footreach` | +10→20 (curriculum) | Phase1: lateral alignment. Phase2: sigmoid reach × vel_sigma (1–10×). Deactivates on deflection. |
| `stayonline` | -2.0 | Penalty for drifting away from goal line (X displacement) |
| `noretreat` | -2.0 | Penalty for retreating backward (negative body-frame X velocity) |
| `feetorientation` | +3.0 | Flat feet (gravity aligned with foot Z) |
| `postorientation` | +3.0 | Upright posture recovery, active only when ball is behind |
| `postangvel` | +3.0 | Low XY angular velocity reward, active only when ball is behind |
| `postlinvel` | +1.0 | Low forward velocity reward, active only when ball is behind |
| `postupperdofpos` | +1.0 | exp(-err) arm recovery to default, active only when ball is behind |
| `postwaistdofpos` | +1.0 | exp(-err) waist recovery to default, active only when ball is behind |
| `penalize_kneeheight` | -100.0 | Penalty when shank drops below 15 cm above floor (prevents kneeling) |
| `penalize_sharpcontact` | -100.0 | Binary penalty when mean foot contact force > 1000 N (requires `feet_contact` sensor) |
| `penalize_self_collision` | -50.0 | Binary penalty on any Trunk-subtree self-collision (requires `self_collision` sensor) |
| `feet_slippage` | +3.0 | exp(-10*contactvel) — rewards stable foot contact, penalises sliding (requires `feet_contact` sensor) |
| `dof_pos_limits` | -3.0 | Joint limit violation penalty |
| `dof_vel_limits` | -2.0 | Penalty for joint velocity > 10 rad/s (sum of squared excess) |
| `torque_limits` | -3.0→-9.0 (curriculum) | Per-joint torque limit violation; penalises hip-yaw spin torques |
| `ang_vel_xy` | -0.1 | Penalise rolling/pitching |
| `ang_vel_z` | -0.5 | Penalise yaw rotation — a goalkeeper should face the field |
| `deviation_waist_joint` | -0.001 | Waist joint regularisation (always active) |
| `torques` | -1e-5 | Normalized torque L2: sum((torque/kp)^2) — dimensionless across joints |
| `action_rate_l2` | -0.3 | Action smoothness |
| `action_acc_l2` | -0.1 | Action jerk penalty (second-order smoothness) |
| `dof_vel` | -5e-4 | Joint velocity regularisation |
| `dof_acc` | -2.5e-7 | Joint acceleration penalty (jerk reduction; matches ILB) |

**Terminations:** `time_out`, `bad_orientation` (>57°), `base_height` (<0.4 m), `ball_exit` (behind goal -0.5 m), `sharpforce` (>1500 N mean foot force).

**`_ball_is_behind` semantics:** `(ball_x < 0) | (delta_vx > 1.0)` — matches ILB exactly. Fires the moment stopball fires (deflection), deactivating `footreach` (no post-save chasing) and activating all post-save recovery rewards immediately.

**Removed (created stand-still or wrong local optimum):**
- `ball_positive_vx`: caused robot to chase ball after already saving it (continuous vx reward never turns off); `_ball_is_behind` + `stopball` are sufficient
- `successland`: with feet-only goalkeeping this became a ball-chasing reward; removed in favour of `stopball`
- `ball_vx_reduction`: peaked when ball stopped naturally — rewarded doing nothing
- `foot_to_ball` (std=0.15): zero gradient at 2–4 m spawn distance
- `posture`: AMP handles motion naturalness; posture+regularisation incentivised standing still

**Ball visibility:** `always_visible=True` during training so policy always has ball position.
Play mode re-enables the visibility gate (warmup + vanish) for partial observability.

## Training Commands

```bash
# Convert motions (once):
uv run sgk_convert --input-dir /home/isaak/BEPImitationlearning/Motions --output-dir src/simple_goalkeeper/motions/data

# Train:
uv run sgk_train Mjlab-BeyondAMP-Goalkeeper-T1 --num-envs 4096

# Play (zero policy sanity check):
uv run sgk_play Mjlab-BeyondAMP-Goalkeeper-T1 --agent zero --num-envs 1

# Play (trained checkpoint):
uv run sgk_play Mjlab-BeyondAMP-Goalkeeper-T1 --checkpoint-file logs/rsl_rl/simple_goalkeeper/<run>/model_500.pt
```

## Standalone Constraint

**No runtime imports from `Imitationlearningbooster`, `BoosterT1mjlab`, or `HandWavingMotion`.**
All needed assets and constants are copied into this folder.

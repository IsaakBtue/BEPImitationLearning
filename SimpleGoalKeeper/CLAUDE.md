# SimpleGoalKeeper — CLAUDE.md

## Phase 1 Scope

**Phase 1 focuses exclusively on foot-based goalkeeping.** The robot must intercept incoming balls using its feet only. There are no hand rewards, no arm-specific observations, and no hand-related AMP body names.

Hand rewards and arm observations are explicitly out of scope and should not be added until Phase 2 is started with a new design review.

## Project Purpose

Standalone, simplified goalkeeper training environment for the Booster T1 humanoid using:
- **mjlab** (MuJoCo-Warp RL framework)
- **beyondAMP** (simplified AMP integration — no custom 6-discriminator runner)
- **21-DOF headless T1** (head joints removed from action/observation space)

## Frame Convention

All reward and observation computations that involve direction use the **robot's local frame**:
- Origin: robot base (Trunk) position
- +X: robot forward direction (ball approaches from here)
- +Y: robot left
- +Z: up

Ball always spawns in the robot's local +X frame (`reset_ball_local_frame` in `mdp/events.py`), ensuring goalkeeper behavior is world-orientation-independent.

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
| `footreach` | +10.0 | Phase1: lateral alignment. Phase2: sigmoid reach × vel_sigma (1–10×) |
| `stopball` | +100.0 | One-time bonus when ball is deflected (delta_vx > 1 m/s). Primary signal. |
| `ball_positive_vx` | +10.0 | Continuous reward for sustained deflection back toward +X |
| `stayonline` | -2.0 | Penalty for drifting away from goal line (X displacement) |
| `noretreat` | -2.0 | Penalty for retreating backward (negative body-frame X velocity) |
| `feetorientation` | +3.0 | Flat feet (gravity aligned with foot Z) |
| `successland` | +4.0 | Dense reward: either foot within 12 cm of ball while ball is in front |
| `penalize_kneeheight` | -100.0 | Penalty when shank drops below 15 cm above floor (prevents kneeling) |
| `dof_vel_limits` | -2.0 | Penalty for joint velocity > 10 rad/s (sum of squared excess) |
| `postupperdofpos` | -1.0 | Arm deviation from default pose, active only after ball passes (recovery) |
| `postwaistdofpos` | -1.0 | Waist deviation from default pose, active only after ball passes (recovery) |
| `ang_vel_xy` | -0.1 | Penalise rolling/pitching |
| `deviation_waist_joint` | -0.001 | Waist joint regularisation (always active) |
| `dof_pos_limits` | -3.0 | Joint limit violation penalty |
| `action_rate_l2` | -0.3 | Action smoothness |
| `action_acc_l2` | -0.1 | Action jerk penalty (second-order smoothness) |
| `dof_vel` | -5e-4 | Joint velocity regularisation |

**Removed (created stand-still local optimum):**
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

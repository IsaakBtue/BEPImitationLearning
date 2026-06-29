# ONNX Export Report — model_8000

**Run:** `2026-06-29_11-18-52_phase1`  
**Checkpoint:** `model_8000.pt` (iteration 8,000)  
**Exported:** `2026-06-29_11-18-52_phase1.onnx`  
**Opset:** 18  
**Format:** ONNX with mjlab metadata  

---

## Training Snapshot at Iteration 8,000

| Metric | Value |
|---|---|
| Iterations completed at export | 8,000 |
| Total planned iterations | 50,000 |
| Run started | 2026-06-29 11:18 UTC+2 |

Run is ongoing. Checkpoint 8000 captured mid-training for evaluation.

---

## Model Architecture

| Property | Value |
|---|---|
| Network | MLP (ActorCritic) |
| Input (obs) | 710 |
| Output (actions) | 21 |
| Hidden dims | [512, 256, 128] |
| Activation | ELU |
| Noise std type | scalar |
| Init noise std | 1.0 |

## Observation Space — Actor (710 dims, history × 10)

| Term | Shape (per step) | History steps | Total |
|---|---|---|---|
| `base_ang_vel` | (3,) | 10 | 30 |
| `projected_gravity` | (3,) | 10 | 30 |
| `joint_pos_rel` | (21,) | 10 | 210 |
| `joint_vel` | (21,) | 10 | 210 |
| `actions` | (21,) | 10 | 210 |
| `ball_pos_b` | (2,) | 10 | 20 |
| **Total** | | | **710** |

Note: `ball_pos_b` is (2,) — XY only (Z masked during training: `always_visible=True`).

## Action Space (21 dims)

21 joint position targets (headless Booster T1, no head joints).  
Action scale: per-joint `effort / stiffness`.

---

## Training Hyperparameters (as saved in params/agent.yaml)

| Parameter | Value |
|---|---|
| Algorithm | AMPPPO |
| `num_steps_per_env` | 24 |
| `num_learning_epochs` | 5 |
| `num_mini_batches` | 4 |
| `learning_rate` | 1e-3 (adaptive, `desired_kl=0.01`) |
| `gamma` | 0.99 |
| `lam` | 0.95 |
| `entropy_coef` | 0.01 |
| `desired_kl` | 0.01 |
| `max_grad_norm` | 1.0 |
| `clip_param` | 0.2 |
| `amp_replay_buffer_size` | 250,000 |
| `amp_task_reward_lerp` | 0.5 |
| `amp_reward_coef` | 1.0 |
| `amp_discr_hidden_dims` | [512, 256, 128] |
| `amp_min_normalized_std` | 0.05 |

> **Note:** This run was trained with the pre-2026-06-29 config. After training started, AMP config
> was updated (discriminator [512,256,128]→[256,256], lerp 0.5→0.6, reward_coef 1.0→0.5,
> entropy 0.01→0.005) to align with G1 and beyondAMP defaults. Next run will use new config.

## Reward Weights (as saved in params/env.yaml — used during this training run)

| Reward | Weight | Type |
|---|---|---|
| `softstop` | 100.0 (base) → 250.0 (cu=3) | one-time, end-goal |
| `stopball` | 20.0 (base) → 50.0 (cu=3) | one-time, deflection |
| `single_foot_save` | 50.0 (base) → 125.0 (cu=3) | one-time, quality |
| `cleanstop` | 25.0 | one-time, quality |
| `inner_face_orientation_save` | 25.0 | one-time, quality |
| `foot_inner_face_continuous` | 5.0 | continuous, gated (~35-55 steps) |
| `footreach` | 10.0 | continuous |
| `foot_proximity` | 5.0 | continuous |
| `foot_clearance` | 2.0 | continuous |
| `ang_vel_z` | -2.0 | continuous penalty |

> **Note:** `ang_vel_z=-2.0` was set for this run (ball-spawn-frame fix context). Reverted to -0.5
> post-training. `softstop`/`stopball` weights also updated (105/15) for next run.

## AMP Motion Dataset (14 files)

| File | Side | Type |
|---|---|---|
| `LeftDoubleStep_own_booster_t1.npz` | Left | double step |
| `LeftSafe1_booster_t1.npz` | Left | single step |
| `LeftSafeFar1_booster_t1.npz` | Left | single step (far) |
| `LeftSafeFront1_booster_t1.npz` | Left | single step (front) |
| `LeftSafeMedium1_booster_t1.npz` | Left | single step (medium) |
| `LeftStep_own_booster_t1.npz` | Left | step |
| `LeftTripleStep_own_booster_t1.npz` | Left | triple step |
| `RightDoubleStep_own_booster_t1.npz` | Right | double step |
| `RightSafe1_booster_t1.npz` | Right | single step |
| `RightSafeFar1_booster_t1.npz` | Right | single step (far) |
| `RightSafeFront1_booster_t1.npz` | Right | single step (front) |
| `RightSafeMedium1_booster_t1.npz` | Right | single step (medium) |
| `RightTripleStep_own_booster_t1.npz` | Right | triple step |
| `Rightstep_own_booster_t1.npz` | Right | step |

AMP obs terms: `joint_pos` + `joint_vel` (42 dims per frame × 2 frames = 84 total).  
AMP anchor: `Trunk`. Discriminator hidden dims: [512, 256, 128] (this run; changed to [256,256] after).

## Metadata Attached to ONNX

| Key | Content |
|---|---|
| `run_path` | Absolute path to training run directory |
| `joint_names` | 21 joint names in T1 headless order |
| `joint_stiffness` | Per-joint stiffness (kp) values |
| `joint_damping` | Per-joint damping (kd) values |
| `default_joint_pos` | Home keyframe joint positions |
| `command_names` | (empty — no velocity commands) |
| `observation_names` | List of observation term names |
| `action_scale` | Per-joint action scale factors |

## Notes

- First run with `ang_vel_z=-2.0` (set during ball-spawn-frame fix 2026-06-20). This may have
  inhibited natural lateral movements during save attempts.
- AMP replay buffer: 250k (reduced from 500k in earlier runs to match T1 temporal ratio).
- Motion dataset: 14-file pool with double/triple step NPZs (added commits `e18d593`–`20550fc`).
- `correct_foot_save_curriculum` activates quality bonuses after cu≥3 (reliable saves established).
- This is an intermediate checkpoint of an ongoing run. Run may be continuing past iteration 8000.

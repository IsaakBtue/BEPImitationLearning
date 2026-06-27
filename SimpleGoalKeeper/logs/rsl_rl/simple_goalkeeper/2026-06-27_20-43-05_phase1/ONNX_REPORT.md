# ONNX Export Report — model_4500

**Run:** `2026-06-27_20-43-05_phase1`  
**Checkpoint:** `model_4500.pt` (iteration 4500)  
**Exported:** `2026-06-27_20-43-05_phase1.onnx`  
**Opset:** 18  
**Format:** ONNX with mjlab metadata  

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

## Training Hyperparameters

| Parameter | Value |
|---|---|
| Algorithm | AMPPPO |
| `num_steps_per_env` | 24 |
| `num_learning_epochs` | 5 |
| `num_mini_batches` | 4 |
| `learning_rate` | 1e-3 (adaptive) |
| `gamma` | 0.99 |
| `lam` | 0.95 |
| `entropy_coef` | 0.01 |
| `desired_kl` | 0.01 |
| `max_grad_norm` | 1.0 |
| `clip_param` | 0.2 |
| `amp_replay_buffer_size` | 500,000 |
| `amp_task_reward_lerp` | 0.5 |
| `amp_reward_coef` | 1.0 |
| `amp_min_normalized_std` | 0.05 |
| Smoothing loss | **disabled** (reverted — see commit history) |

## AMP Motion Dataset (10 files)

| File | Side | Type |
|---|---|---|
| `LeftSafe1_booster_t1.npz` | Left | single step |
| `LeftSafeFar1_booster_t1.npz` | Left | single step (far) |
| `LeftSafeFront1_booster_t1.npz` | Left | single step (front) |
| `LeftSafeMedium1_booster_t1.npz` | Left | single step (medium) |
| `LeftStep_own_booster_t1.npz` | Left | double/triple step |
| `RightSafe1_booster_t1.npz` | Right | single step |
| `RightSafeFar1_booster_t1.npz` | Right | single step (far) |
| `RightSafeFront1_booster_t1.npz` | Right | single step (front) |
| `RightSafeMedium1_booster_t1.npz` | Right | single step (medium) |
| `Rightstep_own_booster_t1.npz` | Right | double/triple step |

AMP obs terms: `joint_pos` + `joint_vel` (42 dims).  
AMP anchor: `Trunk`. Discriminator hidden dims: [512, 256, 128].

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

- This run (`2026-06-27_20-43-05`) started after the `act_inference` bug fix (commit `ae23820`, 2026-06-27 14:33) but **before** today's full smoothing revert (commits `0915064`–`67b69a4`, 2026-06-28). The smoothing was technically compiled into the binary but effectively disabled: `value_smoothness_coef=0.0` killed value smooth entirely; policy smooth ran with coef ≈ 0.0101 (≈10× smaller than G1 default).
- Future runs trained after the full revert (2026-06-28) will have zero smoothing overhead.
- Training used `amp_replay_buffer_size=500k` (vs 100k in all runs before `2e6c452`).

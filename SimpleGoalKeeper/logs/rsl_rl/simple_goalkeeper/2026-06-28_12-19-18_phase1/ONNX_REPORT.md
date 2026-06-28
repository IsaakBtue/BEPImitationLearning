# ONNX Export Report — model_11250

**Run:** `2026-06-28_12-19-18_phase1`  
**Checkpoint:** `model_11250.pt` (iteration 11,250)  
**Exported:** `2026-06-28_12-19-18_phase1.onnx`  
**Opset:** 18  
**Format:** ONNX with mjlab metadata  

---

## Training Snapshot at Iteration 11,250

| Metric | Value |
|---|---|
| `mean_reward` | 31.08 |
| `mean_amp_reward` | 43.10 |
| `mean_episode_length` | 144.7 steps (≈ 2.9 s) |
| `learning_rate` (adaptive) | 3e-4 |
| Total iterations run | ~12,241 |

Run started 2026-06-28 12:19 UTC+2. Training stopped at ~12,241 / 50,000 iterations.  
Reward plateaued at ~30–37 from ~iteration 5,000 onward; model_11250 is representative of the converged behaviour.

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

## Training Hyperparameters

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
| `amp_min_normalized_std` | 0.05 |

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

- This is the first run using the **14-motion dataset** with double/triple step NPZs
  (added in commits `e18d593`–`20550fc`, all resampled to 73 frames / 1.46 s at 50 fps).
- `amp_replay_buffer_size` reduced from 500k → 250k to match T1 temporal ratio (commit `fd7b7e4`).
- Reward plateaued around iteration 5,000–6,000; training converged without further improvement.
  Likely cause: AMP discriminator saturation (logits ≈ −95 to −101) giving near-constant style reward.
- Ball spawn uses t_flight-first approach (reaction time 0.7–1.3 s), added in commit `7f73408`.

# ONNX Export Report — model_13500

**Run:** `2026-06-29_20-36-28_phase1`  
**Checkpoint:** `model_13500.pt` (iteration 13,500)  
**Exported:** `2026-06-29_20-36-28_phase1.onnx`  
**Opset:** 18  
**Format:** ONNX with mjlab metadata  

---

## Training Snapshot at Iteration 13,500

| Metric | Value |
|---|---|
| `mean_reward` | ~55.6 |
| `mean_episode_length` | ~142 steps (≈ 2.84 s) |
| `mean_amp_reward` | ~10.3 |
| `discri_logits` | ~−118 |
| `Policy/noise_std` | ~0.41 (converging from 0.98) |
| `ball_difficulty` curriculum | **1.0** (fully maxed) |
| Total planned iterations | 50,000 |
| Run started | 2026-06-29 20:36 UTC+2 |

Strong mid-run checkpoint at 27% of training. Reward grew from −17 (start) → +55.6. Episode length nearly tripled from 54 → 142 steps. Ball difficulty curriculum fully advanced.

---

## Save Rates at Iteration 13,500

Computed via `rate = logged_value × max_episode_length_s / (weight × dt)`.

| Term | Logged | Weight (cu) | Rate |
|---|---|---|---|
| `softstop` | 1.55 | 262.5 | **88.6%** of episodes |
| `stopball` | 0.247 | 37.5 | **98.8%** of episodes |
| `single_foot_save` | 0.408 | 100.0 | **61.2%** of episodes |
| `cleanstop` | 0.110 | 50.0 | **33.0%** of episodes |

---

## Termination Breakdown

| Cause | Rate |
|---|---|
| `time_out` | ~39% of episodes |
| `ball_exit` | ~3.3% |
| `base_height` | ~1.3% |
| `bad_orientation` | **0%** |
| `sharpforce` | <0.01% |

No falls from bad orientation — robot remains upright and stable throughout.

---

## Model Architecture

| Property | Value |
|---|---|
| Input (obs) | 710 |
| Output (actions) | 21 |
| Hidden dims | [512, 256, 128] |
| Activation | ELU |

---

## Observation Space (actor, 710-dim, 10-step history)

| Term | Shape per step | Total |
|---|---|---|
| `base_ang_vel` | (3,) | (30,) |
| `projected_gravity` | (3,) | (30,) |
| `joint_pos_rel` | (21,) | (210,) |
| `joint_vel` | (21,) | (210,) |
| `actions` | (21,) | (210,) |
| `ball_pos_b` | (2,) | (20,) |

---

## Curriculum Weights at Iteration 13,500

| Term | Weight |
|---|---|
| `softstop` | 262.5 |
| `stopball` | 37.5 |
| `single_foot_save` | 100.0 |
| `cleanstop` | 50.0 |
| `inner_face_orientation_save` | 50.0 |
| `footreach` | 25.0 |
| `foot_inner_face_continuous` | 10.0 |
| `ball_difficulty` | 1.0 (max) |

---

## Training Hyperparameters

| Parameter | Value |
|---|---|
| `amp_task_reward_lerp` | 0.6 |
| `amp_reward_coef` | 0.5 |
| `amp_discr_hidden_dims` | [256, 256] |
| `entropy_coef` | 0.005 |
| `amp_replay_buffer_size` | 250,000 |

---

## Notes

- AMP discriminator logits at −118 (more negative than iter 2,750 at −87). Task reward is dominating motion naturalness at this stage — expected at 27% of training.
- `mean_noise_std` reduced from 0.98 → 0.41, indicating policy is committing to specific strategies.
- Previous export from this run was at iter 2,750 (5.5% of training, softstop <1%). This checkpoint represents a qualitatively different policy — active foot deflection in 89% of episodes.

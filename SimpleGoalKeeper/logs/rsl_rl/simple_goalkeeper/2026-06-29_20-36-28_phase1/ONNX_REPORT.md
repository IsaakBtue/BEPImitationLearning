# ONNX Export Report — model_2750

**Run:** `2026-06-29_20-36-28_phase1`  
**Checkpoint:** `model_2750.pt` (iteration 2,750)  
**Exported:** `2026-06-29_20-36-28_phase1.onnx`  
**Opset:** 18  
**Format:** ONNX with mjlab metadata  

---

## Training Snapshot at Iteration 2,750

| Metric | Value |
|---|---|
| `mean_reward` | ~37–39 |
| `mean_episode_length` | ~117–127 steps (≈ 2.3–2.5 s) |
| `mean_amp_reward` | ~14–17 |
| `discri_logits` | ~-87 to -95 |
| Total planned iterations | 50,000 |
| Run started | 2026-06-29 20:36 UTC+2 |

Early training (5.5% of total). Episode length grown from 23 → ~120 steps. Softstop firing rarely (~0.5% of episodes) — footreach is the dominant signal at this stage.

---

## Purpose of this Export

Intermediate checkpoint to visually inspect:
- Orange foot contact sensor geoms visible in viewer (group 1, rgba 1 0.5 0 0.8)
- Trailing foot toe slippage detection fix (toe-tip sample at x=+0.105 in foot-local)
- `foot_ang_vel_xy` penalty (-0.5) effect on heel-first landings
- `ang_vel_xy` increased from -0.1 → -0.5

---

## Model Architecture

| Property | Value |
|---|---|
| Input (obs) | 710 |
| Output (actions) | 21 |
| Hidden dims | [512, 256, 128] |
| Activation | ELU |

---

## Training Hyperparameters (first run with new AMP config)

| Parameter | Value |
|---|---|
| `amp_task_reward_lerp` | 0.6 |
| `amp_reward_coef` | 0.5 |
| `amp_discr_hidden_dims` | [256, 256] |
| `entropy_coef` | 0.005 |
| `amp_replay_buffer_size` | 250,000 |

---

## Curriculum at iter 2,750

| Term | Weight |
|---|---|
| `softstop` | 210.0 (cu=2) |
| `stopball` | 30.0 (cu=2) |
| `footreach` | 20.0 (cu=2) |
| `single_foot_save` | 50.0 (cu=0) |
| `ball_difficulty` | 0.667 |

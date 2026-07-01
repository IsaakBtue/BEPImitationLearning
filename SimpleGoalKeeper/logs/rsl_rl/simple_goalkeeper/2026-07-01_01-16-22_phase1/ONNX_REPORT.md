# ONNX Export Report — model_9000

**Run:** `2026-07-01_01-16-22_phase1`
**Checkpoint:** `model_9000.pt` (iteration 9,000)
**Exported:** `2026-07-01_01-16-22_phase1.onnx`
**Opset:** 18
**Format:** ONNX with mjlab metadata

---

## Training Snapshot at Iteration 9,000

| Metric | Value |
|---|---|
| `mean_reward` | ~55.1 |
| `mean_episode_length` | ~145.1 steps (≈ 2.90 s) |
| `mean_amp_reward` | ~25.6 |
| `discri_logits` | ~−87.8 |
| `Policy/noise_std` | ~0.40 (converged from 1.00) |
| `ball_difficulty` curriculum | **1.0** (fully maxed since ~iter 5,000) |
| Total planned iterations | 50,000 (run is **still training live**, currently past iter 10,200) |
| Run started | 2026-07-01 01:16 (immediately after commit `54f9828`: shank termination, knee penalty 0.255 m, y_end ±0.9 m) |

Reward climbed steadily from −6.2 (iter 0) → 55.1 (iter 9,000), essentially plateauing from iter ~7,000 onward (52.2 → 52.7 → 54.0 → 55.1, i.e. diminishing returns). Episode length grew from 23 → 145 steps and has also flattened since iter ~7,500.

---

## Save Rates at Iteration 9,000

Computed via `rate = logged_value × max_episode_length_s / (weight × dt)`, `max_episode_length_s = 3.0`, `dt = 0.02`.

| Term | Logged | Weight (cu) | Rate |
|---|---|---|---|
| `softstop` | 1.6053 | 262.5 | **91.7%** of episodes |
| `stopball` | 0.2465 | 37.5 | **98.6%** of episodes |
| `single_foot_save` | 0.4714 | 100.0 | **70.7%** of episodes |
| `inner_face_orientation_save` | 0.1022 | 50.0 | **30.7%** of episodes |
| `cleanstop` | 0.1373 | 50.0 | **41.2%** of episodes |

`foot_inner_face_continuous` (1.53 logged) is a per-step continuous reward, not one-shot — not converted to a rate.

---

## Termination Breakdown (iter ~9,000)

| Cause | Rate |
|---|---|
| `time_out` | ~40.3% of episodes |
| `ball_exit` | ~2.4% |
| `shank_height` | ~1.9% (new termination, added this run) |
| `base_height` | ~0.13% |
| `bad_orientation` | **0%** |
| `sharpforce` | **0%** |

No falls from bad orientation across the whole run so far. `shank_height` (added in commit `54f9828` to catch deep single-step lunges) fired at ~4.3% early on and has already dropped to ~1.9% by iter 9,000 — the policy is adapting to the tighter 0.24 m floor.

---

## Model Architecture

| Property | Value |
|---|---|
| Input (obs) | 710 |
| Output (actions) | 21 |
| Hidden dims | [512, 256, 128] |
| Activation | ELU |

---

## Curriculum Weights at Iteration 9,000

| Term | Weight |
|---|---|
| `softstop` | 262.5 |
| `stopball` | 37.5 |
| `single_foot_save` | 100.0 |
| `inner_face_orientation_save` | 50.0 |
| `cleanstop` | 50.0 |
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
| `num_envs` | 6,144 |
| `learning_rate` | 0.001 (adaptive, `desired_kl=0.01`) |

---

## Notes

- This run began immediately after `54f9828` (shank termination @0.24 m, knee-height penalty tightened 0.15→0.255 m, `y_end_range` narrowed ±1.0→±0.9 m). The new `shank_height` termination is already showing a clear downward trend (4.3% → 1.9%), i.e. the tighter lunge limit is being learned, not just causing resets.
- **Reward growth is flattening from iter ~7,000 onward** — the run is well into diminishing returns for this checkpoint's reward mix. `softstop`/`stopball` rates (91.7% / 98.6%) are near-saturated; remaining headroom is mostly in `single_foot_save` (70.7%), `cleanstop` (41.2%), and `inner_face_orientation_save` (30.7%), the save-quality terms rather than raw save rate.
- **Transient instability observed iter ~9,900–10,010** (after this checkpoint, so it does not affect `model_9000.pt`): `Train/mean_reward` spiked to as low as −4,416 on isolated iterations (67 iterations out of ~10,200 logged so far, all clustered in this ~100-iteration window) before fully recovering to the normal 47–55 range by iter 10,010 and continuing to climb normally through iter 10,220+. This pattern does not appear anywhere else in the run (0–9,500 or 10,010+ are clean). Root cause not yet diagnosed — worth checking driver/mujoco-warp logs from that wall-clock window (~11:10–11:25) for NaN/divergence warnings, since a mean-reward swing of this magnitude across 6,144 parallel envs implies either a widespread transient (e.g. a curriculum/env-reset edge case) or a severe single-env physics blow-up dragging the batch mean down.
- Previous export from this run was none (first export). This is the first checkpoint pushed from `2026-07-01_01-16-22_phase1`.

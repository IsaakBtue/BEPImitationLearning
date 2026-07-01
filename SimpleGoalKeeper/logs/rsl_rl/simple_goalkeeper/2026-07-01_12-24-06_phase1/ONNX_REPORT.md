# ONNX Export Report — model_5000

**Run:** `2026-07-01_12-24-06_phase1`
**Checkpoint:** `model_5000.pt` (iteration 5,000)
**Exported:** `2026-07-01_12-24-06_phase1.onnx`
**Opset:** 18
**Format:** ONNX with mjlab metadata

---

## Training Snapshot at Iteration 5,000

| Metric | Value |
|---|---|
| `mean_reward` | ~35.1 |
| `mean_episode_length` | ~134.2 steps (≈ 2.68 s) |
| `mean_amp_reward` | ~25.7 |
| `discri_logits` | ~−76.0 |
| `Policy/noise_std` | ~0.58 (still annealing from 1.00) |
| `ball_difficulty` curriculum | **1.0** (fully maxed) |
| Total planned iterations | 50,000 (run is **still training live**, currently past iter 5,090) |
| Run started | 2026-07-01 12:24, immediately after commit `b8ec3c3` (knee-height penalty 0.29 m, shank termination 0.275 m, `entropy_coef` 0.01) |

Reward climbed from −6.4 (iter 0) → 25.6 (iter 1,000) → 29.9 (iter 2,000) → 37.1 (iter 3,000), then dipped slightly to 36.5 (iter 4,000) and 35.1 (iter 5,000) — noisier than the smooth climb seen in the `2026-07-01_01-16-22` run at the same stage, consistent with the higher `entropy_coef` (0.01 vs previous run's 0.005) keeping exploration/policy noise higher this early. Episode length has grown from ~23 to ~134 steps.

---

## Save Rates at Iteration 5,000

Computed via `rate = logged_value × max_episode_length_s / (weight × dt)`, `max_episode_length_s = 3.0`, `dt = 0.02`.

| Term | Logged | Weight (cu) | Rate |
|---|---|---|---|
| `softstop` | 1.5945 | 262.5 | **91.1%** of episodes |
| `stopball` | 0.2461 | 37.5 | **98.4%** of episodes |
| `single_foot_save` | 0.4499 | 100.0 | **67.5%** of episodes |
| `inner_face_orientation_save` | 0.1197 | 50.0 | **35.9%** of episodes |
| `cleanstop` | 0.1201 | 50.0 | **36.0%** of episodes |

`foot_inner_face_continuous` (1.6902 logged) is a per-step continuous reward, not one-shot — not converted to a rate.

---

## Termination Breakdown (iter ~5,000)

| Cause | Rate |
|---|---|
| `time_out` | ~37.9% of episodes |
| `shank_height` | ~4.5% |
| `ball_exit` | ~2.7% |
| `base_height` | ~0.04% |
| `bad_orientation` | **0%** |
| `sharpforce` | **0%** |

No falls from bad orientation or sharp-contact terminations so far. `shank_height` (threshold tightened again to 0.275 m in `b8ec3c3`, up from 0.24 m in the prior run) is firing at ~4.5% — noticeably higher than the ~1.9% seen by iter 9,000 of the previous run, but that run had already adapted to a *looser* 0.24 m limit by that point; this run is only 5,000 iterations in against the tighter 0.275 m limit and knee penalty 0.29 m, so some further reduction as training continues is plausible but unconfirmed.

---

## Model Architecture

| Property | Value |
|---|---|
| Input (obs) | 710 |
| Output (actions) | 21 |
| Hidden dims | [512, 256, 128] |
| Activation | ELU |

---

## Curriculum Weights at Iteration 5,000

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
| `entropy_coef` | 0.01 |
| `amp_replay_buffer_size` | 250,000 |
| `num_envs` | 6,144 |
| `learning_rate` | 0.001 (adaptive, `desired_kl=0.01`) |

---

## Notes

- This run began immediately after `b8ec3c3` (knee-height penalty tightened 0.255→0.29 m, `shank_height` termination tightened 0.24→0.275 m, `entropy_coef` raised 0.005→0.01). Two earlier attempts today (`2026-07-01_12-08-01_phase1`, `2026-07-01_12-16-05_phase1`) only reached `model_0.pt` before being restarted — this is the run that actually took off.
- Requested motive for this push: checking how well iteration 5,000 handles the **double- and triple-step motion pools** (`MotionResetManager` routes wider lateral ball crossings, `|cross_y| ∈ [0.35, 0.65)` → double-step, `≥0.65` → triple-step, per side). This report is based on aggregate TensorBoard/wandb metrics only — it does **not** break down save/termination rates by motion pool, since mjlab's episode logging isn't segmented by RSI pool. Visual inspection via `uv run sgk_play Mjlab-BeyondAMP-Goalkeeper-T1 --checkpoint-file logs/rsl_rl/simple_goalkeeper/2026-07-01_12-24-06_phase1/model_5000.pt` is the way to actually judge double/triple-step quality.
- **Reward trend is noisier and lower at the same iteration count than the previous run** (35.1 @ 5,000 here vs the prior run's smoother climb toward 55 by iter 9,000) — likely a combination of being earlier in training and the higher `entropy_coef`, not necessarily a regression. Worth revisiting once this run reaches a comparable iteration count to the last one before drawing conclusions.
- Previous export from this run was none (first export). This is the first checkpoint pushed from `2026-07-01_12-24-06_phase1`.

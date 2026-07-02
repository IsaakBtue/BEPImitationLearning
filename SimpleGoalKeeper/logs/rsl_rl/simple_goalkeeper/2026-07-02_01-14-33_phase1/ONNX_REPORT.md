# ONNX Export Report — model_9750

**Run:** `2026-07-02_01-14-33_phase1`
**Checkpoint:** `model_9750.pt` (iteration 9,750)
**Exported:** `2026-07-02_01-14-33_phase1.onnx`
**Opset:** 18
**Format:** ONNX with mjlab metadata

---

## Training Snapshot at Iteration 9,750

| Metric | Value |
|---|---|
| `mean_reward` | ~45.9 |
| `mean_episode_length` | ~139.0 steps (≈ 2.78 s) |
| `mean_amp_reward` | ~13.4 |
| `discri_logits` | ~−109.6 |
| `Policy/noise_std` | ~0.52 (still annealing from 1.00) |
| `ball_difficulty` curriculum | **1.0** (fully maxed) |
| Total planned iterations | 50,000 (run is **still training live**, currently past iter 10,210) |
| Run started | 2026-07-02 01:14, immediately after commit `7040fa4` (ONNX export + report for `model_5750`) |

Reward climbed smoothly from −5.5 (iter 0) → 21.7 (iter 1,000) → 27.6 (iter 2,000) → 34.1 (iter 4,000) → 37.5 (iter 6,000) → 39.3 (iter 8,000) → 41.9 (iter 9,000) → 45.9 (iter 9,750), a cleaner monotonic-ish climb than the previous run (`2026-07-01_19-03-50`). One transient anomaly: `Train/mean_reward` collapsed to catastrophic negative outliers (as low as −8,943) for a handful of isolated iterations around 8,365–8,405, then fully recovered within ~40–50 iterations with no lasting effect on the trend — the second time this exact shape of transient collapse has been observed across runs (see project memory). `mean_amp_reward` is lower here (13.4) than the prior run's iter-5,750 value (21.9); this run trained with the AMP discriminator's 4× double/triple-step motion-weighting active from iteration 0 (code was on disk from ~01:02–01:13, before this run started at 01:14), so the discriminator is drawing more of its reference batch from the harder double/triple-step clips, which plausibly explains a lower/harder-to-satisfy style reward relative to the unweighted prior run.

---

## Save Rates at Iteration 9,750

Computed via `rate = logged_value × max_episode_length_s / (weight × dt)`, `max_episode_length_s = 3.0`, `dt = 0.02`.

| Term | Logged | Weight (cu) | Rate |
|---|---|---|---|
| `softstop` | 1.5088 | 262.5 | **86.2%** of episodes |
| `stopball` | 0.2448 | 37.5 | **97.9%** of episodes |
| `single_foot_save` | 0.4288 | 100.0 | **64.3%** of episodes |
| `inner_face_orientation_save` | 0.2050 | 50.0 | **61.5%** of episodes |
| `cleanstop` | 0.1130 | 50.0 | **33.9%** of episodes |

`foot_inner_face_continuous` (1.9528 logged), `foot_clearance` (0.3182), `foot_proximity` (0.5227), and `footreach` (9.1458) are per-step continuous rewards, not one-shot — not converted to a rate. All five save-rate terms are roughly flat-to-improving versus the prior checkpoint (`model_5750` at iter 5,750): `single_foot_save` 66.4%→64.3% (flat), `inner_face_orientation_save` 43.7%→61.5% (up), `cleanstop` 31.8%→33.9% (flat-to-up).

---

## Termination Breakdown (iter ~9,750)

| Cause | Rate |
|---|---|
| `time_out` | ~38.9% of episodes |
| `ball_exit` | ~4.2% |
| `shank_height` | ~3.5% |
| `base_height` | **0%** |
| `bad_orientation` | **0%** |
| `sharpforce` | **0%** |

No falls from bad orientation, base-height collapse, or sharp-contact terminations. `ball_exit` (~4.2%) and `shank_height` (~3.5%) remain the two active failure modes, both broadly in line with prior runs at a comparable stage.

---

## Model Architecture

| Property | Value |
|---|---|
| Input (obs) | 710 |
| Output (actions) | 21 |
| Hidden dims | [512, 256, 128] |
| Activation | ELU |

---

## Curriculum Weights at Iteration 9,750

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

- **This checkpoint predates the post-save ball-visibility fix landed 2026-07-02** (actor's `ball_pos_b` switched from `always_visible=True` to the G1-matching vanish gate in `goalkeeper_env_cfg.py`). `model_9750` was trained with the ball permanently visible to the actor, so it should still be expected to track/follow the ball after a save in `sgk_play` — that behavior is not fixed until a checkpoint from a run started after this fix lands.
- This run's AMP discriminator trained with the 4× double/triple-step motion-weighting active from iteration 0 (the feature code was written to disk just before this run launched, though not committed to git until the `model_9750` push). Despite that, no direct per-motion-pool save-rate breakdown is available — same logging limitation as prior reports (mjlab's episode logging is not segmented by RSI/motion pool) — so this report cannot confirm whether double/triple-step save quality specifically improved.
- Second occurrence of the transient catastrophic-negative-reward spike (iter ~8,365–8,405 here; iter ~9,900–10,010 in `2026-07-01_01-16-22`), both self-recovering with no lasting effect — worth root-causing if a third instance appears.

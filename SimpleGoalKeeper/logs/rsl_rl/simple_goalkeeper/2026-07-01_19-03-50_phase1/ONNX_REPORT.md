# ONNX Export Report — model_5750

**Run:** `2026-07-01_19-03-50_phase1`
**Checkpoint:** `model_5750.pt` (iteration 5,750)
**Exported:** `2026-07-01_19-03-50_phase1.onnx`
**Opset:** 18
**Format:** ONNX with mjlab metadata

---

## Training Snapshot at Iteration 5,750

| Metric | Value |
|---|---|
| `mean_reward` | ~36.6 |
| `mean_episode_length` | ~134.6 steps (≈ 2.69 s) |
| `mean_amp_reward` | ~21.9 |
| `discri_logits` | ~−86.9 |
| `Policy/noise_std` | ~0.56 (still annealing from 1.00) |
| `ball_difficulty` curriculum | **1.0** (fully maxed) |
| Total planned iterations | 50,000 (run is **still training live**, currently past iter 6,129) |
| Run started | 2026-07-01 19:03, immediately after commit `e066a9c` (regression + statistical coverage for the fixed RSI `reset()` — the literal G1 `continue_keep` port with the clamp/else-branch/80-20-split fidelity fixes from `10bb5c9`) |

Reward climbed from −5.5 (iter 0) → 21.8 (iter 1,000) → 30.9 (iter 2,000) → 32.0 (iter 3,000) → 36.6 (iter 4,000), dipped to 32.7 (iter 5,000), then recovered to 36.6 (iter 5,750) — noisier, non-monotonic movement rather than a smooth climb, but trending flat-to-up around the mid-30s since iter 4,000. Episode length has grown from ~22 to ~135 steps. `mean_amp_reward` dropped from 24.0 (iter 4,000) to 14.7 (iter 5,000) before recovering to 21.9 (iter 5,750) — a larger single-step swing than seen in prior runs at this stage, plausibly reflecting the discriminator adapting to the newly-fixed RSI reset distribution (previously silently-wrong 50/50 split, now the intended 80/20).

---

## Save Rates at Iteration 5,750

Computed via `rate = logged_value × max_episode_length_s / (weight × dt)`, `max_episode_length_s = 3.0`, `dt = 0.02`.

| Term | Logged | Weight (cu) | Rate |
|---|---|---|---|
| `softstop` | 1.5227 | 262.5 | **87.0%** of episodes |
| `stopball` | 0.2431 | 37.5 | **97.2%** of episodes |
| `single_foot_save` | 0.4424 | 100.0 | **66.4%** of episodes |
| `inner_face_orientation_save` | 0.1457 | 50.0 | **43.7%** of episodes |
| `cleanstop` | 0.1059 | 50.0 | **31.8%** of episodes |

`foot_inner_face_continuous` (1.6722 logged), `foot_clearance` (0.3154), and `foot_proximity` (0.5063) are per-step continuous rewards, not one-shot — not converted to a rate.

---

## Termination Breakdown (iter ~5,750)

| Cause | Rate |
|---|---|
| `time_out` | ~36.4% of episodes |
| `shank_height` | ~5.3% |
| `ball_exit` | ~3.9% |
| `base_height` | ~0.2% |
| `bad_orientation` | **0%** |
| `sharpforce` | **0%** |

No falls from bad orientation or sharp-contact terminations. `shank_height` (~5.3%) and `ball_exit` (~3.9%) are the two active failure modes; both are broadly in line with the previous run's rates at a comparable stage.

---

## Model Architecture

| Property | Value |
|---|---|
| Input (obs) | 710 |
| Output (actions) | 21 |
| Hidden dims | [512, 256, 128] |
| Activation | ELU |

---

## Curriculum Weights at Iteration 5,750

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

- This run began immediately after `e066a9c` (regression + statistical coverage added for the RSI `reset()` fix chain: `d496aa3` literal G1 `continue_keep` port, `10bb5c9` clamp/else-branch/80-20-split fidelity corrections). This is the first checkpoint pushed since that fix landed, so it's the first data point on whether the corrected RSI distribution changes training dynamics versus the prior run (`2026-07-01_12-24-06`, which trained under the silently-wrong 50/50 split).
- No ONNX/report breakdown by RSI motion pool (single/double/triple/wide × side) is available — same limitation as the previous report: mjlab's episode logging is not segmented by motion pool.
- `mean_amp_reward` swung more sharply between iter 4,000–5,750 here (24.0 → 14.7 → 21.9) than in the `12-24-06` run's comparable window — worth watching as more iterations land, since it could reflect the discriminator re-adjusting to the corrected 80/20 RSI split rather than a training regression.

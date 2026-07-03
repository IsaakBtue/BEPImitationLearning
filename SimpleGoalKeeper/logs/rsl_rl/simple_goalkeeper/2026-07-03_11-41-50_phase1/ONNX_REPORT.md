# ONNX Export Report — model_6250

**Run:** `2026-07-03_11-41-50_phase1`
**Checkpoint:** `model_6250.pt` (iteration 6,250)
**Exported:** `2026-07-03_11-41-50_phase1.onnx`
**Opset:** 18
**Format:** ONNX with mjlab metadata

---

## Training Snapshot at Iteration 6,250

| Metric | Value |
|---|---|
| `mean_reward` | ~35.9 |
| `mean_episode_length` | ~133.5 steps (≈ 2.67 s) |
| `mean_amp_reward` | ~10.8 |
| `discri_logits` | ~−112.2 |
| `Policy/noise_std` | ~0.56 (still annealing from 1.00) |
| `ball_difficulty` curriculum | **1.0** (fully maxed) |
| Total planned iterations | 50,000 (run is **still training live**, at iter ~6,491 as of this report) |
| Run started | 2026-07-03 11:41, immediately after commit `48c90e8` (reinstate ball-conditioned NPZ tier RSI as 30/50/20 split) |

Reward climbed monotonically from −5.5 (iter 0) → 18.0 (iter 250) → 22.2 (iter 1,000) → 29.2 (iter 2,000) → 33.4 (iter 3,000) → 31.6 (iter 4,000, minor dip) → 36.2 (iter 5,000) → 35.8 (iter 6,000) → 35.9 (iter 6,250), with no divergence or catastrophic-negative-reward spikes anywhere in the logged history (checked full run, no `Train/mean_reward` value below −50 at any iteration) — a clean run, similar in character to the `2026-07-02_20-18-38` control run.

**This is the first full run to combine all three fixes landed 2026-07-03:**
- `ce69f36` — v2 `hide_when_behind` gate (G1 torso-edge + catch-window ball-visibility mask, correcting the v1 gate that caused 17% iteration divergence in `2026-07-02_22-56-40`), plus Step motions re-added to the AMP dataset.
- `e79e138` — widened ball `y_end_range` to ±1.1 (train+play) to force double-stepping.
- `48c90e8` — reinstated ball-conditioned NPZ tier RSI as a 30/50/20 reset split (routes resets by predicted goal-line crossing Y).

No prior run has trained under this exact combination, so this checkpoint is the first read on whether double-stepping actually converges under the wider ball range with the corrected visibility gate.

---

## Save Rates at Iteration 6,250

Computed via `rate = logged_value × max_episode_length_s / (weight × dt)`, `max_episode_length_s = 3.0`, `dt = 0.02`.

| Term | Logged | Weight (cu) | Rate |
|---|---|---|---|
| `softstop` | 1.4099 | 262.5 | **80.6%** of episodes |
| `stopball` | 0.2347 | 37.5 | **93.9%** of episodes |
| `single_foot_save` | 0.3987 | 100.0 | **59.8%** of episodes |
| `inner_face_orientation_save` | 0.1547 | 50.0 | **46.4%** of episodes |
| `cleanstop` | 0.0725 | 50.0 | **21.8%** of episodes |

`foot_inner_face_continuous` (1.8373 logged), `foot_clearance` (0.3119), `foot_proximity` (0.4682), and `footreach` (8.6839) are per-step continuous rewards, not one-shot — not converted to a rate.

No like-for-like prior checkpoint exists for a direct save-rate trend: the last full `ONNX_REPORT.md` (`model_9750`, run `2026-07-02_01-14-33`) predates the v2 visibility gate, the widened ball range, and the new RSI split, so its save rates (`single_foot_save` 64.3%, `inner_face_orientation_save` 61.5%, `cleanstop` 33.9%) are not directly comparable — this run is establishing a new baseline under the current feature set, not continuing the old one.

---

## Termination Breakdown (iter ~6,250)

| Cause | Rate |
|---|---|
| `time_out` | ~36.1% of episodes |
| `shank_height` | ~5.75% |
| `ball_exit` | ~5.6% |
| `base_height` | ~0.08% |
| `bad_orientation` | **0%** |
| `sharpforce` | **0%** |

No falls from bad orientation or sharp-contact terminations, and `base_height` collapse is negligible (~0.08%). `ball_exit` (~5.6%) and `shank_height` (~5.75%) are the two active failure modes — both slightly higher than the `model_9750` report's iter-9,750 values (4.2% / 3.5%), plausibly because this checkpoint is far earlier in training (iter 6,250 vs 9,750) under a harder task (wider ball range forcing double-steps).

---

## Model Architecture

| Property | Value |
|---|---|
| Input (obs) | 710 |
| Output (actions) | 21 |
| Hidden dims | [512, 256, 128] |
| Activation | ELU |

---

## Curriculum Weights at Iteration 6,250

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

- **First full run combining the v2 `hide_when_behind` gate, the widened ±1.1 ball range, and the 30/50/20 ball-conditioned RSI split.** No direct per-motion-pool (single/double/triple-step) save-rate breakdown is available — same mjlab episode-logging limitation as prior reports — so this checkpoint cannot yet confirm whether double-stepping specifically improved under the wider range; that will need `sgk_play`/`sgk_play_rsi` qualitative review or a future logging change.
- Training was still live at export time (past iter 6,491); a later checkpoint from this same run may supersede `model_6250` once `ball_exit`/`shank_height` termination rates and save rates stabilize further.
- No reward-collapse anomalies observed in this run's history so far, unlike the two prior occurrences noted in `2026-07-01_01-16-22` (iter ~9,900–10,010) and `2026-07-02_01-14-33` (iter ~8,365–8,405) — worth continuing to watch as this run progresses past iter 8,000.

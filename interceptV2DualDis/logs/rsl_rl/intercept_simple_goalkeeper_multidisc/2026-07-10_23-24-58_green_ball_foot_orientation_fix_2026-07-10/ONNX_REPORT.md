# ONNX Export Report — model_19000

**Run:** `2026-07-10_23-24-58_green_ball_foot_orientation_fix_2026-07-10`
**Branch:** `green-ball-baseline`
**Checkpoint:** `model_19000.pt` (iteration 19,000)
**Exported:** `2026-07-10_23-24-58_green_ball_foot_orientation_fix_2026-07-10.onnx`
**Format:** ONNX with mjlab metadata

---

## Why this checkpoint was pushed

This run plateaued. `stopball`/`softstop`/`mean_episode_length` all flattened around iteration ~9,000 and have shown no further improvement through iteration 20,250 (the run is still live and left running, not stopped). `model_19000.pt` is pushed as the representative "converged" checkpoint for this branch.

## Training Snapshot at Iteration 19,000

| Metric | Value |
|---|---|
| `Train/mean_reward` | 33.66 |
| `Train/mean_episode_length` | 140.7 steps (≈ 2.8 s) |
| `Train/mean_amp_reward` | 7.40 |
| `Train/mean_discri_logits` | −134.9 |
| `Policy/mean_noise_std` | 0.524 |
| `ball_difficulty` curriculum | 1.0 (fully saturated) |
| Total iterations logged so far | 20,250 (run still live) |

## Plateau Evidence

`Train/mean_reward` trend: −0.9 (iter 0) → 23.4 (1,000) → 21.4-23.1 (2,000-4,000, noisy) → **31.9 (9,000)** → 32.8 (15,000) → **33.7 (19,000)** — reward has moved less than 2 points over the last 10,000 iterations after an earlier fast climb. `Episode_Reward/stopball` (~0.245) and `softstop` (~1.61-1.62) have been flat since iteration 9,000 in the same way (see `docs/BugFixes.md`/health-check history for the full trend). `mean_episode_length` similarly flat (~138-145) since iteration 9,000-12,000. No instability accompanying the plateau: `shank_height` terminations low (~1.4 raw count, well below early-training levels), no NaN/inf anywhere in the scalar log.

## Episode Rewards at Iteration 19,000 (raw logged values, not converted to rates)

| Term | Logged value | Curriculum weight |
|---|---|---|
| `softstop` | 1.611 | 210.0 |
| `stopball` | 0.245 | 30.0 |
| `single_foot_save` | 0.514 | 50.0 |
| `inner_face_orientation_save` | 0.259 | 25.0 |
| `cleanstop` | 0.027 | 25.0 |
| `foot_inner_face_continuous` | 1.701 | 5.0 |
| `footreach` | — | 20.0 |
| `foot_clearance` | 0.335 | (per-step, uncurriculummed) |
| `foot_proximity` | 0.564 | (per-step, uncurriculummed) |

Not converted to save-rate percentages this time (the logged-value→rate formula documented elsewhere in this repo assumes a strictly boolean one-shot reward; a couple of these terms produced rates >100% under that formula against the current curriculum weights here, suggesting the formula doesn't cleanly apply to every term as currently weighted — flagging rather than reporting a number I can't verify is correct).

## Termination Breakdown at Iteration 19,000 (relative share, among terminations only)

| Cause | Raw logged value | Share of terminations |
|---|---|---|
| `time_out` | 25.50 | 90.5% |
| `shank_height` | 1.375 | 4.9% |
| `ball_exit` | 1.292 | 4.6% |
| `base_height` | 0.0 | 0% |
| `bad_orientation` | 0.0 | 0% |
| `sharpforce` | 0.0 | 0% |

Vast majority of episodes run to `time_out` (i.e., complete without a hard failure). No falls from bad orientation or sharp-contact terminations at all.

## Model Architecture

| Property | Value |
|---|---|
| Actor input (obs) | 71 (`actor_current` group, single-step) / 710 (`actor`, 10-step history) |
| Output (actions) | 21 |
| Hidden dims | [512, 256, 128] |
| Activation | ELU |

## Curriculum Weights at Iteration 19,000 (all at ceiling — `ball_difficulty` saturated)

| Term | Weight |
|---|---|
| `softstop` | 210.0 |
| `single_foot_save` | 50.0 |
| `stopball` | 30.0 |
| `inner_face_orientation_save` | 25.0 |
| `cleanstop` | 25.0 |
| `footreach` | 20.0 |
| `foot_inner_face_continuous` | 5.0 |

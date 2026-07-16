# ONNX Export Report — model_15500

**Run:** `2026-07-15_22-08-11_green_overshoot_arms_contact_fix_2026-07-15`
**Branch:** `master`
**Checkpoint:** `model_15500.pt` (iteration 15,500)
**Exported:** `2026-07-15_22-08-11_green_overshoot_arms_contact_fix_2026-07-15.onnx`
**Format:** ONNX with mjlab metadata

---

## Why this checkpoint was pushed

Snapshot before widening `y_end_range` back to 1.3m and restarting. This run bundled four fixes on top of the prior architecture-restoration work: (1) halved the far-region `footreach` `vel_sigma` escalation (G1's version is paired with dive-specific landing-safety rewards this grounded-stepping task has no equivalent for — no direct G1 fix exists, this was a judgment call), (2) removed arms from the AMP discriminator entirely (21 → 13 joints, arms have no task-grounding reward), (3) added a genuine ball-contact sensor and rewired `stopball`/`softstop`'s correct-foot gate to it (the old sensor was accidentally just a ground-contact check), (4) `y_end_range` was reverted to 0.9m for this run specifically to isolate the `vel_sigma` boost from the range-widening change.

## Training Snapshot at Iteration 15,500

| Metric | Value |
|---|---|
| `Train/mean_reward` | 29.07 |
| `Train/mean_episode_length` | 124.9 steps (≈ 2.5 s) |
| `Policy/mean_noise_std` | 0.774 |
| `ball_difficulty` curriculum | 1.0 (saturated since ~iteration 1500) |
| `Loss/AMP` | 0.059 (converged; initial spike to ~1042 at step 0 self-recovered within ~30 iterations, same benign pattern seen throughout this project) |
| `Loss/est_ball` | 0.050 |
| `Loss/est_region` | 0.491 |

## Overshoot/Stability Check (the specific issue this run's fixes targeted)

`Episode_Termination/shank_height` — the termination most associated with the "doesn't stand properly, overshoots and falls" complaint that motivated fix #1 above:

Trend across this run: 51.9 (iter 0) → 16.1 → 10.1 → 6.5 → 8.2 → 8.5 → 7.8 → 7.5 → **8.8 (iter 15,500)**.

This is markedly lower and more stable than the pre-fix baseline observed on the immediately preceding run (`green_far_travel_fix_2026-07-15`), which held flat at ~24-31 for its entire ~2000+ post-saturation window with no downward trend. Not proof the overshoot problem is fully solved (still nonzero, and this run's four fixes are bundled/confounded), but a real, measurable improvement in the right direction.

## Episode Rewards at Iteration 15,500 (raw logged values)

| Term | Logged value |
|---|---|
| `footreach` | 21.594 |
| `foot_proximity` | 1.631 |
| `foot_clearance` | 1.235 |
| `foot_inner_face_continuous` | 3.052 |
| `softstop` | 0.149 |
| `stopball` | 0.032 |
| `single_foot_save` | 0.027 |
| `inner_face_orientation_save` | 0.012 |
| `cleanstop` | 0.001 |

`softstop`/`stopball` are measured under the NEW, stricter `ball_contact` sensor gate (fix #3) — not directly comparable to pre-fix logged values from earlier runs, which used the looser ground-contact check.

## Termination Breakdown at Iteration 15,500 (raw logged values)

| Cause | Raw logged value |
|---|---|
| `time_out` | 21.875 |
| `shank_height` | 8.792 |
| `ball_exit` | 0.792 |
| `base_height` | 0.000 |
| `bad_orientation` | 0.000 |
| `sharpforce` | 0.000 |

No falls from bad orientation or sharp-contact at all. `shank_height` is the dominant non-timeout termination, consistent with the stability discussion above.

## Model Architecture

Actor MLP inferred directly from the checkpoint's `actor.*` weight shapes (same export path used for prior checkpoints); ONNX metadata (`joint_names`, `joint_stiffness`, `joint_damping`, `default_joint_pos`, `command_names`, `observation_names`, `action_scale`) attached via `mjlab.rl.exporter_utils`.

**Not yet resolved:** this snapshot precedes a `y_end_range` widening back to 1.3m and a fresh restart — the numbers above describe the 0.9m-range configuration only, not yet validated at the wider range.

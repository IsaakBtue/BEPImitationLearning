# ONNX Export Report — model_7000

**Run:** `2026-09-01_12-00-05_6144_watchdogresume_2026-09-01_12-00`
**Branch:** `v2-blue-ball-waypoint`
**Checkpoint:** `model_7000.pt` (`ckpt["iter"] == 7000`)
**Exported:** `2026-09-01_12-00-05_6144_watchdogresume_2026-09-01_12-00.onnx`
**Task:** `Mjlab-BeyondAMP-Goalkeeper-T1-MultiDisc` (HIM / multi-discriminator architecture)
**Format:** ONNX (opset 18) with mjlab metadata attached
**Exported via:** `uv run sgk_export <checkpoint.pt>` (no export-script fixes needed this time — the multi-disc dict-cfg path and `group_obs_term_dim` positional-lookup fixes from the earlier `model_39750` export are already in place).

This checkpoint came from a run launched by the unattended GPU watchdog
(`intercept_gpu_watchdog.sh`, see `.claude/skills/gpu-watchdog-automation/`),
resumed from an earlier lineage rather than started fresh — the run
directory name reflects the watchdog's own naming convention, not a
human-chosen descriptive name.

## Architecture Verification (full HIM stack, not actor-only)

Confirmed directly from the checkpoint's `model_state_dict` layer shapes —
identical architecture to the previously-verified `model_39750` export
(same task, same hyperparameters):

| Submodule | Layer weight shapes (out, in) | Role |
|---|---|---|
| `history_encoder` | (128,710) → (64,128) → (16,64) | Compresses the raw 710-dim (`obs_history` = 71 single-step dims × 10-step history) into a 16-dim latent code |
| `ball_estimator` | (128,710) → (32,128) → (4,32) | Estimates a 4-dim ball state from the same 710-dim obs history |
| `region_estimator` | (128,710) → (32,128) → (4,32) | Estimates a 4-class region logit vector from the same obs history; exported graph applies `argmax` → 1-dim `region_arg`, matching training (`him_actor_critic.py`) |
| `actor` | (512,92) → (256,512) → (256,256) → (21,256) | Final policy: `obs_current(71) + history_latent(16) + estimate_ball(4) + region_arg(1) = 92` → 21 joint actions |

Derived vs. actual dimension check performed at export time (fails loudly
if it ever mismatches):
```
obs_current(71) + history_latent(16) + estimate_ball(4) + region_arg(1) = 92 == actor's real input dim (92)  ✓
```

Per-term obs order/size/scale, read live from `env_cfg.observations["actor"]`
(training concat order, not hardcoded) — unchanged since the last export:

| Term | Size (per frame) | Scale |
|---|---|---|
| `base_ang_vel` | 3 | 0.25 |
| `projected_gravity` | 3 | 1.0 |
| `joint_pos_rel` | 21 | 1.0 |
| `joint_vel` | 21 | 0.05 |
| `actions` | 21 | 1.0 |
| `ball_pos_b` | 2 | 1.0 |

`num_one_step_obs = 71`, `history_length = 10` → raw ONNX input `obs` is
`[1, 710]`, **unscaled** (the scale table above is baked into the graph's
first op — the deploy side must feed raw sensor values, not pre-scaled
ones — see `.claude/skills/exporting-him-checkpoints-to-onnx/` for why this
convention was chosen). ONNX output `actions` is `[1, 21]`.

## Parity Verification

- `onnx.checker.check_model()` — passed.
- Rebuilt every submodule (`history_encoder`/`ball_estimator`/
  `region_estimator`/`actor`) directly from the checkpoint's raw weight
  tensors using `export_onnx.py`'s own `_build_mlp_from_prefix`/
  `HimInterceptExportWrapper` (imported directly — this verification ran
  inside the real project venv, so the skill's "outside the training
  stack" permissive-unpickler stub wasn't needed here) and compared the
  same forward pass against `onnxruntime`'s output over 20 random
  `[1, 710]` inputs.
- **Max abs diff (PyTorch vs. ONNX): `1.359e-05`** — float32 noise floor,
  consistent with the earlier `model_39750` export's `1.14e-05`. Confirms
  architecture + baked-in scale correctness, not learned-behavior quality.

## What this report does NOT verify

Per the export skill's own scope: this confirms the exported graph computes
what the checkpoint was trained to compute (architecture + numerical
parity), not that the checkpoint's learned behavior is good — this is an
early checkpoint (iteration 7000 of a 20000-iteration target, itself
resumed mid-lineage through several stop/resume cycles via the GPU
watchdog), not a converged final policy. No live rollout / success-rate
probe was run as part of this export.

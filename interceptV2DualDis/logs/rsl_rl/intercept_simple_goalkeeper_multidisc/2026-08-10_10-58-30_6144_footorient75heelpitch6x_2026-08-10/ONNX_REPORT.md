# ONNX Export Report — model_39750

**Run:** `2026-08-10_10-58-30_6144_footorient75heelpitch6x_2026-08-10`
**Branch:** `v2-blue-ball-waypoint`
**Checkpoint:** `model_39750.pt` (`ckpt["iter"] == 39750`)
**Exported:** `2026-08-10_10-58-30_6144_footorient75heelpitch6x_2026-08-10.onnx`
**Task:** `Mjlab-BeyondAMP-Goalkeeper-T1-MultiDisc` (HIM / multi-discriminator architecture)
**Format:** ONNX (opset 18) with mjlab metadata attached

---

## Export-script bug fixed to produce this export

`export_onnx.py`'s `_load_env()` had never been updated for the multi-disc
task's rl_cfg, which is registered as a **plain dict** consumed by
`HimAMPOnPolicyRunner` (`tasks/__init__.py`), not an `AMPRunnerCfg`
dataclass instance. The exporter still had `assert isinstance(agent_cfg,
AMPRunnerCfg)`, which fails outright for this task — this export path was
non-functional for the actual `-MultiDisc` task before this fix. Mirrored
`play.py`'s existing `is_multidisc` dict-branch handling
(`AMPEnvWrapper(env, clip_actions=None, motion_dataset=None)`).

A second, related bug was hit immediately after: `group_obs_term_dim` in
this mjlab version is a **list** positionally aligned with
`observation_manager.active_terms[group]`, not a `dict` keyed by term name
— `term_dims_lookup["actor"][term_name]` raised `TypeError`. Fixed to index
by position via `active_terms["actor"]` instead of `actor_group.terms.keys()`
(the latter can include a term the manager itself skipped).

A third bug (introduced together with the second fix, caught by the
exporter's own self-check): `group_obs_term_dim` reports the
**history-flattened** per-term size (e.g. `base_ang_vel`: 3 per frame ×
history_length 10 = 30), but `HimInterceptExportWrapper`'s offset math
expects the **single-frame** size. Fixed by dividing each reported dim by
`history_length` before passing it in as `term_sizes`. The exporter's own
`expected_actor_in == actor_in_dim` assertion (92 == 92) is what caught
this — confirms the derive-don't-hardcode design in
`.claude/skills/exporting-him-checkpoints-to-onnx/` did its job.

All three fixes are scoped to `export_onnx.py` only; no training-affecting
code was touched.

## Architecture Verification (full HIM stack, not actor-only)

Confirmed directly from the checkpoint's `model_state_dict` layer shapes —
this export includes all four submodules, not just the actor trunk:

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
(training concat order, not hardcoded):

| Term | Size (per frame) | Scale |
|---|---|---|
| `base_ang_vel` | 3 | 0.25 |
| `projected_gravity` | 3 | 1.0 |
| `joint_pos_rel` | 21 | 1.0 |
| `joint_vel` | 21 | 0.05 |
| `actions` | 21 | 1.0 |
| `ball_pos_b` | 2 | 1.0 |

`num_one_step_obs = 71`, `history_length = 10` → raw ONNX input `obs_history`
is `[1, 710]`, **unscaled** (the scale table above is baked into the graph's
first op, so the deploy side must feed raw sensor values, not pre-scaled
ones — see the skill doc for why this convention was chosen). ONNX output
`actions` is `[1, 21]`.

## Parity Verification

- `onnx.checker.check_model()` — passed.
- Independently rebuilt every submodule (`history_encoder`/`ball_estimator`/
  `region_estimator`/`actor`) directly from the checkpoint's raw weight
  tensors (not via the training stack — permissive-unpickler stub used to
  load the checkpoint outside the `rsl_rl_amp`/`simple_goalkeeper.rsl_rl_multi`
  environment) and re-ran the same `HimInterceptExportWrapper` forward pass
  against `onnxruntime`'s output over 20 random `[1, 710]` inputs.
- **Max abs diff (PyTorch vs. ONNX): `1.14e-05`** — float32 noise floor, as
  expected for a pure architecture/precision check (no learned-behavior
  claim implied).

## What this report does NOT verify

Per the export skill's own scope: this confirms the exported graph computes
what the checkpoint was trained to compute (architecture + numerical
parity), not that the checkpoint's learned behavior is good. No live
rollout / success-rate probe was run as part of this export. No TensorBoard
scalar history was available for this run directory (no `tfevents` file
present — this run appears to have logged to WandB only), so no training
trend is reported here.

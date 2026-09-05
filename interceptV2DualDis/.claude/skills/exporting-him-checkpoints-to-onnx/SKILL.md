---
name: exporting-him-checkpoints-to-onnx
description: Use when exporting an interceptV2DualDis (HIM/multi-discriminator) checkpoint to ONNX for deployment, or when a deployed intercept policy shows degraded/erratic behavior (e.g. hesitant leg lifts, jerky motion) that might trace back to the export rather than the checkpoint itself. Covers why a naive actor-only export breaks HIM checkpoints, how observation scale silently drifts between training and deploy, and how to verify an export without the training stack.
---

# Exporting HIM Checkpoints to ONNX

## Overview

`export_onnx.py` has two behaviors depending on checkpoint architecture:

- **Plain flat-MLP actor** (`state_dict` has only `actor.*`): the whole policy
  IS the actor, so exporting `actor.*` alone as `obs -> actions` is correct.
- **HIM architecture** (`state_dict` also has `history_encoder.*`,
  `ball_estimator.*`, `region_estimator.*` -- `him_actor_critic.py`,
  `HimActorCritic`): the actor's real input is `obs_current(71) +
  history_latent(16) + estimate_ball(4) + region_arg(1) = 92`, built at
  inference by running the raw 710-dim obs history through all three
  estimator heads first. **Exporting only `actor.*` for a HIM checkpoint
  produces an ONNX with a 92-dim input and the three estimator sub-networks
  missing entirely** -- nothing outside the graph can supply the
  history_latent/estimate_ball/region_arg values, and 92 isn't even a
  multiple of 71, so any deploy consumer's history-length auto-detection
  (dividing the input dim by the single-step obs size) fails outright.

This was a real, deployed bug: a HIM checkpoint (`interceptV2DualDis`,
`model_18500`) was hand-exported without accounting for this, shipped to a
robot deploy repo, and only the estimator-omission half was initially
caught. The current `export_onnx.py` detects HIM architecture from the
`state_dict` and exports the FULL forward pass (all three estimator heads +
actor) behind a single raw `obs_history` input instead.

## The Second, Subtler Bug: Observation Scale Drift

Training applies fixed per-term scale multipliers before an observation ever
reaches the network (`goalkeeper_env_cfg.py`'s `ObservationTermCfg(scale=...)`,
e.g. `base_ang_vel*0.25`, `joint_vel*0.05` -- the 2026-07-20 "obs-scaling
audit" fix, matching G1's real `obs_scales`). This is applied transparently
by mjlab's `ObservationManager` -- nothing in `play.py` or any training
script touches it explicitly, so it's easy to forget it exists at all when
writing an exporter or a deploy consumer by hand.

**Any deploy consumer that reconstructs observations manually (i.e. not
through mjlab) has to replicate this scale itself, and nothing forces it to
stay in sync when the training-side scale changes.** This bit a downstream
robot deploy repo directly: their C++ observation-building code was written
against an OLDER checkpoint trained before the 2026-07-20 scale fix (when
those terms were effectively unscaled), and nobody updated it when a newer
HIM checkpoint trained under the new scaled convention was deployed --
`joint_vel` arrived ~20x its trained-on magnitude, most visible exactly
during fast motions like a swing-leg lift.

**The fix belongs in the export, not in either side's hand-written code.**
`export_onnx.py`'s HIM path reads the live `env_cfg.observations["actor"]
.terms[name].scale` for every term and bakes it into the ONNX graph as its
first op (a fixed elementwise multiply). The exported graph then accepts
RAW, unscaled sensor values -- the deploy side never needs to know what
scale convention a given training run used, and a future scale change on
the training side is automatically reflected in the next export without
anyone hand-updating a second implementation.

## Recipe (what the HIM export path actually does)

1. Detect HIM architecture: `any(k.startswith("history_encoder.") for k in
   state_dict)`.
2. Pick the right task_id for metadata/env loading --
   `Mjlab-BeyondAMP-Goalkeeper-T1-MultiDisc`, NOT the plain
   `Mjlab-BeyondAMP-Goalkeeper-T1` (wrong env_cfg entirely: no
   region/ball-estimator obs groups, no HIM actor-critic).
3. Derive every HIM hyperparameter instead of hardcoding it:
   - `history_latent_dim`, `estimate_ball_dim`, `num_regions` = output size
     of the last layer of `history_encoder`/`ball_estimator`/
     `region_estimator` (from the checkpoint's own weight shapes).
   - Per-term obs order/size/scale = `env_cfg.observations["actor"].terms`
     (dict order = training concat order) + `observation_manager
     .group_obs_term_dim["actor"]` for sizes.
   - `history_length` = `env_cfg.observations["actor"].history_length`.
4. **Assert** `obs_current_dim + history_latent_dim + estimate_ball_dim + 1
   == actor's actual first-layer input dim` (read from the checkpoint
   weights). If a future obs-term or HIM hyperparameter change makes these
   disagree, this fails loudly at export time instead of silently producing
   a broken ONNX -- this is the whole point of deriving instead of
   hardcoding.
5. Build the wrapper module: multiply raw `obs_history` by the derived scale
   vector, run the three estimator heads, slice `obs_current` as the newest
   per-term frame of the (already-scaled) `obs_history` (valid at
   deployment -- see below), concat, run `actor`. Export with a single
   input.

### Why slicing `obs_current` from `obs_history` is valid at deployment

`goalkeeper_multidisc_amp_cfg.py` builds `obs_current` as a genuinely
separate, independently-sampled observation group (`actor_current`,
`history_length=0`) from `obs_history` (`actor`, `history_length=10`) --
during TRAINING these can differ slightly because each group draws its own
noise sample (`enable_corruption`). At DEPLOYMENT there is no synthetic
per-group noise at all (real sensor values only), so `obs_current` and
`obs_history`'s newest frame are numerically identical -- meaning a single
raw `obs_history` input is sufficient; no second `obs_current` input tensor
is needed in the exported graph. Confirmed against
`him_amp_on_policy_runner.py`'s own comment: *"mjlab flattens history
per-term rather than per-frame"* -- i.e. history is stored as
`[term1_t0..t9, term2_t0..t9, ...]`, not `[frame0_all_terms,
frame1_all_terms, ...]`, so "newest per term" means the LAST `term_size`
elements of each term's own `history_length * term_size` sub-block, not a
single contiguous slice at the end of the whole vector.

## Verifying an Export Without the Training Stack

You don't need mjlab/a GPU to sanity-check the ARCHITECTURE half of an
export (only the metadata-attachment step needs the real env):

1. Load the raw checkpoint with `torch.load(..., weights_only=False)`. This
   will fail with `ModuleNotFoundError: No module named 'rsl_rl_amp'`
   outside the training environment (the pickle references training-only
   classes for the optimizer/discriminator state). Fix: install a permissive
   stub module via `sys.meta_path` that returns a throwaway class for any
   attribute access, so unpickling completes without needing the real
   package. Only `model_state_dict`'s tensors are needed for architecture
   work -- discard everything else and remove the stub from `sys.modules`
   immediately after, before doing anything else in the process (a stub
   module left registered can interfere with unrelated later imports, e.g.
   `torch.onnx`'s own module introspection during `torch.onnx.export`).
2. Rebuild each submodule (`history_encoder`, `ball_estimator`,
   `region_estimator`, `actor`) directly from `state_dict` weights with
   plain `nn.Linear`/`nn.Sequential` -- don't need the real `HimActorCritic`
   class, just its layer shapes and activations (ReLU for the three
   estimator heads, ELU for actor -- `him_actor_critic.py`).
3. Export, then `onnx.checker.check_model()` + parity-check
   `onnxruntime` output against the raw PyTorch forward pass over ~20 random
   inputs (expect agreement to float32 noise, ~1e-5).
4. If baking in a scale multiplier, cross-check it's mathematically
   equivalent to applying the scale externally before the OLD (unscaled-input)
   graph: `new_model(raw) == old_model(scale * raw)` should match to `0.0`
   exactly (it's a pure linear op composed with the rest of the graph, no
   reason for it to differ at all, let alone by float noise).
5. What you CANNOT verify this way: `get_base_metadata()` (needs the real
   mjlab env to build `ManagerBasedRlEnv`), and the checkpoint's actual
   learned behavior (a policy can be architecturally correct and still
   perform badly -- this only proves the graph computes what the checkpoint
   was trained to compute, not that what it was trained to compute is good).

## Quick Reference

| Question | How to answer it |
|---|---|
| Does this checkpoint need the HIM export path? | `any(k.startswith("history_encoder.") for k in state_dict)` |
| What's the actor's real input composition? | `obs_current(one-step obs dim) + history_latent + estimate_ball + region_arg(1)` -- assert this equals the actor's actual first-layer input dim from the checkpoint weights, don't assume |
| Does the exported graph need a separate `obs_current` input? | No -- slice it from `obs_history`'s newest per-term frame; valid at deployment (no synthetic per-group noise on real sensors) even though training samples the two groups independently |
| Where does per-term obs scale come from, and should I hardcode it? | `env_cfg.observations["actor"].terms[name].scale` -- always derive live, never hardcode, since training-side scale values have changed before (2026-07-20 audit) without any deploy-side code being updated to match |
| Can I verify an export without the training stack? | Architecture + parity: yes (permissive-unpickler + manual weight rebuild + onnxruntime parity check). Metadata + learned-behavior quality: no, needs the real env / a live rollout |
| Wrong task_id for HIM metadata? | Use `Mjlab-BeyondAMP-Goalkeeper-T1-MultiDisc`, not the plain `-T1` task |

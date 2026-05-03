# Divergence from Upstream (Humanoid-Goalkeeper)

This document tracks substantive changes where the Booster T1 adaptation deviates from the G1 tracking task pipeline.

## 2026-05-03 — Remove motion-dependent rewards from play config

**File:** `my_mjlab_project/src/my_mjlab_project/tasks/goalkeeper_env_cfg.py`

**What:** Added removal of 6 motion-dependent reward terms in `goalkeeper_play_env_cfg()`:
`motion_global_root_pos`, `motion_global_root_ori`, `motion_body_pos`,
`motion_body_ori`, `motion_body_lin_vel`, `motion_body_ang_vel`.

**Why:** The play environment disables the motion command manager (sets command to None)
to allow autonomous inference without a motion file. However, these upstream
reward functions attempt to access command attributes (`anchor_pos_w`, etc.),
causing AttributeError crashes on the first environment step.

**Impact:** Fixes critical bug that prevented play/inference mode from running.
Goalkeeper rewards (`eereach`, `catch_success`, etc.) remain active.

## 2026-05-02 — Remove motion-reference observations from actor/critic

**File:** `my_mjlab_project/src/my_mjlab_project/tasks/goalkeeper_env_cfg.py`

**What:** Removed `command`, `motion_anchor_pos_b`, `motion_anchor_ori_b` from
the actor and critic observation groups in `goalkeeper_env_cfg()`.

**Why:** The upstream G1 tracking task treats reference motion as an explicit
input to the policy. This requires a motion file at inference time. For an
autonomous goalkeeper agent that decides its own response based on ball
position, the motion reference must not appear in the observation space.
The 6 motion files are retained for RSI (reference state initialisation) and
style-shaping rewards during training only.

**Impact:** All checkpoints trained before 2026-05-02 are incompatible with the
new observation space (actor dim changed). Full retrain required.
